"""
This is the entry point for the model on SageMaker.

When the data is passed to sagemaker for inference, the following functions are called.

data -> input_fn(data) -> parsed_data \
                                       -> predict_fn(parsed_data, model) -> inference -> output_fn(inference) -> lambda -> eagle.io
        model_fn(dir)  -> model       /
"""
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
import json
import numpy as np
import torch
import logging
import os


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def numpy_to_tvar(x):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Variable(torch.from_numpy(x).type(torch.FloatTensor).to(device))


class Encoder(nn.Module):
    def __init__(self, input_dim, enc_hid_dim, dec_hid_dim, enc_layers,
                 dec_layers, dropout_p):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.dropout_p = dropout_p

        self.input_linear = nn.Linear(self.input_dim, self.enc_hid_dim)
        self.lstm = nn.LSTM(input_size=self.enc_hid_dim,
                            hidden_size=self.enc_hid_dim,
                            num_layers=self.enc_layers,
                            bidirectional=True)
        self.output_linear = nn.Linear(self.enc_hid_dim * 2, self.dec_hid_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input):
        embedded = self.dropout(torch.tanh(self.input_linear(input)))

        outputs, (hidden, cell) = self.lstm(embedded)

        hidden = torch.tanh(
            self.output_linear(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # for different number of decoder layers
        hidden = hidden.repeat(self.dec_layers, 1, 1)

        return outputs, (hidden, hidden)


class Global_Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Global_Attention, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(self.enc_hid_dim * 2 + self.dec_hid_dim,
                              self.dec_hid_dim)
        self.v = nn.Parameter(torch.rand(self.dec_hid_dim))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # only pick up last layer hidden
        hidden = torch.unbind(hidden, dim=0)[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        energy = energy.permute(0, 2, 1)

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        attention = torch.bmm(v, energy).squeeze(1)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, enc_hid_dim, dec_hid_dim, dec_layers,
                 dropout_p, attention):
        super(Decoder, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dec_layers = dec_layers
        self.dropout_p = dropout_p
        self.attention = attention

        self.input_dec = nn.Linear(self.output_dim, self.dec_hid_dim)
        self.lstm = nn.LSTM(input_size=self.enc_hid_dim * 2 + self.dec_hid_dim,
                            hidden_size=self.dec_hid_dim,
                            num_layers=self.dec_layers)
        self.out = nn.Linear(
            self.enc_hid_dim * 2 + self.dec_hid_dim + self.dec_hid_dim,
            self.output_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(0)
        input = torch.unsqueeze(input, 2)

        embedded = self.dropout(torch.tanh(self.input_dec(input)))

        a = self.attention(hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        lstm_input = torch.cat((embedded, weighted), dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        input_dec = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        output = self.out(torch.cat((output, weighted, input_dec), dim=1))

        return output.squeeze(1), (hidden, cell), a


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[1]
        max_len = trg.shape[0]

        outputs = torch.zeros(max_len, batch_size,
                              self.decoder.output_dim).to(self.device)

        decoder_attn = torch.zeros(max_len, src.shape[0]).to(self.device)

        encoder_outputs, (hidden, cell) = self.encoder(src)

        # only use y initial y
        output = src[-1, :, 0]

        for t in range(0, max_len):
            output, (hidden,
                     cell), attn_weight = self.decoder(output, hidden, cell,
                                                       encoder_outputs)

            outputs[t] = output.unsqueeze(1)

            teacher_force = random.random() < teacher_forcing_ratio

            output = (trg[t].view(-1) if teacher_force else output)

        return outputs


def predict_ts(model, X_test, max_gap_size=6, BATCH_SIZE=1):
    """
    Generate a prediction for the given data and scaler.

    @param model: (Seq2Seq) The model that will generate the inferences.
    @param X_test: (np.array[float]) A numpy array containing the data and with the dimensions (1,72,1)
    @param gap_size: (int) The number of data points in the gap.

    @return
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        x_test = np.transpose(X_test, [1, 0, 2])

        empty_y_tensor = torch.zeros(max_gap_size, BATCH_SIZE, 1).to(device)

        x_test_tensor = numpy_to_tvar(x_test)

        output = model(x_test_tensor, empty_y_tensor, 0)
        output = output.view(-1)

        # scalar
        output_numpy = output.cpu().data.numpy().reshape(-1)

    return output_numpy.tolist()

def init_model():
    """
    Initialises the model used to make inferences on the data.

    Returns:
        The model used to make this inference. (In this case it's a Seq2Seq model).

    Returned model will have the following function called on it:
        model.load_state_dict(torch.load(os.path.join(model_dir, 'checkpoint.pt'), map_location=device))

        And will be passed to the predict_ts(model, ...) function.

    """
    # model hyperparameters
    INPUT_DIM = 1
    OUTPUT_DIM = 1
    ENC_HID_DIM = 25
    DEC_HID_DIM = 25
    ENC_DROPOUT = 0.2
    DEC_DROPOUT = 0.2
    ECN_Layers = 2
    DEC_Layers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model
    glob_attn = Global_Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, ECN_Layers, DEC_Layers,
                  ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_Layers,
                  DEC_DROPOUT, glob_attn)

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    return model




"""

Note that this code will be stored on S3, separate from the rest of the program

as it is only required by SageMaker.  As such, several imports will show as errors.

"""


#================== SERVING AND PREDICTION FUNCTIONS =====================#
#  Used by Sagemaker


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model()
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    logger.info("Model successfully loaded.")
    return model


def input_fn(serialized_input_data, content_type='application/json'):
    """
    Used by SageMaker to parse the input to the model.
    """
    if content_type == 'application/json':
        logger.info("Input successfully parsed.")
        return json.loads(serialized_input_data)
    raise Exception('Requested unsupported ContentType: ' + content_type + '. Requires: application/json')


def output_fn(prediction_output, accept='application/json'):
    """
    Converts the outputted json to bytes
    """
    if accept == 'application/json':
        # logger.info("Output successfully parsed.")
        return json.dumps(prediction_output).encode('utf-8')
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def predict_fn(gap_data, model):
    """
    Generates a prediction with the model.

    Example Format of gap_data packets:
    {
        "length": 6, # Length of gap
        "input": [1, 2, 3, 4, 5, 6, 7, 8]
    }

    Outputted as:
    {
        "length": 6,
        "input": [1, 2, 3, 4, 5, 6, 7, 8]
        "inference": [8, 7, 6, 5, 4, 3]
    }
    """
    logger.info("Commencing model inference.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test = np.asarray(gap_data['input']).reshape((1, len(gap_data['input']), 1))

    gap_size = gap_data['length']

    output_list = predict_ts(model, X_test, gap_size, BATCH_SIZE=1)
    gap_data['inference'] = output_list
    logger.info("Model inference complete. Gap was size {}, input was size {}.".format(gap_data["length"],
                                                                                       len(gap_data['input'])))

    return gap_data


#===================== End of Serving and Prediction Functions ======================#
