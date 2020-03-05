import numpy as np
import pandas as pd
import os



# output csv
# including cleaned data
def generate_csv(outputfilepath, df):
    df.to_csv(outputfilepath, sep=',', encoding='utf-8')



def resampleCSV(df,filepath):
    df['date'] = pd.to_datetime(df['date'],dayfirst=True)   ##use day first for GOV download csv
    df.set_index('date',inplace=True)
    # df[df < 0] = 0

    df.replace(0, np.nan).fillna(method='ffill')


    newcsv = df.resample('10min').mean()
    newcsv = newcsv.interpolate(method='linear', axis=0).bfill()
    print(newcsv.describe())
    # filedir, name = os.path.split(filepath)
    filename, file_extension = os.path.splitext(filepath)
    # outputcsv = os.path.join(filedir, name + '_resample' + '.csv')
    outputcsv = os.path.join(filename + '_resample' + '.csv')
    newcsv.to_csv(outputcsv)


filepath = './data/WL_Deeral.csv'


df = pd.read_csv(filepath)

resampleCSV(df,filepath)