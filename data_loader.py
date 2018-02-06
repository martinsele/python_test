from fileinput import filename

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from matplotlib.finance import candlestick_ohlc



def read_fx_data_from_file(fileName, formatSpec):
    '''
    Read data from file given and return as pandas dataframe with datetime index and OpenHighLowClose data
    :param fileName:
    :param formatSpec:
    :return: DataFrame read from the file and its label
    '''
    dataR = pd.read_csv(fileName, index_col=1)
    dataR.index = pd.to_datetime(dataR.index, format=formatSpec)
    dataR.sort_index(inplace=True)
    label = dataR['Name'][0]
    dataR.drop('Name', axis=1, inplace=True)
    return dataR, label


def read_fx_data(dirStr):
    '''
    Read various FX history data from given directory and merges them together
    :param dirStr:
    :return: array of DataFrames each for given input data; and labels of the individual DataFrames as dictionary
    '''

    formatSpec1 = '%Y-%m-%d %H:%M:%S'
    formatSpec2 = '%m/%d/%Y %H:%M'

    dirN = os.fsencode(dirStr)
    data = []
    labels = {}
    fileIdx = 0

    for file in os.listdir(dirN):
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            try:
                fileData, label = read_fx_data_from_file(os.path.join(dirStr, filename), formatSpec=formatSpec1)
            except:
                fileData, label = read_fx_data_from_file(os.path.join(dirStr, filename), formatSpec=formatSpec2)

            labels[fileIdx] = label
            fileIdx += 1
            data.append(fileData)

    # Drop columns where not all data are present
    scatData = pd.concat([df['Close'] for df in data], axis=1)
    for df in data:
        df.drop(scatData.index[scatData.isnull().any(1).nonzero()[0]], errors='ignore', inplace=True)

    return data, labels



######  VISUALIZE CANDLES  #######

def visualizeCandles(data, ticks):
    dataStart = data.index[0]
    dataEnd = data.index[ticks]

    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    dayFormatter = DateFormatter('%d')      # e.g., 12

    quotes = data.loc[dataStart:dataEnd, :].reset_index().values
    if len(quotes) == 0:
        raise SystemExit

    fig, ax = plt.subplots(1,1, figsize=(15,8))
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)
    ax.xaxis.set_minor_formatter(dayFormatter)

    #plot_day_summary(ax, quotes, ticksize=3)
    quotes[:, 0] = date2num(quotes[:, 0])
    candlestick_ohlc(ax, quotes, width=0.05)

    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()
    return



if __name__ == '__main__':
    data, labels = read_fx_data('./data')

    print(data)
    print(labels)
    # eurUsd = read_fx_data_from_file('./data/fxhistoricaldata_EURUSD_hour.csv', formatSpec='%Y-%m-%d %H:%M:%S')
    # eurGbp = read_fx_data_from_file('./data/fxhistoricaldata_EURGBP_hour.csv', formatSpec='%Y-%m-%d %H:%M:%S')
    # eurJpy = read_fx_data_from_file('./data/fxhistoricaldata_EURJPY_hour.csv', formatSpec='%Y-%m-%d %H:%M:%S')
    # eurChf = read_fx_data_from_file('./data/fxhistoricaldata_EURCHF_hour.csv', formatSpec='%m/%d/%Y %H:%M')
    #
    # ###### CLEAN NAN VALUES ######
    # startDate = max([eurUsd.index[0], eurGbp.index[0], eurJpy.index[0], eurChf.index[0]])
    # eurUsd = eurUsd.loc[startDate:].dropna()
    # eurGbp = eurGbp.loc[startDate:].dropna()
    # eurJpy = eurJpy.loc[startDate:].dropna()
    # eurChf = eurChf.loc[startDate:].dropna()
    #
    # plt.plot(eurUsd['Close'], label='USD')
    # plt.plot(eurChf['Close'], label='CHF')
    # plt.plot(eurGbp['Close'], label='GBP')
    # # plt.plot(eurJpy['Close'], label='JPY')
    # plt.legend()
    # plt.show()
    #
    # scatData = pd.concat([eurUsd['Close'], eurGbp['Close'], eurJpy['Close'], eurChf['Close']], axis=1)
    # scatData.columns = ['usd', 'gbp', 'jpy', 'chf']
    #
    # # visualizeCandles(eurUsd, 50)
    #
    # print(scatData.pct_change().corr())
    # #         usd         gbp         jpy         chf
    # # usd     1.000000    0.331700    0.376447    0.342612
    # # gbp     0.331700    1.000000    0.105364    0.167030
    # # jpy     0.376447    0.105364    1.000000    0.410838
    # # chf     0.342612    0.167030    0.410838    1.000000
    #
    # pd.plotting.scatter_matrix(scatData.pct_change(), alpha=0.2, diagonal='kde')
    # plt.show()
