import pandas as pd
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Data:
    def __init__(self, dir_path):
        self.data =               None
        self.files =              None
        self.dirPath =            dir_path
        self.traindataSet =       None
        self.validDataSet =       None
        self.testDataSet =        None
        self.scaler =             None
        self.input_feat =         None
        self.output_feat =        None

    def load_dataFromCsv(self, _columns=None):
        if _columns != None and type(_columns) is not list:
            print('load_datafromCSV(): unexpected type of columns argument')
            raise ValueError

        self.files = glob('{}/*.csv'.format(self.dirPath))
        if self.files is not None and len(self.files) > 0:
            for file in self.files:
                if _columns is not None:
                    df = pd.read_csv(file, usecols=_columns)
                else:
                    df = pd.read_csv(file)
                name = os.path.basename(file)[:-4]
                df = df.set_index('Date')
                df = df.rename(columns={col: name + '_' + col for col in df.columns if col != 'Date'})
                fileName = os.path.basename(file)[:-4]
                df[fileName] = (df['{}_Low'.format(fileName)] + df['{}_High'.format(fileName)]) / 2
                df = df.drop(columns=['{}_Low'.format(fileName), '{}_High'.format(fileName)])
                if df is not None:
                    if self.data is None:
                       self.data = df
                    else:
                        self.data = pd.concat([self.data, df], axis=1)
            self.data = self.data.dropna()
            self.data['Date'] = self.data.index.values
            self.data = self.data.reset_index(drop=True)

    def plotfeature(self, ax, split_index, feat:str, title:str):
        ax.plot(self.data.iloc[:split_index].index.values, self.data[feat].iloc[:split_index], label='Train_{}'.format(feat))
        ax.plot(self.data.iloc[split_index:].index.values, self.data[feat].iloc[split_index:], label='Validation_{}'.format(feat))
        ax.legend()
        ax.set_title(title)
        return ax


    def CreateTrainValidationDataSet(self, input_feat:list, output_feat:list, ratio=0.5, show=False):
        if type(input_feat) != list or type(output_feat) != list:
            print('CreateTrinValidationDataSet(): wrong argument type')
            raise ValueError

        if len(input_feat) == 0 or len(output_feat) == 0:
            print('CreateTrinValidationDataSet(): input_feat or output_feat has zero length')
            raise ValueError

        missing_in = [i for i in input_feat if i not in self.data.columns]
        missing_out = [i for i in output_feat if i not in self.data.columns]
        if len(missing_in) > 0:
            print('inputs: {} not present within data'.format(missing_in))
            raise ValueError

        if len(missing_out) > 0:
            print('outputs {} not preset within data'.format(missing_out))
            raise ValueError

        split_index = int(np.round(len(self.data.index.values) * ratio))
        self.traindataSet = {'input': self.data[input_feat].iloc[0:split_index], 'output': self.data[output_feat].iloc[0:split_index]}
        self.validDataSet = {'input': self.data[input_feat].iloc[split_index:], 'output': self.data[output_feat].iloc[split_index:]}

        if show is True:
            fig_input, axs_inputs = plt.subplots(len(input_feat), figsize=(16,9))
            fig_output, axs_outputs = plt.subplots(len(output_feat), figsize=(16,9))

            if type(axs_outputs) != np.ndarray:
                axs_outputs = [axs_outputs]
            if type(axs_inputs) != np.ndarray:
                axs_inputs = [axs_inputs]

            for ax_input, feat in zip(axs_inputs, input_feat):
                ax_input = self.plotfeature(ax_input, split_index, feat, 'Inputs')
            for ax_output, feat in zip(axs_outputs, output_feat):
                ax_output = self.plotfeature(ax_output, split_index, feat, 'Outputs')
            plt.show()


    def scaleData(self):
        "Call before Data is split into data sets(before CreateTrainValidationDataSet())"
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data[[i for i in self.data if i != 'Date']].to_numpy())
        self.data[[i for i in self.data if i != 'Date']] = self.scaler.transform(self.data[[i for i in self.data if i != 'Date']].to_numpy())
        print('Scaleing DONE')


def PrepareData(show=False):
    data = Data('Data')
    data.load_dataFromCsv(['Date', 'Low', 'High'])
    data.scaleData()
    data.CreateTrainValidationDataSet(ratio=0.8, show=show, input_feat=['SNP', 'Dow_Jones_Industrial', 'EUR_USD'],
                                      output_feat=['CrudeOil'])
    print('data was loaded')
    return data



if __name__ == "__main__":
    data = PrepareData(True)
