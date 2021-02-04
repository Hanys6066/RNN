from matplotlib import pyplot as plt
import numpy as np
from Dataloader import PrepareData

def ferr_mse(input:np.ndarray, expected:np.ndarray,  der:bool):
    if der is False:
        return 1/2 * np.power(expected - input, 2.0)
    else:
        return expected - input

def fc_sigmoid(input:np.ndarray, der:bool):
    if der is False:
        return 1 / (1 + np.exp(-input))
    else:
        return np.exp(-input) / np.power(1 + np.exp(-input), 2.0)

def fc_RELU(input:np.ndarray, der:bool):
    if der is False:
        output = input if input > 0 else 0
    else:
        output = 1.0 if input > 0 else 0
    return  output

def fc_LRELU(input:np.ndarray, der:bool, alpha=0.01):
    if der is False:
        output = input if input > 0 else alpha * input
    else:
        output = 1.0 if input > 0 else alpha
    return output


def check_arguments(func):
    def __inner(*args, **kwargs):
        if type(args[1]) is not int or args[1] <= 0:
            print('argument input_dim has to be type unsigned integer greater than zero')
            raise ValueError
        if type(args[2]) is not list:
            print('argument structure has to be type list')
            raise ValueError
        if len(kwargs.keys()) > 0:
            for _arg in kwargs.keys():
                if _arg == "fc_init":
                    if kwargs[_arg] != 'rand':
                        print('Only rand initialization implemented')
                        raise NotImplementedError
                elif _arg == 'fc_err':
                    if kwargs[_arg] != 'MSE':
                        print('Only MSE error function implemented')
                        raise NotImplementedError

        return func(*args, **kwargs)
    return __inner

def checkInputOutPutSeq(func):
    def __inner(*args, **kwargs):
        if type(args[1]) is not list or type(args[1]) is not np.ndarray:
            print('argument train_input_seq has to be type np.ndarray or list')
            raise ValueError
        if type(args[1]) is not list or type(args[1]) is not np.ndarray:
            print('argument train_output_seq has to be type np.ndarray or list')
            raise ValueError

        if len(args[1]) != len(args[1]):
            print('length of input and output sequence has to be same')
            raise ValueError

        return func(*args, **kwargs)

    return __inner


class RNN:
    @check_arguments
    def __init__(self, input_dim, structure, fc_err="MSE",  fc_init = None, alpha=0.8):
        self.__structure =          [len(i) for i in structure]
        self.__alpha =              alpha
        self.__input_dim =          input_dim
        self.__fc_error =           ferr_mse if fc_err == 'MSE' else None
        self.__fc_act =             structure
        self.__fc_init  =           fc_init
        self.__weights =            None
        self.__weights_h =          None
        self.__weightsErr=          None
        self.__weights_hErr =       None
        self.__ht =                 []
        self.__pacErr=              None
        self.__NeuronOut =          None
        self.__createnetwork()

    def __createnetwork(self):
        _tmp = np.append([self.__input_dim], self.__structure)

        if self.__fc_init is None:
            self.__weights = [np.zeros((act, prev)) for prev, act in zip(_tmp[:-1], _tmp[1:])]
            self.__weights_h = [np.zeros((n , n)) for n in self.__structure]
        elif self.__fc_init == "rand":
            self.__weights = [np.random.rand(act, prev) for  prev, act in zip(_tmp[:-1], _tmp[1:])]
            self.__weights_h = [np.random.rand(n, n) for n in self.__structure]
        else:
            raise NotImplementedError

        self.__weightsErr = [np.zeros(n) for n in _tmp]
        self.__weights_hErr = [np.zeros(n) for n in self.__structure]
        self.__NeuronOut = [np.zeros(n) for n in self.__structure]
        self.__yt_1 = [np.zeros(n) for n in self.__structure]
        self.__ht.append([np.zeros(n) for n in self.__structure]) #initial value of h_t is set to zeroes
        self.__pacErr = [np.zeros(n) for n in self.__structure]

    def forward(self, _input:np.ndarray):
        _h = None
        _y = None
        _y_1 = None
        _hmat = []
        _x = None
        for _layer in range(len(self.__structure)):
            if _layer == 0:
                _x = _input

            _h = self.__weights_h[_layer].dot(self.__yt_1[_layer]) + self.__weights[_layer].dot(_x)
            _y = np.array([self.__fc_act[_layer][i](_h[i], False) for i in range(len(_h))])
            self.__yt_1[_layer] = _y
            _hmat.append(_h)
        return _y, _hmat

    def predict(self,input_seq:np.ndarray):
        _y = None
        for t in range(len(input_seq)):
            _input = input_seq[t]
            _y, _h = self.forward(_input)
            self.__ht.append(np.array(_h))
        return _y

    def train(self, input:np.ndarray, output:np.ndarray, seq_length:int, batch_size:int):
        if type(input) != np.ndarray or type(output) != np.ndarray:
            print('train(): wrong argument type')
            raise ValueError
        if type(batch_size) != int or batch_size <= 0:
            print('train() wrong type or value of argument batch_size')
            raise ValueError

        if type(seq_length) != int or seq_length <= 0:
            print('train(): wrong type or value of argument seq_length')
            raise ValueError

        for _layer in range(len(self.__structure) -1, -1, -1):
            if _layer == len(self.__structure) - 1:
                pass
            else:
                pass
    @checkInputOutPutSeq
    def backpropagation(self, train_input_seq_batch, train_output_seq_batch):
        _E_y =      None
        _Delta =    None
        _y_h =      None
        _ht_hk =    None
        for train_input_seq, train_output_seq in zip(train_input_seq_batch, train_output_seq_batch):
            _net_out = self.predict(train_input_seq)
            for t in range(len(train_input_seq)):
                _input = train_input_seq[t]
                _output = train_output_seq[t]

                for _layer in range(len(self.__structure)):
                    if _layer == len(self.__structure) - 1:
                        _E_y = self.__fc_error(_net_out[t], _output)
                        _y_h = np.array([self.__fc_act[_layer][i](self.__ht_1[_layer][i], True) for i in range(len(self.__ht_1))])
                        _Delta = _E_y * _y_h
                        _ht_hk = np.prod([self.__weights_h[_layer].transpose().dot(self.__ht[t_1][_layer]) for t_1 in range(len(train_input_seq))], axis=1)
                    else:
                        _Delta = self.__weights[_layer + 1].transpose().dot(self.__pacErr)
                        _y_h = np.array([self.__fc_act[_layer][i](self.__ht_1[_layer][i], True) for i in range(len(self.__ht_1))])




if __name__ == '__main__':
    input_size = 3
    structure = [[fc_LRELU, fc_LRELU, fc_sigmoid], [fc_RELU, fc_RELU, fc_RELU], [fc_LRELU]]
    rnn = RNN(input_size, structure, fc_init='rand', fc_err="MSE")
    data = PrepareData(False)
    res = rnn.predict(data.traindataSet['input'].to_numpy()[0:10])
    print('Done')