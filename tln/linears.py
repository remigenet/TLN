"""
Linear Models from the paper: Are Transformers Effective for Time Series Forecasting? 

It is a keras3 implementation, the original implementation (in pytorch) is available at: https://github.com/cure-lab/LTSF-Linear
"""
import keras
from keras.layers import Layer, Dense, AveragePooling1D


class CLinear(Layer):
    """
    C for classic
    Just one Linear layer
    """
    def __init__(self, pred_len, individual=False, **kwargs):
        super().__init__(**kwargs)
        self.pred_len = pred_len
        self.Linear = Dense(self.pred_len)

    def build(self, input_shape):
        
        super().build(input_shape)

    def call(self, x):

        x = keras.ops.transpose(x, [0, 2, 1])
        x = self.Linear(x)
        x = keras.ops.transpose(x, [0, 2, 1])
        
        return x # [Batch, Output length, Channel]

class NLinear(Layer):
    """
    N for Normalized
    Just one Linear layer
    """
    def __init__(self, pred_len, individual=False, **kwargs):
        super().__init__(**kwargs)
        self.pred_len = pred_len
        self.individual = individual
        self.Linear = None

    def build(self, input_shape):
        seq_len, channels = input_shape[1], input_shape[2]
        if self.individual:
            self.Linear = [Dense(self.pred_len) for _ in range(channels)]
        else:
            self.Linear = Dense(self.pred_len)
        super().build(input_shape)

    def call(self, x):
        # x: [Batch, Input length, Channel]
        last_seq = x[:,-1:,:]
        x = x - last_seq
        if self.individual:
            outputs = [layer(x[:,:,i]) for i, layer in enumerate(self.Linear)]
            x = keras.ops.stack(outputs, axis=-1)
        else:
            x = keras.ops.transpose(x, [0, 2, 1])
            x = self.Linear(x)
            x = keras.ops.transpose(x, [0, 2, 1])
        
        return x + last_seq# [Batch, Output length, Channel]


class MovingAvg(Layer):
    def __init__(self, kernel_size, stride=1, **kwargs):
        super(MovingAvg, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride

    def build(self, input_shape):
        self.avg = AveragePooling1D(pool_size=self.kernel_size, strides=self.stride, padding='valid')

    def call(self, x):
        front = keras.ops.repeat(x[:, 0:1, :], (self.kernel_size - 1) // 2, axis=1)
        end = keras.ops.repeat(x[:, -1:, :], (self.kernel_size - 1) // 2, axis=1)
        x = keras.ops.concatenate([front, x, end], axis=1)
        return self.avg(x)

class SeriesDecomp(Layer):
    def __init__(self, kernel_size, **kwargs):
        super(SeriesDecomp, self).__init__(**kwargs)
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def call(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(Layer):
    def __init__(self, pred_len, individual=False, **kwargs):
        super().__init__(**kwargs)
        self.pred_len = pred_len
        self.individual = individual
        self.kernel_size = 5
        self.decomposition = SeriesDecomp(self.kernel_size)

    def build(self, input_shape):
        self.seq_len = input_shape[1]
        self.channels = input_shape[2]
        
        if self.individual:
            self.linear_seasonal = [Dense(self.pred_len) for _ in range(self.channels)]
            self.linear_trend = [Dense(self.pred_len) for _ in range(self.channels)]
        else:
            self.linear_seasonal = Dense(self.pred_len)
            self.linear_trend = Dense(self.pred_len)

        super().build(input_shape)

    def call(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = keras.ops.transpose(seasonal_init, (0, 2, 1))
        trend_init = keras.ops.transpose(trend_init, (0, 2, 1))
        if self.individual:
            seasonal_output = keras.ops.stack([self.linear_seasonal[i](seasonal_init[:, i, :]) 
                                               for i in range(self.channels)], axis=1)
            trend_output = keras.ops.stack([self.linear_trend[i](trend_init[:, i, :]) 
                                            for i in range(self.channels)], axis=1)
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)
        
        return keras.ops.transpose(seasonal_output + trend_output, (0, 2, 1))