"""
Linear Models from the paper: TSMixer: An All-MLP Architecture for Time Series Forecasting

It is a keras3 implementation, the original implementation (in tensorflow) is available at: https://github.com/google-research/google-research/tree/master/tsmixer/tsmixer_basic
"""
import keras
from keras.layers import Layer, Dense, Dropout, LayerNormalization, BatchNormalization


@keras.utils.register_keras_serializable(package="tln", name="TSMixer_RevNorm")
class TSMixer_RevNorm(Layer):
    """Reversible Instance Normalization."""
    
    def __init__(self, axis, eps=1e-5, affine=True, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.eps = eps
        self.affine = affine
        
    def build(self, input_shape):
        if self.affine:
            self.affine_weight = self.add_weight(
              name='affine_weight', shape=(input_shape[-1],), initializer='ones'
            )
            self.affine_bias = self.add_weight(
              name='affine_bias', shape=(input_shape[-1],), initializer='zeros'
            )
    
    def call(self, x, mode = 'norm', mean = None, stdev = None, target_slice=None):
        if mode == 'norm':
            mean = keras.ops.stop_gradient(
                keras.ops.mean(x, axis=self.axis, keepdims=True)
            )
            stdev = keras.ops.stop_gradient(
                keras.ops.sqrt(
                    keras.ops.var(x, axis=self.axis, keepdims=True) + self.eps
                    )
            )
            normalized = self._normalize(x, mean, stdev)
            return normalized, mean, stdev
        elif mode == 'denorm':
            return self._denormalize(x, mean, stdev, target_slice)
        else:
            raise NotImplementedError
        

    def _normalize(self, x, mean, stdev):
        x = x - mean
        x = x / stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, mean, stdev, target_slice=None):
        if self.affine:
            x = x - self.affine_bias[target_slice]
            x = x / self.affine_weight[target_slice]
        if target_slice is not None:
            x = x * stdev[..., target_slice]
            x = x + mean[..., target_slice]
        else:
            x = x * stdev
            x = x + mean
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'eps': self.eps,
            'affine': self.affine,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.utils.register_keras_serializable(package="tln", name="TSMixerResBlock")
class TSMixerResBlock(Layer):
    """Residual block of TSMixer."""

    def __init__(self, norm_type, activation, dropout, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.norm_type = norm_type
        self.activation = activation
        self.dropout_rate = dropout
        self.ff_dim = ff_dim

    def build(self, input_shape):
        seq_len = input_shape[-2]
        n_features = input_shape[-1]

        self.norm = (
            LayerNormalization(axis=[-2, -1])
            if self.norm_type == 'L'
            else BatchNormalization()
        )

        # Temporal Linear
        self.temporal_dense = Dense(seq_len, activation=self.activation)
        self.temporal_dropout = Dropout(self.dropout_rate)

        # Feature Linear
        self.feature_dense1 = Dense(self.ff_dim, activation=self.activation)
        self.feature_dropout1 = Dropout(self.dropout_rate)
        self.feature_dense2 = Dense(n_features)
        self.feature_dropout2 = Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs):
        # Temporal Linear
        x = self.norm(inputs)
        x = keras.ops.transpose(x, [0, 2, 1])  # [Batch, Channel, Input Length]
        x = self.temporal_dense(x)
        x = keras.ops.transpose(x, [0, 2, 1])  # [Batch, Input Length, Channel]
        x = self.temporal_dropout(x)
        res = x + inputs

        # Feature Linear
        x = self.norm(res)
        x = self.feature_dense1(x)  # [Batch, Input Length, FF_Dim]
        x = self.feature_dropout1(x)
        x = self.feature_dense2(x)  # [Batch, Input Length, Channel]
        x = self.feature_dropout2(x)
        return x + res

    def get_config(self):
        config = super().get_config()
        config.update({
            'norm_type': self.norm_type,
            'activation': self.activation,
            'dropout': self.dropout_rate,
            'ff_dim': self.ff_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.utils.register_keras_serializable(package="tln", name="TSMixer")
class TSMixer(keras.Model):
    """TSMixer with Reversible Instance Normalization model."""

    def __init__(self, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice=None, **kwargs):
        super().__init__(**kwargs)
        if target_slice is not None:
            raise NotImplementedError("target_slice is not implemented yet and only set to match the original implementation. Will cause crash here.")
        self.pred_len = pred_len
        self.norm_type = norm_type
        self.activation = activation
        self.n_block = n_block
        self.dropout_rate = dropout
        self.ff_dim = ff_dim
        self.target_slice = target_slice

        self.rev_norm = TSMixer_RevNorm(axis=-2)
        self.res_blocks = [TSMixerResBlock(norm_type, activation, dropout, ff_dim) for _ in range(n_block)]
        self.output_dense = Dense(pred_len)

    def call(self, inputs):
        x = inputs  # [Batch, Input Length, Channel]
        x, mean, stdev = self.rev_norm(x, mode='norm')
        for block in self.res_blocks:
            x = block(x)
        if self.target_slice:
            x = x[:, :, self.target_slice]
        x = keras.ops.transpose(x, [0, 2, 1])  # [Batch, Channel, Input Length]
        x = self.output_dense(x)  # [Batch, Channel, Output Length]
        x = keras.ops.transpose(x, [0, 2, 1])  # [Batch, Output Length, Channel]
        x = self.rev_norm(x, mean=mean, stdev=stdev, mode='denorm', target_slice=self.target_slice)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'pred_len': self.pred_len,
            'norm_type': self.norm_type,
            'activation': self.activation,
            'n_block': self.n_block,
            'dropout': self.dropout_rate,
            'ff_dim': self.ff_dim,
            'target_slice': self.target_slice,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
