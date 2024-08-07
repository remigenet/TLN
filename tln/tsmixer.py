"""
Linear Models from the paper: TSMixer: An All-MLP Architecture for Time Series Forecasting

It is a keras3 implementation, the original implementation (in tensorflow) is available at: https://github.com/google-research/google-research/tree/master/tsmixer/tsmixer_basic
"""
import keras
from keras.layers import Layer, Dense, Permute, Dropout, LayerNormalization, BatchNormalization 


class TSMixer_RevNorm(Layer):
    """Reversible Instance Normalization."""
    
    def __init__(self, axis, eps=1e-5, affine=True):
        super().__init__()
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

def TSMixer_res_block(inputs, norm_type, activation, dropout, ff_dim):
    """Residual block of TSMixer."""
    
    norm = (
      LayerNormalization(axis=[-2, -1])
      if norm_type == 'L'
      else BatchNormalization()
    )
    
    # Temporal Linear
    x = norm(inputs)
    x = Permute([2, 1])(x)  # [Batch, Channel, Input Length]
    x = Dense(x.shape[-1], activation=activation)(x)
    x = Permute([2, 1])(x)  # [Batch, Input Length, Channel]
    x = Dropout(dropout)(x)
    res = x + inputs
    
    # Feature Linear
    x = norm(res)
    x = Dense(ff_dim, activation=activation)(x)  # [Batch, Input Length, FF_Dim]
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
    x = Dropout(dropout)(x)
    return x + res


def TSMixer(
        input_shape,
        pred_len,
        norm_type,
        activation,
        n_block,
        dropout,
        ff_dim,
        target_slice = None,
    ):
    """Build TSMixer with Reversible Instance Normalization model."""
    if target_slice is not None:
        raise NotImplementedError("target_slice is not implemented yet and only set to match the original implementation. Will cause crash here.")
    inputs = keras.Input(shape=input_shape)
    x = inputs  # [Batch, Input Length, Channel]
    rev_norm = TSMixer_RevNorm(axis=-2)
    x, mean, stdev = rev_norm(x, mode = 'norm')
    for _ in range(n_block):
        x = TSMixer_res_block(x, norm_type, activation, dropout, ff_dim)
    if target_slice:
        x = x[:, :, target_slice]
    
    x = Permute([2, 1])(x)  # [Batch, Channel, Input Length]
    x = Dense(pred_len)(x)  # [Batch, Channel, Output Length]
    outputs = Permute([2, 1])(x)  # [Batch, Output Length, Channel])
    outputs = rev_norm(outputs, mean = mean, stdev = stdev, mode='denorm', target_slice=target_slice)
    return keras.Model(inputs, outputs)
