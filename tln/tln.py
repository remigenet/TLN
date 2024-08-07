"""
Temporal Linear Network (TLN) original implementation in Keras.
"""

import keras
from keras.layers import Layer, Dense, Conv1D, Flatten, Input
from typing import List, Optional, Union


class SequentialDense(Layer):
    """
    A custom layer that performs sequential dense operations.

    This layer applies dense transformations along both the feature and time dimensions
    of the input sequence.

    Args:
        end_features (int): The number of output features.
        end_sequence_length (int): The length of the output sequence.
        activation (str or callable, optional): Activation function to use. Defaults to 'linear'.
        it_kernel_initializer (keras.initializers.Initializer, optional): Initializer for the input-time kernel. Defaults to RandomUniform(0,1).
        f_kernel_initializer (keras.initializers.Initializer, optional): Initializer for the feature kernel. Defaults to RandomUniform(0,1).
        ot_kernel_initializer (keras.initializers.Initializer, optional): Initializer for the output-time kernel. Defaults to RandomUniform(0,1).
        kernel_regularizer (keras.regularizers.Regularizer, optional): Regularizer function applied to the kernels. Defaults to None.
        **kwargs: Additional keyword arguments passed to the parent Layer class.
    """
    def __init__(self, end_features, end_sequence_length, activation='linear',
                 it_kernel_initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
                 f_kernel_initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
                 ot_kernel_initializer=keras.initializers.RandomUniform(minval=0.0, maxval=1.0),
                 kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.end_features = end_features
        self.activation = keras.activations.get(activation)
        self.end_sequence_length = end_sequence_length
        self.dense_features = Dense(end_features)
        self.dense_times = Dense(end_sequence_length)
        self.it_kernel_initializer = keras.initializers.get(it_kernel_initializer)
        self.f_kernel_initializer = keras.initializers.get(f_kernel_initializer)
        self.ot_kernel_initializer = keras.initializers.get(ot_kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        _, seq_len, n_features = input_shape
        self.it_kernel = self.add_weight(
            shape=(1, seq_len, 1),
            name="it_kernel",
            initializer=self.it_kernel_initializer,
            regularizer=self.kernel_regularizer,
        )
        self.f_kernel = self.add_weight(
            shape=(1, 1, self.end_features),
            name="f_kernel",
            initializer=self.f_kernel_initializer,
            regularizer=self.kernel_regularizer,
        )
        self.ot_kernel = self.add_weight(
            shape=(1, self.end_sequence_length, 1),
            name="ot_kernel",
            initializer=self.ot_kernel_initializer,
            regularizer=self.kernel_regularizer,
        )
        self.dense_features.build(input_shape)
        self.dense_times.build((input_shape[0], self.end_features, seq_len))
        super().build(input_shape)

    def call(self, inputs):
        x = inputs * self.it_kernel
        x = self.dense_features(x)
        x = x * self.f_kernel
        x = keras.ops.transpose(x, (0, 2, 1))
        x = self.dense_times(x)
        x = keras.ops.transpose(x, (0, 2, 1))
        x = self.ot_kernel * x
        return self.activation(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.end_sequence_length, self.end_features)

    def get_config(self):
        config = super().get_config()
        config.update({
            'end_features': self.end_features,
            'end_sequence_length': self.end_sequence_length,
            'activation': keras.activations.serialize(self.activation),
            'it_kernel_initializer': keras.initializers.serialize(self.it_kernel_initializer),
            'f_kernel_initializer': keras.initializers.serialize(self.f_kernel_initializer),
            'ot_kernel_initializer': keras.initializers.serialize(self.ot_kernel_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['activation'] = keras.activations.get(config['activation'])
        config['it_kernel_initializer'] = keras.initializers.get(config['it_kernel_initializer'])
        config['f_kernel_initializer'] = keras.initializers.get(config['f_kernel_initializer'])
        config['ot_kernel_initializer'] = keras.initializers.get(config['ot_kernel_initializer'])
        config['kernel_regularizer'] = keras.regularizers.get(config['kernel_regularizer'])
        return cls(**config)

class LinearLayer(Layer):
    """
    A custom layer that applies a linear transformation using pre-computed weights and biases.

    Args:
        equivalent_weights (np.ndarray): Pre-computed weights for the linear transformation.
        equivalent_bias (np.ndarray): Pre-computed bias for the linear transformation.
        **kwargs: Additional keyword arguments passed to the parent Layer class.
    """
    def __init__(self, equivalent_weights, equivalent_bias, **kwargs):
        super().__init__(**kwargs)
        self.equivalent_weights = self.add_weight(
            shape=equivalent_weights.shape,
            initializer=keras.initializers.Constant(equivalent_weights),
            trainable=False,
            name='equivalent_weights'
        )
        self.equivalent_bias = self.add_weight(
            shape=equivalent_bias.shape,
            initializer=keras.initializers.Constant(equivalent_bias),
            trainable=False,
            name='equivalent_bias'
        )

    def call(self, inputs):
        return keras.ops.einsum('bij,ijkl->bkl', inputs, self.equivalent_weights) + self.equivalent_bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "equivalent_weights": keras.ops.convert_to_numpy(self.equivalent_weights),
            "equivalent_bias": keras.ops.convert_to_numpy(self.equivalent_bias)
        })
        return config

class TLN(keras.Model):
    """
    A custom model that applies a series of linear transformations to the input.

    Args:
        output_len (int): The length of the output sequence.
        output_features (int, optional): The number of output features. Defaults to 1.
        flatten_output (bool, optional): Whether to flatten the output. Defaults to False.
        hidden_layers (int, optional): The number of hidden layers. Defaults to 1.
        use_convolution (bool, optional): Whether to use convolution layers. Defaults to True.
        hidden_sizes (List[int], optional): List of hidden layer sizes. If not provided, sizes will be calculated at build time.
        kernel_size (Union[int, List[int]], optional): Kernel size(s) for convolution layers. Defaults to 5.
        **kwargs: Additional keyword arguments passed to the parent Model class.
    """
    def __init__(self,
                 output_len: int,
                 output_features: int = 1,
                 flatten_output: bool = False,
                 hidden_layers: int = 1,
                 use_convolution: bool = True,
                 hidden_sizes: Optional[List[int]] = None,
                 kernel_size: Optional[Union[int, List[int]]] = 5,
                 **kwargs):
        super().__init__(**kwargs)
        self.output_len = output_len
        self.output_features = output_features
        self.flatten_output = flatten_output
        self.hidden_layers = hidden_layers
        self.use_convolution = use_convolution
        self.hidden_sizes = hidden_sizes
        self.kernel_size = kernel_size

        # Process kernel_size
        if isinstance(kernel_size, (list, tuple)):
            if len(kernel_size) != hidden_layers:
                raise ValueError(f"Received {len(kernel_size)} values in kernel_size while having {hidden_layers} hidden layers. Please provide the same number of values or only provide one integer if you want to use the same everywhere.")
            self.kernel_size = kernel_size
        else:
            self.kernel_size = [kernel_size] * hidden_layers

        # Layers will be defined in the build method
        self.layers_list = []

    def build(self, input_shape):
        _, self.input_seq_len, self.input_features = input_shape

        # Calculate hidden sizes if not provided
        if self.hidden_sizes is None:
            self.hidden_sizes = [
                int(self.input_features - (i + 1) * (self.output_features - self.input_features) / self.hidden_layers)
                for i in range(self.hidden_layers)
            ]
        elif len(self.hidden_sizes) != self.hidden_layers:
            raise ValueError(f"Received {len(self.hidden_sizes)} values in hidden_sizes while having {self.hidden_layers} hidden layers. Please provide the same number of values.")

        # Define layers
        for i in range(self.hidden_layers):
            if i == self.hidden_layers - 1:
                hidden_size = self.output_features
            else:
                hidden_size = self.hidden_sizes[i]
            
            if self.use_convolution:
                self.layers_list.append(SequentialDense(
                    end_features=hidden_size,
                    end_sequence_length=int(self.output_len + self.kernel_size[i] - 1)
                ))
                self.layers_list.append(Conv1D(
                    filters=self.output_features,
                    kernel_size=self.kernel_size[i],
                    padding='valid'
                ))
            else:
                self.layers_list.append(SequentialDense(
                    end_features=hidden_size,
                    end_sequence_length=self.output_len
                ))

        if self.flatten_output:
            self.layers_list.append(Flatten())

        # Build all layers
        for layer in self.layers_list:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)

        super().build(input_shape)

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x

    def compute_linear_equivalent_weights(self):
        if not self.built:
            raise ValueError("The model must be built before calling this method.")
        
        backend = keras.backend.backend()
        if backend == 'jax':
            return self._compute_linear_equivalent_weights_jax()
        elif backend == 'torch':
            return self._compute_linear_equivalent_weights_torch()
        else:  # TensorFlow or any other backend
            return self._compute_linear_equivalent_weights_tf()

    def _compute_linear_equivalent_weights_jax(self):
        equivalent_weights = keras.ops.zeros((self.input_seq_len, self.input_features, self.output_len, self.output_features))
        equivalent_bias = self.call(keras.ops.zeros((1, self.input_seq_len, self.input_features)))[0]
    
        for i in range(self.input_seq_len):
            for j in range(self.input_features):
                input_tensor = keras.ops.zeros((1, self.input_seq_len, self.input_features))
                input_tensor = input_tensor.at[0, i, j].set(1.0)
                output = self.call(input_tensor)
                equivalent_weights = equivalent_weights.at[i, j].set(output[0] - equivalent_bias)
    
        return equivalent_weights, equivalent_bias

    def _compute_linear_equivalent_weights_torch(self):
        equivalent_weights = keras.ops.zeros((self.input_seq_len, self.input_features, self.output_len, self.output_features))
        equivalent_bias = self.call(keras.ops.zeros((1, self.input_seq_len, self.input_features)))[0]
    
        for i in range(self.input_seq_len):
            for j in range(self.input_features):
                input_tensor = keras.ops.zeros((1, self.input_seq_len, self.input_features))
                input_tensor = input_tensor.cpu().numpy()
                input_tensor[0, i, j] = 1.0
                input_tensor = keras.ops.convert_to_tensor(input_tensor)
                output = self.call(input_tensor)
                equivalent_weights[i, j] = output[0] - equivalent_bias
    
        return equivalent_weights, equivalent_bias

    def _compute_linear_equivalent_weights_tf(self):
        equivalent_weights = keras.ops.zeros((self.input_seq_len, self.input_features, self.output_len, self.output_features)).numpy()
        equivalent_bias = self.call(keras.ops.zeros((1, self.input_seq_len, self.input_features)))[0]

        for i in range(self.input_seq_len):
            for j in range(self.input_features):
                input_tensor = keras.ops.zeros((1, self.input_seq_len, self.input_features))
                input_tensor = input_tensor.numpy()
                input_tensor[0, i, j] = 1.0
                input_tensor = keras.ops.convert_to_tensor(input_tensor)
                output = self.call(input_tensor)
                equivalent_weights[i, j] = output[0] - equivalent_bias
                
        equivalent_weights = keras.ops.convert_to_tensor(equivalent_weights)

        return equivalent_weights, equivalent_bias


    def get_linear_equivalent_model(self):
        if not self.built:
            raise ValueError("The model must be built before calling this method.")
        equivalent_weights, equivalent_bias = self.compute_linear_equivalent_weights()
        
        input_layer = Input(shape=(self.input_seq_len, self.input_features))
        linear_layer = LinearLayer(equivalent_weights, equivalent_bias)
        output = linear_layer(input_layer)
        
        linear_model = keras.Model(inputs=input_layer, outputs=output)
        
        # Freeze the model
        linear_model.trainable = False
        
        return linear_model

    def get_config(self):
        config = super().get_config()
        config.update({
            'output_len': self.output_len,
            'output_features': self.output_features,
            'flatten_output': self.flatten_output,
            'hidden_layers': self.hidden_layers,
            'use_convolution': self.use_convolution,
            'hidden_sizes': self.hidden_sizes,
            'kernel_size': self.kernel_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_len, self.output_features)