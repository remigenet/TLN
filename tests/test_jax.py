import os
BACKEND = 'jax' 
os.environ['KERAS_BACKEND'] = BACKEND

import pytest
import keras
from keras import ops
from keras import backend
from keras import random
from tln import TLN, NLinear, DLinear, CLinear, TSMixer
from tln.linears import MovingAvg, SeriesDecomp
from tln.tsmixer import TSMixer_RevNorm, TSMixer_res_block

def generate_random_tensor(shape):
    return random.normal(shape=shape, dtype=backend.floatx())

def test_tln_basic():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    output_len = 5
    
    tln_layer = TLN(output_len=output_len, output_features=4)
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output_sequence = tln_layer(input_sequence)
    assert output_sequence.shape == (batch_size, output_len, 4), f"Expected shape {(batch_size, output_len, 4)}, but got {output_sequence.shape}"

def test_tln_flatten_output():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    output_len = 5
    
    tln_layer = TLN(output_len=output_len, output_features=4, flatten_output=True)
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output_sequence = tln_layer(input_sequence)
    assert output_sequence.shape == (batch_size, output_len * 4), f"Expected shape {(batch_size, output_len * 4)}, but got {output_sequence.shape}"

def test_tln_hidden_layers():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    output_len = 5
    
    tln_layer = TLN(output_len=output_len, output_features=4, hidden_layers=3, hidden_sizes=[16, 12, 8])
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output_sequence = tln_layer(input_sequence)
    assert output_sequence.shape == (batch_size, output_len, 4), f"Expected shape {(batch_size, output_len, 4)}, but got {output_sequence.shape}"

def test_tln_without_convolution():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    output_len = 5
    
    tln_layer = TLN(output_len=output_len, output_features=4, use_convolution=False)
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output_sequence = tln_layer(input_sequence)
    assert output_sequence.shape == (batch_size, output_len, 4), f"Expected shape {(batch_size, output_len, 4)}, but got {output_sequence.shape}"

def test_tln_linear_equivalent():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    output_len = 5
    
    tln_layer = TLN(output_len=output_len, output_features=4)
    # Build the model first
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    _ = tln_layer(input_sequence)
    
    equivalent_weights, equivalent_bias = tln_layer.compute_linear_equivalent_weights()
    
    assert equivalent_weights.shape == (time_steps, features, output_len, 4), f"Expected shape {(time_steps, features, output_len, 4)}, but got {equivalent_weights.shape}"
    assert equivalent_bias.shape == (output_len, 4), f"Expected shape {(output_len, 4)}, but got {equivalent_bias.shape}"

def test_tln_linear_equivalent_model():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    output_len = 5
    
    tln_layer = TLN(output_len=output_len, output_features=4)
    # Build the model first
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    _ = tln_layer(input_sequence)
    
    linear_model = tln_layer.get_linear_equivalent_model()
    
    tln_output = tln_layer(input_sequence)
    linear_output = linear_model(input_sequence)
    
    # Calculate the maximum absolute difference
    max_diff = ops.max(ops.abs(tln_output - linear_output))

    tolerance = 1e-5
    
    assert max_diff <= tolerance, f"Maximum difference ({max_diff}) exceeds tolerance ({tolerance})"

def test_tln_in_model():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    output_len = 5
    
    inputs = keras.Input(shape=(time_steps, features))
    tln_layer = TLN(output_len=output_len, output_features=4)
    outputs = tln_layer(inputs)
    model = keras.Model(inputs, outputs)
    
    model.compile(optimizer='adam', loss='mse')
    
    x = generate_random_tensor((batch_size, time_steps, features))
    y = generate_random_tensor((batch_size, output_len, 4))
    
    # Fit for a few epochs
    history = model.fit(x, y, epochs=3, batch_size=16, verbose=0)
    
    # Check if loss is decreasing
    assert history.history['loss'][0] > history.history['loss'][-1], "Loss should decrease during training"
    
    # Test prediction
    test_input = generate_random_tensor((1, time_steps, features))
    prediction = model.predict(test_input)
    assert prediction.shape == (1, output_len, 4), f"Expected prediction shape {(1, output_len, 4)}, but got {prediction.shape}"

def test_tln_serialization():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    output_len = 5
    
    tln_layer = TLN(output_len=output_len, output_features=4)
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    _ = tln_layer(input_sequence)  # Build the layer
    
    config = tln_layer.get_config()
    new_tln_layer = TLN.from_config(config)
    
    assert tln_layer.output_len == new_tln_layer.output_len
    assert tln_layer.output_features == new_tln_layer.output_features
    assert tln_layer.flatten_output == new_tln_layer.flatten_output
    assert tln_layer.hidden_layers == new_tln_layer.hidden_layers
    assert tln_layer.use_convolution == new_tln_layer.use_convolution

def test_tln_fit_predict():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 10, 8
    output_len = 5
    
    inputs = keras.Input(shape=(time_steps, features))
    tln_layer = TLN(output_len=output_len, output_features=4)
    outputs = tln_layer(inputs)
    model = keras.Model(inputs, outputs)
    
    model.compile(optimizer='adam', loss='mse')
    
    x_train = generate_random_tensor((batch_size * 2, time_steps, features))
    y_train = generate_random_tensor((batch_size * 2, output_len, 4))
    
    # Fit for a few epochs
    history = model.fit(x_train, y_train, epochs=5, batch_size=16, validation_split=0.2, verbose=0)
    
    # Check if loss is decreasing
    assert history.history['loss'][0] > history.history['loss'][-1], "Loss should decrease during training"
    
    # Test prediction
    x_test = generate_random_tensor((batch_size, time_steps, features))
    predictions = model.predict(x_test)
    assert predictions.shape == (batch_size, output_len, 4), f"Expected prediction shape {(batch_size, output_len, 4)}, but got {predictions.shape}"

def test_clinear():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 100, 8
    pred_len = 24

    clinear = CLinear(pred_len=pred_len)
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output = clinear(input_sequence)

    assert output.shape == (batch_size, pred_len, features), f"Expected shape {(batch_size, pred_len, features)}, but got {output.shape}"

def test_nlinear():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 100, 8
    pred_len = 24

    # Test with individual=False
    nlinear = NLinear(pred_len=pred_len, individual=False)
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output = nlinear(input_sequence)
    assert output.shape == (batch_size, pred_len, features), f"Expected shape {(batch_size, pred_len, features)}, but got {output.shape}"

    # Test with individual=True
    nlinear_individual = NLinear(pred_len=pred_len, individual=True)
    output_individual = nlinear_individual(input_sequence)
    assert output_individual.shape == (batch_size, pred_len, features), f"Expected shape {(batch_size, pred_len, features)}, but got {output_individual.shape}"

def test_dlinear():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 100, 8
    pred_len = 24

    # Test with individual=False
    dlinear = DLinear(pred_len=pred_len, individual=False)
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output = dlinear(input_sequence)
    assert output.shape == (batch_size, pred_len, features), f"Expected shape {(batch_size, pred_len, features)}, but got {output.shape}"

    # Test with individual=True
    dlinear_individual = DLinear(pred_len=pred_len, individual=True)
    output_individual = dlinear_individual(input_sequence)
    assert output_individual.shape == (batch_size, pred_len, features), f"Expected shape {(batch_size, pred_len, features)}, but got {output_individual.shape}"

def test_moving_avg():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 100, 8
    kernel_size = 5

    moving_avg = MovingAvg(kernel_size=kernel_size)
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output = moving_avg(input_sequence)

    assert output.shape == (batch_size, time_steps, features), f"Expected shape {(batch_size, time_steps, features)}, but got {output.shape}"

def test_series_decomp():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 100, 8
    kernel_size = 5

    series_decomp = SeriesDecomp(kernel_size=kernel_size)
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    res, moving_mean = series_decomp(input_sequence)

    assert res.shape == (batch_size, time_steps, features), f"Expected shape {(batch_size, time_steps, features)}, but got {res.shape}"
    assert moving_mean.shape == (batch_size, time_steps, features), f"Expected shape {(batch_size, time_steps, features)}, but got {moving_mean.shape}"
    
    # Check if the decomposition is correct
    reconstructed = res + moving_mean

    max_diff = ops.max(ops.abs(input_sequence - reconstructed))

    tolerance = 1e-5
    
    assert max_diff <= tolerance, "Decomposition should be reversible"

def test_tsmixer_revnorm():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 100, 8
    
    rev_norm = TSMixer_RevNorm(axis=-2)
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    
    # Test normalization
    normalized, mean, stdev = rev_norm(input_sequence, mode='norm')
    assert normalized.shape == (batch_size, time_steps, features), f"Expected shape {(batch_size, time_steps, features)}, but got {normalized.shape}"
    
    # Test denormalization
    denormalized = rev_norm(normalized, mode='denorm', mean=mean, stdev=stdev)
    assert denormalized.shape == (batch_size, time_steps, features), f"Expected shape {(batch_size, time_steps, features)}, but got {denormalized.shape}"

    max_diff = ops.max(ops.abs(input_sequence - denormalized))

    tolerance = 1e-5
    
    assert max_diff <= tolerance, "Normalization should be reversible"

def test_tsmixer_res_block():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 100, 8
    ff_dim = 16
    
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output = TSMixer_res_block(input_sequence, norm_type='L', activation='relu', dropout=0.1, ff_dim=ff_dim)
    
    assert output.shape == (batch_size, time_steps, features), f"Expected shape {(batch_size, time_steps, features)}, but got {output.shape}"

def test_tsmixer():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 100, 8
    pred_len = 24
    
    model = TSMixer(
        input_shape=(time_steps, features),
        pred_len=pred_len,
        norm_type='L',
        activation='relu',
        n_block=2,
        dropout=0.1,
        ff_dim=16
    )
    
    input_sequence = generate_random_tensor((batch_size, time_steps, features))
    output = model(input_sequence)
    
    assert output.shape == (batch_size, pred_len, features), f"Expected shape {(batch_size, pred_len, features)}, but got {output.shape}"

def test_tsmixer_training():
    assert keras.backend.backend() == BACKEND
    batch_size, time_steps, features = 32, 100, 8
    pred_len = 24
    
    model = TSMixer(
        input_shape=(time_steps, features),
        pred_len=pred_len,
        norm_type='L',
        activation='relu',
        n_block=2,
        dropout=0.1,
        ff_dim=16
    )
    
    model.compile(optimizer='adam', loss='mse')
    
    x = generate_random_tensor((batch_size, time_steps, features))
    y = generate_random_tensor((batch_size, pred_len, features))
    
    history = model.fit(x, y, epochs=1, verbose=0)
    
    assert 'loss' in history.history, "Model should train successfully"