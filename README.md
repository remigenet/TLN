# TLN: Temporal Linear Network

This is the original implementation of the [paper](https://arxiv.org/abs/2410.21448).

TLN (Temporal Linear Network) is a neural network architecture that extends the capabilities of linear models while maintaining interpretability and computational efficiency. TLN is designed to effectively capture both temporal and feature-wise dependencies in multivariate time series data. Our approach is a variant of TSMixer that maintains strict linearity throughout its architecture. It removes activation functions, introduces specialized kernel initializations, and incorporates dilated convolutions to handle various time scales, all while preserving the linear nature of the model. Unlike transformer-based models that may lose  temporal information due to their permutation-invariant nature, TLN explicitly preserves and leverages the temporal structure of the input data. A key innovation of TLN is its ability to compute an equivalent linear model, offering a level of interpretability not found in more complex architectures like TSMixer. This feature allows for seamless conversion between the full TLN model and its linear equivalent, facilitating both training flexibility and inference optimization.

The implementation is made in Keras3 in a backend-agnostic way, to be compatible with TensorFlow, JAX, and Torch.  The package also includes implementations of comparison models from the literature, such as CLinear, NLinear, DLinear from ["Are Transformers Effective for Time Series Forecasting?"](https://arxiv.org/abs/2205.13504) and [TSMixer: An All-MLP Architecture for Time Series Forecasting](https://arxiv.org/abs/2303.06053) in keras3.

## Installation
Install TLN directly from PyPI:
```bash
pip install temporal_linear_network
```

## Key Features
- Fully linear architecture that maintains interpretability
- Ability to extract equivalent linear weights for analysis
- Consistent performance across varying sequence lengths
- Compatible with all Keras 3 backends (TensorFlow, JAX, PyTorch)
- Includes implementations of comparison models (CLinear, DLinear, NLinear, TSMixer)


## Usage
Here's a basic example demonstrating how to use TLN for time series forecasting:

```python
import keras
from tln import TLN

# Create and configure the model
model = TLN(
    output_len=prediction_horizon,
    output_features=1,
    hidden_layers=2,
    use_convolution=True
)

# Build and compile the model
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='mean_squared_error',
    jit_compile=True
)

# Train the model
history = model.fit(
    X_train, 
    y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.2
)

# Make predictions
predictions = model.predict(X_test)

# Extract equivalent linear weights for interpretation
weights, bias = model.compute_linear_equivalent_weights()
```

## Advanced Features

### Linear Model Conversion
TLN can be converted to its equivalent linear form for analysis:

```python
# Get the equivalent linear model
linear_model = model.get_linear_equivalent_model()
```

### Comparison Models
The package also includes implementations of other linear architectures:

#### Linear Models from "Are Transformers Effective for Time Series Forecasting?"
```python
from tln import CLinear, NLinear, DLinear

# Classic Linear model
model = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    CLinear(pred_len=prediction_horizon, individual=False),
    keras.layers.Dense(1),  # For multivariate to univariate predictions
    keras.layers.Flatten()
])

# Normalized Linear model
model = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    NLinear(pred_len=prediction_horizon, individual=False),
    keras.layers.Dense(1),  # For multivariate to univariate predictions
    keras.layers.Flatten()
])

# Decomposition Linear model
model = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    DLinear(pred_len=prediction_horizon, individual=False),
    keras.layers.Dense(1),  # For multivariate to univariate predictions
    keras.layers.Flatten()
])
```

#### TSMixer Model
```python
from tln import TSMixer

# TSMixer with single block
model = keras.Sequential([
    keras.layers.Input(shape=input_shape),
    TSMixer(
        input_shape=input_shape,
        pred_len=prediction_horizon,
        norm_type='L',
        activation='relu',
        n_block=1,
        dropout=0.0,
        ff_dim=5,
        target_slice=None
    ),
    keras.layers.Dense(1),  # For multivariate to univariate predictions
    keras.layers.Flatten()
])
```

Please cite our work if you use this repo:

```
@article{genet2024tln,
  title={A Temporal Linear Network for Time Series Forecasting},
  author={Genet, Remi and Inzirillo, Hugo},
  journal={arXiv preprint arXiv:2410.21448},
  year={2024}
}
```

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg