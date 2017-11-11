# Mushroom Neural Network
## Quickstart
### Interactive Neural Network Training
1. Navigate to `neural_network/`
2. Ensure the `presentation_mode` parameter in `run_interactive.m` is `false`
3. Run `run_interactive.m` in Octave or MATLAB
4. Using the interactive console, one can choose to load a prexisting configuration or train a new one

### Presentation Mode
1. Navigate to `neural_network/`
2. Ensure the `presentation_mode` parameter in `run_interactive.m` is `true`
3. Run `run_interactive.m` in Octave or MATLAB

### Training Multiple Neural Networks
1. Navigate to `neural_network/`
2. Run `run_interactive.m` in Octave or MATLAB
3. Once training is complete, trained networks will show up in `neural_network/trained_networks`

## Overview
### `datasets/`
- Contains the datasets used to train and validate the neural network

### `docs/`
- Planning documents used in the process of creating the neural network

### `prototypes/`
- Initial prototypes of the neural network

### `neural_network/`
#### `train.m`
- Function to train neural network

#### `autorun.m`
- Trains multiple neural networks and saves them uniquely by training parameters and timestamp

#### `predict.m`
- Uses input and expected output to check neural network's ability to predict

#### `find_error.m`
- Compares the neural network to the validation set `correction-set.csv` and computes its accuracy

### `neural_network/trained_networks`
- Directory of all trained networks
- Can be loaded via `run_interactive` to explore stats and accuracy
