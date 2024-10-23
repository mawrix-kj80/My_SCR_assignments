'''Model components.'''

import torch.nn as nn


def make_fc_model(num_inputs,
                  num_outputs,
                  num_hidden=None,
                  activation='tanh'):
    '''
    Create FC model.

    Parameters
    ----------
    num_inputs : int
        Number of inputs.
    num_outputs : int
        Number of outputs.
    num_hidden : int or list thereof
        Number of hidden neurons.
    activation : str
        Activation function type.

    '''

    layers = []
    input_dim = num_inputs

    if num_hidden is None:
        num_hidden = [16, 16, 16]

    activation_functions = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'leaky_relu': nn.LeakyReLU()
    }

    for hidden_units in num_hidden:
        layers.append(nn.Linear(input_dim, hidden_units))
        if activation.lower() in activation_functions:
            layers.append(activation_functions[activation])
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        input_dim = hidden_units

    layers.append(nn.Linear(input_dim, num_outputs))

    model = nn.Sequential(*layers)

    return model

