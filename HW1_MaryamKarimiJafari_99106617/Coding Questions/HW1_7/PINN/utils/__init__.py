'''
Some utilities.

Modules
-------
wave : Wave equation setup.
model : Model components.
pinn : Physics-informed NN.
train : PINN training.
vis : Visualization tools.

'''

from .wave import Wave_1D_Equation

from .model import make_fc_model

from .pinn import PINN

from .train import test_pinn, train_pinn

from .vis import make_colors

