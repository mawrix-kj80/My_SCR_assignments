'''Heat equation setup.'''

import torch
import torch.nn as nn


class Wave_1D_Equation(nn.Module):
    '''
    Simple 1D wave equation problem.

    Summary
    -------
    This class represents a simple problem based on the 1D wave equation.
    It is posed in a way that allows the PDE to have an analytical solution.
    Dirichlet boundary conditions are imposed such that the
    wave displacement at both ends of length is kept at zero.
    The initial displacement along the space interval is described by a sine function.
    There is no external source of disturbance and force within the length.

    Note that the wave function setup here is implemented as a PyTorch module,
    because this allows for a smooth integration into the PINN framework.

    Parameters
    ----------
    c : float
        wave speed coefficient.
    length : float
        Length of the space interval.
    maxtime : float
        End of the time interval.
    n : int
        Determines the initial wave displacement.

    '''

    def __init__(self,
                 c=1.0,
                 length=1.0,
                 maxtime=1.0,
                 n=1):

        super().__init__()

        c = abs(c)
        length = abs(length)
        maxtime = abs(maxtime)
        n = abs(int(n))

        self.register_buffer('c', torch.as_tensor(c))
        self.register_buffer('length', torch.as_tensor(length))
        self.register_buffer('maxtime', torch.as_tensor(maxtime))
        self.register_buffer('n', torch.as_tensor(n))

    @property
    def sqrt_lambda(self):
        return self.n * torch.pi / self.length

    @staticmethod
    def boundary_condition(t):
        '''Return zeros as boundary condition.'''
        return torch.zeros_like(t)

    def initial_condition(self, x):
        '''Evaluate the initial condition.'''
        out = torch.sin(self.sqrt_lambda * x)
        return out, self.sqrt_lambda * self.c * out

    def exact_solution(self, t, x):
        '''Compute the exact solution.'''

        spatial_solution = torch.sin(self.sqrt_lambda * x)
        temporal_solution = torch.cos(self.sqrt_lambda * self.c * t) + torch.sin(self.sqrt_lambda * self.c * t)
                               

        u = spatial_solution * temporal_solution
        return u

    def forward(self, t, x):
        return self.exact_solution(t=t, x=x)

