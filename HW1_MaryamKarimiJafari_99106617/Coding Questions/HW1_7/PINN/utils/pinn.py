import torch
import torch.nn as nn
from .wave import Wave_1D_Equation
from .model import make_fc_model

class PINN(nn.Module):
    '''
    PINN model class.

    Summary
    -------
    A PINN module for the simple 1D wave equation PDE problem is implemented.
    It contains a feedforward NN that predicts the behaviour of a wave
    as a function of both time and a spatial coordinate.
    Moreover, it allows for computing the regression and physics loss.
    The former loss penalizes predictions that deviate from actual data,
    while the latter measures violations of the physical constraints.

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
    pde_weight : float
        PDE loss weight.
    bc_weight : float
        Boundary condition loss weight.
    ic_weight : float
        Initial condition loss weight.
    reduction : {'mean', 'sum'}
        Determines the loss reduction mode.
    alpha : float
        Wave speed coefficient.
    length : float
        Length of the space interval.
    maxtime : float
        End of the time interval.
    n : int
        Determines the initial situation.
    '''

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 num_hidden=None,
                 activation='tanh',
                 pde_weight=1.0,
                 bc_weight=1.0,
                 ic_weight=1.0,
                 reduction='mean',
                 c=1.0,
                 length=1.0,
                 maxtime=1.0,
                 n=1):

        super().__init__()

        self.c = c  # Store wave speed
        # set up Wave_1D_Equation
        self.equation = Wave_1D_Equation(c=c, length=length, maxtime=maxtime, n=n)
        # create NN model
        self.model = make_fc_model(num_inputs, num_outputs, num_hidden, activation)
        # store loss weights as tensor
        self.pde_weight = pde_weight
        self.bc_weight = bc_weight
        self.ic_weight = ic_weight
        # initialize criterion
        self.reduction = reduction
        self.criterion = nn.MSELoss(reduction=reduction)

    def forward(self, t, x):
        '''Predict PDE solution.'''
        # Reshape t and x to be 2-dimensional if they are not already
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if x.dim() == 1:
            x = x.unsqueeze(1)
        inputs = torch.cat([t, x], dim=1)
        u = self.model(inputs)
        return u

    def make_collocation(self, num_pde, num_bc, num_ic):
        '''Create collocation points.'''

        # sample points uniformly
        t_pde = torch.rand(num_pde, 1) * self.equation.maxtime
        x_pde = torch.rand(num_pde, 1) * self.equation.length
        t_bc = torch.rand(num_bc, 1) * self.equation.maxtime
        x_bc = torch.cat([torch.zeros(num_bc // 2, 1), self.equation.length * torch.ones(num_bc // 2, 1)], dim=0)
        x_ic = torch.rand(num_ic, 1) * self.equation.length
        
        # reshape tensors
        t_pde = t_pde.view(-1, 1)
        x_pde = x_pde.view(-1, 1)
        t_bc = t_bc.view(-1, 1)
        x_bc = x_bc.view(-1, 1)
        x_ic = x_ic.view(-1, 1)

        # create output dict
        out_dict = {'pde': (t_pde, x_pde), 'bc': (t_bc, x_bc), 'ic': (torch.zeros(num_ic, 1), x_ic)}
        return out_dict

    def data_loss(self, t, x, y):
        '''Compute standard regression loss.'''
        # predict solution
        u_pred = self.forward(t, x)
        # compute loss
        loss = self.criterion(u_pred, y)
        return loss

    def pde_loss(self, t, x):
        '''Compute PDE-based loss.'''
        # enable grad
        x.requires_grad_(True)
        t.requires_grad_(True)
        # predict solution
        u = self.forward(t, x)
        # autodiff prediction
        u_t = compute_grad(u, t)
        u_tt = compute_grad(u_t, t)
        u_x = compute_grad(u, x)
        u_xx = compute_grad(u_x, x)
        # disable grad
        x.requires_grad_(False)
        t.requires_grad_(False)
        # compute residual
        residual = u_tt - (self.c ** 2) * u_xx
        # compute loss
        if self.reduction == 'mean':
            loss = torch.mean(residual ** 2)
        else:
            loss = torch.sum(residual ** 2)
        return loss

    def bc_loss(self, t):
        '''Compute boundary condition loss.'''
        # predict solution at left/right boundary
        x_left = torch.zeros_like(t)
        x_right = self.equation.length * torch.ones_like(t)
        u_left = self.forward(t, x_left)
        u_right = self.forward(t, x_right)
        # get boundary condition
        u_bc = self.equation.boundary_condition(t)
        # compute loss
        loss = self.criterion(u_left, u_bc) + self.criterion(u_right, u_bc)
        return loss

    def ic_loss(self, x):
        '''Compute initial condition loss.'''
        # predict solution at initial time
        t_zero = torch.zeros_like(x, requires_grad=True)
        u_pred = self.forward(t_zero, x)
        u_t_pred = compute_grad(u_pred, t_zero)
        # get initial condition
        u0, ut0 = self.equation.initial_condition(x)
        # compute loss
        loss = self.criterion(u_pred, u0) + self.criterion(u_t_pred, ut0)
        return loss

    def physics_loss(self,
                     pde_t,
                     pde_x,
                     bc_t=None,
                     ic_x=None):
        '''Compute total physics loss.'''
        # set bc_t & ic_x, if None
        if bc_t is None:
            bc_t = torch.rand(pde_t.size(0), 1) * self.equation.maxtime
        if ic_x is None:
            ic_x = torch.rand(pde_x.size(0), 1) * self.equation.length
        # compute loss terms
        final_pde_loss = self.pde_weight * self.pde_loss(pde_t, pde_x)
        final_bd_loss = self.bc_weight * self.bc_loss(bc_t)
        final_ic_loss = self.ic_weight * self.ic_loss(ic_x)
        # compute total loss
        total_loss = final_pde_loss + final_bd_loss + final_ic_loss
        return total_loss

def compute_grad(outputs, inputs):
    """Compute gradients."""
    u_grad = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    return u_grad

