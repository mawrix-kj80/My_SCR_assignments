'''PINN training.'''

from warnings import warn
import torch
import numpy as np

def test_pinn(pinn, colloc_dict):
    '''
    Test PINN physics loss.

    Summary
    -------
    The physics loss of a PINN is computed for given collocation points.
    It is remarked that, due to the occurrence of the partial derivatives
    in the loss function, the autograd machinery needs to be enabled.

    Parameters
    ----------
    pinn : PINN module
        PINN model with a physics loss method.
    colloc_dict : dict
        Dict of collocation points.

    '''
    # put model in eval mode
    pinn.eval()
    # calculate its loss
    # with torch.no_grad():
    loss = pinn.physics_loss(*colloc_dict['pde'], bc_t=colloc_dict['bc'][0], ic_x=colloc_dict['ic'][1])
    return loss

def train_pinn(pinn,
               optimizer,
               num_epochs,
               train_colloc,
               val_colloc=None,
               print_every=1):
    '''
    Train PINN by minimizing the physics loss.

    Summary
    -------
    A CPU-based non-batched training scheme for PINNs is provided.
    The physics loss is minimized for a given set of collocation points.
    It is assumed that no real observational data is available,
    such that the regression loss can be omitted.

    Parameters
    ----------
    pinn : PINN module
        PINN model with a physics loss method.
    num_epochs : int
        Number of training epochs.
    train_colloc : dict
        Dict of collocation points for training.
    val_colloc : dict
        Dict of collocation points for validation.
    print_every : int
        Determines when losses are printed.

    '''

    # Ensure num_epochs is an integer
    num_epochs = int(num_epochs)
    
    # perform initial test
    train_loss = test_pinn(pinn, train_colloc)
    val_loss = test_pinn(pinn, val_colloc) if val_colloc is not None else None
    
    # print initial losses
    if print_every > 0:
        print(f'Epoch 0: Train Loss = {train_loss:.6f}', end='')
        if val_loss is not None:
            print(f', Val Loss = {val_loss:.6f}')
        else:
            print()

    # history of train, val losses
    # Initialize history arrays
    train_loss_history = np.zeros(num_epochs + 1)
    val_loss_history = np.zeros(num_epochs + 1) if val_colloc is not None else None

    # Store initial losses
    train_loss_history[0] = train_loss
    if val_loss is not None:
        val_loss_history[0] = val_loss

    # loop over training epochs
    for epoch in range(1, num_epochs + 1):
        pinn.train()
        optimizer.zero_grad()
        loss = pinn.physics_loss(*train_colloc['pde'], bc_t=train_colloc['bc'][0], ic_x=train_colloc['ic'][1])
        loss.backward()
        optimizer.step()

        # compute val. loss
        train_loss = test_pinn(pinn, train_colloc)
        val_loss = test_pinn(pinn, val_colloc) if val_colloc is not None else None
        
        # print losses
        if epoch % print_every == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}', end='')
            if val_loss is not None:
                print(f', Val Loss = {val_loss:.6f}')
            else:
                print()
        
        # update history
        # Store losses
        train_loss_history[epoch] = train_loss
        if val_loss is not None:
            val_loss_history[epoch] = val_loss
            
    # Combine train and val loss history into a single dictionary
    history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history if val_loss_history is not None else None
    }

    return history
