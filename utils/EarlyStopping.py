# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# Basic library
#
import numpy as np

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# PyTorch library
#
import torch

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience = 10, verbose = False, path = 'model.pth', delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience   = patience
        self.delta      = delta
        self.verbose    = verbose
        self.path       = path
        self.counter    = 0
        self.best_loss  = np.Inf
        self.early_stop = False
        
    def __call__(self, val_loss, model):
       
        if self.best_loss - val_loss > self.delta:
            self.save_checkpoint(val_loss, model)
        else:
            print('[INFO] EarlyStopping counter: {} out of {}\n'.format(self.counter, self.patience))
            self.counter += 1

        if self.counter >= self.patience:
            # Early stopping
            self.early_stop = True
            

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print('[INFO] Validation loss decreased ({:.6f} --> {:.6f})\n'.format(self.best_loss, val_loss))
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss   