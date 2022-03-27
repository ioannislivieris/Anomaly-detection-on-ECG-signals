# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
# PyTorch library
#
import torch
import torch.nn.functional          as F



def ELBO(output, data, latent_mean, latent_logvar):
    MSE = F.mse_loss(output, data, size_average=False) 
    
    KL  = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        
    return MSE + KL, MSE, KL