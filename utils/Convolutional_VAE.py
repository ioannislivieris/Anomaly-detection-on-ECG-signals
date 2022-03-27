# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# PyTorch library
#
import torch
import torch.nn                     as nn



# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Encoder
#
class Encoder(nn.Module):
    def __init__(self, nFeatures, z_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # nn.Conv1d(Input channel, Output channel, kernel size)                        
            nn.Conv1d( nFeatures, 32, kernel_size = 4, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size = 8, stride=2, padding=0),
            nn.ReLU(),
            )
        
        self.fc1  = nn.Linear(64*32, 128)

        self.fc1a = nn.Linear(128, z_dim)
        self.fc1b = nn.Linear(128, z_dim)
        #
        self.relu = nn.ReLU()
        
    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(-1, 64*32)
        
        encoded = self.fc1( encoded )
        encoded = self.relu( encoded )
        
        z_loc   = self.fc1a( encoded )
        z_scale = self.fc1b( encoded )
        
        return z_loc, z_scale
    
    
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Decoder
#
class Decoder(nn.Module):
    def __init__(self, nFeatures, z_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 64*32)
        self.relu = nn.ReLU()

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size = 8, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, nFeatures,  kernel_size = 2, stride=2, padding=0),
            ) 
        
    def forward(self, z):
        z = self.fc1(z)
        z = self.relu(z)

        z = self.fc2(z)
        z = self.relu(z)
        
        z = z.view(-1, 64, 32)
    
        z = self.decoder(z)
        
        return z


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Convolutional VAE
#
class VAE(nn.Module):
    def __init__(self, nFeatures, z_dim, device):
        super(VAE, self).__init__()
        self.encoder = Encoder(nFeatures, z_dim)
        self.decoder = Decoder(nFeatures, z_dim)
        self.z_dim   = z_dim
        self.device  = device
        
    def reparameterize(self, z_loc, z_scale):
        std     = z_scale.mul(0.5).exp_()
        epsilon = torch.randn(*z_loc.size()).to(self.device)
        z       = z_loc + std * epsilon
        
        return z
    
    def transform(self, x):
        
        # Encoder pass
        #
        z_loc, z_scale = self.encoder(x)

        # Re-Parameterization
        #
        latent = self.reparameterize(z_loc, z_scale)       
        
        return ( latent )
    
    def forward(self, x):
        # Encoder pass
        #
        z_loc, z_scale = self.encoder(x)
        
        # Re-Parameterization
        #
        z = self.reparameterize(z_loc, z_scale)
        
        # Decoder pass
        #
        decoded = self.decoder(z)
        
        return ( z_loc, z_scale, decoded )
