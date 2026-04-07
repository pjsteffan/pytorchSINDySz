import torch 
import torch.nn as nn
import torch.optim as optim
import lightning as L

from itertools import combinations_with_replacement
import numpy as np





def pytorch_hilbert(signal, axis=0):
    N = signal.size(axis)
    # Compute the FFT of the original signal
    signal_fft = torch.fft.fft(signal, dim=axis)
    
    # Create the Hilbert transform filter
    H = signal.new_zeros(N)
    H[0] = 0  # DC component
    if N % 2 == 0:
        H[1:N//2] = 2  # Positive frequencies
        H[N//2] = 0  # Nyquist frequency if N is even
    else:
        H[1:(N+1)//2] = 2  # Positive frequencies
    
    # Apply the filter to the FFT of the signal
    analytic_signal_fft = H.unsqueeze(1) * signal_fft
    
    # Compute the inverse FFT to get the analytic signal
    analytic_signal = torch.fft.ifft(analytic_signal_fft, dim=axis)
    

    return analytic_signal



def extract_real_component(x):
    return torch.abs(pytorch_hilbert(x, axis=0))
def extract_imaginary_component(x):
    return(torch.angle(pytorch_hilbert(x, axis=0)))

class SINDyModel(nn.Module):
    def __init__(self, time_dim, system_features, latent_features,poly_order):
        super(SINDyModel, self).__init__()
        self.time_dim = time_dim
        self.system_features = system_features
        self.latent_features = latent_features
        self.poly_order = poly_order
        self.library_dim = self.compute_library_dim()
        self.encoder = nn.Linear(system_features, latent_features)  # Example encoder
        self.decoder = nn.Linear(latent_features, system_features)
        self.SINDy_predict = nn.Linear(self.library_dim, latent_features)  # SINDy prediction layer

    def compute_library_dim(self):
        self_features = self.latent_features   
        hilbert_features = 2 * self.latent_features

        poly_features = 0
        for n in range(1, self.poly_order+1):
            list_combinations = list(combinations_with_replacement(range(self.latent_features), n))
            poly_features += len(list_combinations)
        
        return self_features + hilbert_features + poly_features


    def compute_library(self, z):
        library = []

        sample_list = range(z.shape[1])

        for n in range(1, self.poly_order+1):
            list_combinations = list(combinations_with_replacement(sample_list, n))
            for combination in list_combinations:
                library.append(torch.prod(z[:, combination], dim=1).unsqueeze(1))


        library.append(z)
        library.append(extract_real_component(z))
        library.append(extract_imaginary_component(z))

        return torch.cat(library, dim=1) 

    def forward(self, x):
        
        #Run the forward pass through the model
        z = self.encoder(x)
        theta_x = self.compute_library(z)
        y_hat = self.SINDy_predict(theta_x)
        x_hat = self.decoder(z)


        #Compute the gradients that will be needed for the SINDy loss
        gradients_z_wrt_x = torch.autograd.grad(outputs=z, inputs=x,
                                grad_outputs=torch.ones_like(z),
                                retain_graph=True, create_graph=True)[0]
        
        gradients_x_hat_wrt_z = torch.autograd.grad(outputs=x_hat, inputs=z,
                                grad_outputs=torch.ones_like(x_hat),
                                retain_graph=True, create_graph=True)[0]

        return y_hat, x_hat, gradients_z_wrt_x, gradients_x_hat_wrt_z, self.SINDy_predict.weight
    

class SINDyLoss(nn.Module):
    def __init__(self):
        super(SINDyLoss, self).__init__()
        self.lambda1 = 1.0  #SINDy loss in x_dot
        self.lambda2 = 1.0  #SINDy loss in z_dot
        self.lambda3 = 1.0  #SINDy regularization loss 
    def forward(self, x, y_hat, x_hat, gradients_z_wrt_x, gradients_x_hat_wrt_z, SINDy_weights):
        x_dot = torch.diff(x, dim=0)  # Numerical derivative of x with respect to time
        loss = nn.MSELoss()(x_hat, x) #Reconstruction loss
        loss += self.lambda1 * nn.MSELoss()(x_dot,gradients_x_hat_wrt_z*y_hat) #SINDy loss in x_dot space   
        loss += self.lambda2 * nn.MSELoss()(gradients_z_wrt_x*x_dot, y_hat) #SINDy loss in z_dot space
        loss += self.lambda3 * SINDy_weights.abs().sum() # L1 regularization on SINDy coefficients
        return loss 




class SINDySz(L.LightningModule):
    def __init__(self, model):
        super(SINDySz, self).__init__()
        self.model = model
        self.criterion = SINDyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat, x_hat, gradients_z_wrt_x, gradients_x_hat_wrt_z, SINDy_weights= self.forward(x)
        loss = self.criterion(x, y_hat, x_hat, gradients_z_wrt_x, gradients_x_hat_wrt_z, SINDy_weights)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer


if __name__ == "__main__":
    time_dim = 100
    system_features = 4
    latent_features = 10
    poly_order = 2
    
    model = SINDyModel(time_dim, system_features, latent_features,poly_order)
    sindy_sz = SINDySz(model)
    loss = sindy_sz.training_step(torch.randn(time_dim, system_features,requires_grad=True),0)
    print('done')
    #trainer = L.Trainer(max_epochs=10)
    #trainer.fit(model, train_loader)