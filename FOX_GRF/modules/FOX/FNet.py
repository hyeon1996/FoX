import torch
import torch.optim as optim

from torch import nn as nn
from torch.nn import functional as F

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        h_ = F.relu(self.FC_input(x))
        h_ = F.relu(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractabxle normal distribution "q"

        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        return z, mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.FC_hidden_tau = nn.Linear(64, hidden_dim)
        self.FC_hidden2_tau = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output_tau = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)

    def forward(self, x):
        h = F.relu(self.FC_hidden(x))
        h = F.relu(self.FC_hidden2(h))
        x_hat = torch.sigmoid(self.FC_output(h))

        return x_hat

class DecoderTau(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(DecoderTau, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = F.relu(self.FC_hidden(x))
        h = F.relu(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat

class VAEModel(nn.Module):
    def __init__(self, Encoder, Decoder, DecoderTau):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.DecoderTau = DecoderTau
        self.optimizer1 = optim.Adam(self.Decoder.parameters(), lr = 1e-3)
        self.optimizer2 = optim.Adam(self.DecoderTau.parameters(), lr = 1e-3)
        self.optimizer3 = optim.Adam(self.Encoder.parameters(), lr = 1e-3)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, z_i,z_max,z_min, update = False):
        if update:
            z = torch.cat((z_i,z_max,z_min),0)
            x_hat = self.Decoder(z)
            x_hat_tau = self.DecoderTau(z_i)
            return x_hat,x_hat_tau
        else:
            z = torch.cat((z_i, z_max, z_min), 3)

            x_hat = self.Decoder(z)

        return x_hat

    def get_log_pi(self, own_variable, other_variable):
        log_prob = -1 * F.mse_loss(own_variable,
                                   other_variable, reduction='none')
        log_prob = torch.sum(log_prob, -1, keepdim=True)
        return log_prob

    def update(self,tau,z_i,z_max,z_min,formation,log_var_i, log_var_max, log_var_min, mean_i, mean_max, mean_min):
        formation = formation / 1000

        f_prime,tau_prime = self.forward(z_i, z_max, z_min, update=True)

        reproduction_loss = F.mse_loss(f_prime, formation)
        KLD_i = - 0.5 * torch.sum(1 + log_var_i - mean_i.pow(2) - torch.exp(log_var_i))
        KLD_max = - 0.5 * torch.sum(1 + log_var_max - mean_max.pow(2) - torch.exp(log_var_max))
        KLD_min = - 0.5 * torch.sum(1 + log_var_min - mean_min.pow(2) - torch.exp(log_var_min))

        decoder1_loss = reproduction_loss + KLD_i + KLD_max + KLD_min

        reconstruction_loss = F.mse_loss(tau_prime, tau)
        decoder2_loss = reconstruction_loss + KLD_i
        encoder_loss = decoder1_loss - 0.1 * decoder2_loss

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()

        decoder1_loss.backward(retain_graph=True)
        decoder2_loss.backward(retain_graph = True)
        encoder_loss.backward()

        with torch.no_grad():
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

        self.optimizer1.step()
        self.optimizer2.step()
        self.optimizer3.step()

        f_loss = decoder1_loss.to('cpu').detach().item()

        return f_loss



