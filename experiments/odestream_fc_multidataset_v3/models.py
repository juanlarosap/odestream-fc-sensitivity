"""Historical architecture; no model instances are created during import."""
import torch
from torch import nn
from .neural_ode import NNODEF, NeuralODE

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        #print("input_dim, ", input_dim)
        #print("hidden_dim, ", hidden_dim)
        #print("latent_dim, ", latent_dim)

        self.rnn = nn.GRU(input_dim + 1, hidden_dim)
        self.hid2lat = nn.Linear(hidden_dim, 2 * latent_dim)

    def forward(self, x, t):
        #print(x.shape)
        #print("t in RNNEncoder", t.shape)
        # Concatenate time to input
        t = t.clone()
        #print("t in RNNEncoder", t.shape)
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.
        xt = torch.cat((x, t), dim=-1)
       # print(xt.shape)

        _, h0 = self.rnn(xt.flip((0,)))  # Reversed
        # Compute latent dimension
        z0 = self.hid2lat(h0[0])
        z0_mean = z0[:, :self.latent_dim]
        z0_log_var = z0[:, self.latent_dim:]
        return z0_mean, z0_log_var

class NeuralODEDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(NeuralODEDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        func = NNODEF(latent_dim, hidden_dim, time_invariant=True)
        self.ode = NeuralODE(func)
        self.l2h = nn.Linear(latent_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, z0, t):
        zs = self.ode(z0, t, return_whole_sequence=False)

        hs = self.l2h(zs)
        xs = self.h2o(hs)
        #print ("true")
        #print (zs.shape, hs.shape, xs.shape )

        return xs

class ODEVAE(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim, input_dim ):
        super(ODEVAE, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = RNNEncoder(input_dim, hidden_dim, latent_dim)

        self.decoder = NeuralODEDecoder(output_dim, hidden_dim, latent_dim)

    def forward(self, x, t, y, MAP=False):
        #print("x = ", x.shape)
       #print(t.shape)

        z_mean, z_log_var = self.encoder(x, t)
        if MAP:
            z = z_mean
        else:
            z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var)
        x_p = self.decoder(z, t)
        #print(x_p.shape)
        return x_p, z, z_mean, z_log_var

    def generate_with_seed(self, seed_x, t):
        seed_t_len = seed_x.shape[0]
        # print ("t ", t)
        # print ("seed_t_len", seed_t_len)
        # print ("T seed_t_len", t[:seed_t_len])

        z_mean, z_log_var = self.encoder(seed_x, t[:seed_t_len])
        x_p = self.decoder(z_mean, t)
        return x_p

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seq):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        out, _ = self.lstm(x, (h0, c0))

        # Index the last time step output
        out = out[-1:,: , :]
        #print(out.shape)
        out = out.reshape(out.size(1), out.size(2))
        # Pass the output through the fully connected layer
        out = self.fc(out)

        return out

class ConcatenationLayer(nn.Module):
    def __init__(self):
        super(ConcatenationLayer, self).__init__()

    def forward(self, x1, x2):
        # Concatenate the outputs of both models along the specified dimension (dim=1)
        concatenated_output = torch.cat((x1, x2), dim=1)
        return concatenated_output

class CombinedModel(nn.Module):
    def __init__(self, model1, model2, concatenation_layer, size, output_size):
        super(CombinedModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.concatenation_layer = concatenation_layer
        self.final_layer = nn.Linear(output_size*2 , output_size)

    def forward(self, x,t,yr):
        output1 , z, z_mean, z_log_var= self.model1(x,t,yr)

       # print(z.shape)
       # print(z_mean.shape)
       # print(z_log_var.shape)
      #  exit()
        output2 = self.model2(x)

        concatenated_output = self.concatenation_layer(output1, output2)


        concatenated_output = self.final_layer(concatenated_output)
        return concatenated_output,output1 , output2,  z, z_mean, z_log_var
