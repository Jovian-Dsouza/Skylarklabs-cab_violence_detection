import torch
import torch.nn as nn
import math
from torch.autograd import Function

class BinaryFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()
        #torch.tensor(0.0, device=x.device) if x < 0.5 else torch.tensor(1.0. device=x.device)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class LSTMcell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.U = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.V = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))

        self.init_weights()
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x_t, h_t, c_t, u):
        HS = self.hidden_size
        A = x_t @ self.U + h_t @ self.V + self.bias

        i_t = torch.sigmoid(A[:, :HS])
        f_t = torch.sigmoid(A[:, HS:HS*2])
        g_t = torch.tanh(A[:, HS*2:HS*3])
        o_t = torch.sigmoid(A[:, HS*3:])

        c_t = f_t * c_t + i_t * g_t
        h_t = u * (o_t * torch.tanh(c_t)) + (1-u) * h_t
        
        return (h_t, c_t)

class SkipLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Initialize the LSTMCells
        layers = []
        layers.append(LSTMcell(input_size, hidden_size))
        for n in range(self.n_layers-1):
            layers.append(LSTMcell(hidden_size, hidden_size))
        self.lstm_layers = nn.ModuleList(layers)

        # Initialize the parameters for the Skip 
        self.skip_w = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.skip_b = nn.Parameter(torch.Tensor(1))
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.skip_w, gain=1.0)
        nn.init.ones_(self.skip_b)

    def forward(self, x, init_states=None):
        """shape of x is (batch, sequence, features)
           init_states is (h_t, c_t) 
        """
        batch_size, seq_len , _ = x.size()

        if init_states is None:

            h_t = [torch.zeros(batch_size, self.hidden_size).to(x.device) for n in range(self.n_layers)]
            c_t = [torch.zeros(batch_size, self.hidden_size).to(x.device) for n in range(self.n_layers)]
            u_t = torch.tensor(1.0, device=x.device)
        else:
            h_t, c_t, u_t = init_states


        output = []
        for t in range(seq_len):
            x_t = x[:, t, :]

            # If we are training we cant skip steps as they are need for gradient computation
            # TODO: Enable skip computation for batch inputs where batch_size > 1
            if self.training or (not self.training and batch_size!=1):
                u = BinaryFunction.apply(u_t)

                h_t[0], c_t[0] = self.lstm_layers[0](x_t, h_t[0], c_t[0], u)
                for i in range(1, self.n_layers):
                    h_t[i], c_t[i] = self.lstm_layers[i](h_t[i-1], h_t[i], c_t[i], u)
                
                # TODO check if leaning cum_u from h_t is better than c_t
                cum_u = torch.sigmoid(c_t[-1] @ self.skip_w + self.skip_b)
                u_t = u * cum_u + (1-u) * (u_t + torch.min(cum_u, 1-u_t))
            
            else:
                # Faster evaluation when batch size is 1
                u = BinaryFunction.apply(u_t)
                if(u == 1.0):
                    h_t[0], c_t[0] = self.lstm_layers[0](x_t, h_t[0], c_t[0], u)
                    for i in range(1, self.n_layers):
                        h_t[i], c_t[i] = self.lstm_layers[i](h_t[i-1], h_t[i], c_t[i], u)
                    
                    # TODO check if leaning cum_u from h_t is better than c_t
                    u_t = torch.sigmoid(c_t[-1] @ self.skip_w + self.skip_b)

                else:
                    # Skip steps
                    t += torch.ceil(0.5/u_t) - 1

            output.append(h_t[-1].unsqueeze(0))
        
        output = torch.cat(output, dim=0)
        output = output.transpose(0, 1).contiguous() 

        return output , (h_t, c_t, u_t)


## TODO DEPRECATED FUCNTION 
class SingleLayerSkipLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.U = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.V = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))

        self.skip_w = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.skip_b = nn.Parameter(torch.Tensor(1))

        self.init_weights()
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.skip_w, gain=1.0)
        nn.init.ones_(self.skip_b)

    def forward(self, x, initial_states=None):
        """shape of x is (batch, sequence, features)
           initial_states is (h_t, c_t, u_t) 
        """
        batch_size, seq_len , _ = x.size()

        if initial_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            u_t = torch.tensor(1.0, device=x.device)
        else:
            h_t, c_t , u_t = initial_states

        HS = self.hidden_size

        output = []
        for t in range(seq_len):
            x_t = x[:, t, :]

            if self.training or (not self.training and batch_size!=1):
                A = x_t @ self.U + h_t @ self.V + self.bias

                i_t = torch.sigmoid(A[:, :HS])
                f_t = torch.sigmoid(A[:, HS:HS*2])
                g_t = torch.tanh(A[:, HS*2:HS*3])
                o_t = torch.sigmoid(A[:, HS*3:])

                c_t = f_t * c_t + i_t * g_t

                u = BinaryFunction.apply(u_t)
                h_t = u * (o_t * torch.tanh(c_t)) + (1-u) * h_t
                cum_u = torch.sigmoid(h_t @ self.skip_w + self.skip_b)
                u_t = u * cum_u + (1-u) * (u_t + torch.min(cum_u, 1-u_t))
            
            else:
                # Faster evaluation when batch size is 1
                if(BinaryFunction.apply(u_t) == 1.0):
                    A = x_t @ self.U + h_t @ self.V + self.bias
                    i_t = torch.sigmoid(A[:, :HS])
                    f_t = torch.sigmoid(A[:, HS:HS*2])
                    g_t = torch.tanh(A[:, HS*2:HS*3])
                    o_t = torch.sigmoid(A[:, HS*3:])

                    c_t = f_t * c_t + i_t * g_t

                    h_t = o_t * torch.tanh(c_t)

                    # TODO Ct or ht
                    u_t = torch.sigmoid(c_t @ self.skip_w + self.skip_b)
                    # u_t = torch.sigmoid(h_t @ self.skip_w + self.skip_b)

                else:
                    # Skip steps
                    t += torch.ceil(0.5/u_t) - 1
                    

            output.append(h_t)

        output = torch.stack(output, dim=1)


        return output , (h_t, c_t, u_t)



if __name__ == "__main__":
    skiplstm = SkipLSTM(10, 20)
    x = torch.rand((2, 30, 10))
    print("Input shape",x.shape)
    print("Output shape", skiplstm(x)[0].shape)