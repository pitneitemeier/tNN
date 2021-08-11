from torch import nn
import torch


class even_act(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x**2 / 2 - x**4 / 12 + x**6 / 46

class odd_act(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x - x**3 / 3 + x**5 * (2 / 15)

class complex_act(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return (torch.tanh(torch.real(x)) + 1j * torch.tanh(torch.imag(x)))

class complex_relu(nn.Module):
  def __init__(self):
    super().__init__()
    self.relu = nn.ReLU()
  def forward(self, x): 
    return self.relu(torch.real(x)) + 1j * self.relu(torch.imag(x))

class complex_tanh(nn.Module):
  def __init__(self):
    super().__init__()
    self.tanh = nn.Tanh()
  def forward(self, x): 
    return self.tanh(torch.real(x)) + 1j * self.tanh(torch.imag(x))

class complex_celu(nn.Module):
  def __init__(self):
    super().__init__()
    self.celu = nn.CELU()
  def forward(self, x): 
    return self.celu(torch.real(x)) + 1j * self.celu(torch.imag(x))


class rad_phase_act(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    x = x.reshape(x.shape[0], int(x.shape[1] / 2), 2)
    return x[:, :, 0] * torch.exp(1j*x[:, :, 1])

class rad_phase_conv_act(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    #assumes x of shape (N, C, L) with C%2=0
    half = int(x.shape[1]/2)
    ret = x[:, :half, :] * torch.exp(1j*x[:, half:, :])
    #print('rad_phase shape', ret.shape)
    return ret


class Euler_act(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return torch.exp(1j*x)



class to_complex(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x + 0.j

class sinh_act(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return torch.sinh(x)

class cosh_act(nn.Module):
  def __init__(self, bias=True):
    super().__init__()
    self.b = nn.Parameter(torch.tensor(0.))
    if bias:
      self.b.requiresGrad = True
  def forward(self, x):
    return torch.cosh(x) + self.b

class exp_act(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return torch.exp(x)

class mod_relu(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.))
        self.b.requiresGrad = True
        self.relu = nn.ReLU()

    def forward(self, z):
        return self.relu(torch.abs(z) + self.b) * torch.exp(1.j * torch.angle(z))


def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(torch.real(input))-fi(torch.imag(input))) + 1j*(fr(torch.imag(input))+fi(torch.real(input)))

class ComplexConv1d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self,input):    
        return apply_complex(self.conv_r, self.conv_i, input)

class ComplexLinear(nn.Module):
    def __init__(self,in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.lin_r = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.lin_i = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        
    def forward(self,input):    
        return apply_complex(self.lin_r, self.lin_i, input)


class Attention_Block(nn.Module):
  def __init__(self, embed_dim, num_heads):
    super().__init__()
    self.att_layer = nn.MultiheadAttention( embed_dim, num_heads, batch_first=True)
    self.key_conv = nn.Conv1d(32, 32, 1)
    self.query_conv = nn.Conv1d(32, 32, 1)
    self.value_conv = nn.Conv1d(32, 32, 1)
    self.linear_conv1 = nn.Conv1d(32, 32, 1)
    self.linear_conv2 = nn.Conv1d(32, 32, 1)
    self.relu = nn.ReLU()
  def forward(self, input):
    att_out, _ = self.att_layer(self.query_conv(input), self.key_conv(input), self.value_conv(input))
    out = self.linear_conv2( self.relu( self.linear_conv1(att_out + input) ) )
    return out + att_out + input

import copy
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class parallel_Linear(nn.Module):
  def __init__(self, in_features, out_features, num_heads, bias=True, device=None, dtype=None):
      super().__init__()
      self.layers = nn.ModuleList([nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype) for _ in range(num_heads)])
  def forward(self, input):
    return torch.stack([layer(input) for layer in self.layers], dim=2)

import torch.nn.functional as F
class time_attention(nn.Module):
  def __init__(self, mult_size, num_heads):
    super().__init__()
    self.t_embed = parallel_Linear(in_features=mult_size, out_features=int(mult_size/num_heads), num_heads=num_heads)
    self.lat_embed = parallel_Linear(mult_size, int(mult_size/num_heads), num_heads)
    self.value_embed = parallel_Linear(mult_size * 2, int(mult_size/num_heads), num_heads)
    self.linear_out = nn.Linear(mult_size, mult_size)
    
  def forward(self, t_inp, lat_inp):
    t_embed = self.t_embed(t_inp)
    lat_emb = self.lat_embed(lat_inp)
    value_emb = self.value_embed( torch.cat((t_inp, lat_inp), dim=1))
    att_weights = F.softmax( t_embed * lat_emb, dim=1)
    out = self.linear_out( (att_weights * value_emb).reshape(t_inp.shape) )
    return out

    out = self.linear_out( att_weights)


    
