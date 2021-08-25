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

class Identity(nn.Module):
  def __init__(self):
      super().__init__()
  def forward(self, input):
    return input

class selfAttentionBlock(nn.Module):
  def __init__(self, hidden_size, act_fun=None):
      super().__init__()
      self.value_linear = nn.Linear(hidden_size, hidden_size)
      self.weight_linear = nn.Linear(hidden_size, hidden_size)
      self.ff_linear1 = nn.Linear(hidden_size, hidden_size)
      self.ff_linear2 = nn.Linear(hidden_size, hidden_size)
      self.att_Norm = nn.LayerNorm(hidden_size)
      self.ff_Norm = nn.LayerNorm(hidden_size)
      if act_fun is not None:
        self.act_fun=act_fun()
      else:
        self.act_fun = Identity
  def forward(self, input):
    attended = self.att_Norm( self.value_linear(input) * self.weight_linear(input) ) + input
    out = self.ff_Norm( self.ff_linear2( self.act_fun( self.ff_linear1(attended) ) ) ) + input
    return out

class selfAttentionBlock1(nn.Module):
  def __init__(self, hidden_size, act_fun=None):
      super().__init__()
      self.value_linear = nn.Linear(hidden_size, hidden_size)
      self.weight_linear = nn.Linear(hidden_size, hidden_size)
      self.ff_linear1 = nn.Linear(hidden_size, hidden_size)
      self.ff_linear2 = nn.Linear(hidden_size, hidden_size)
      self.inp_Norm = nn.LayerNorm(hidden_size)
      self.att_Norm = nn.LayerNorm(hidden_size)
      if act_fun is not None:
        self.act_fun=act_fun()
      else:
        self.act_fun = Identity
  def forward(self, input):
    out = self.act_fun( self.inp_Norm(input) )
    out = self.att_Norm( self.value_linear(out) * self.weight_linear(out) + out )
    out = self.ff_linear2( self.act_fun( self.ff_linear1(out) ) ) + out
    return out

class selfAttention(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, num_hidden_layers, act_fun, last_act=None):
    super().__init__()
    self.hidden_layers = nn.ModuleList([selfAttentionBlock(hidden_size=hidden_size, act_fun=act_fun) for _ in range(num_hidden_layers)])
    self.first_layer = nn.Sequential(
      nn.Linear(in_features=input_size, out_features=hidden_size),
      act_fun(),
    )
    if last_act is None:
      self.last_layer = nn.Linear(hidden_size, output_size)
    else:
      self.last_layer = nn.Sequential(
        nn.Linear(hidden_size, output_size),
        last_act()
      )
  def forward(self, input):
    out = self.first_layer(input)
    for layer in self.hidden_layers:
      out = layer(out)
    return self.last_layer(out)

class selfAttention1(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, num_hidden_layers, act_fun, last_act=None):
    super().__init__()
    self.hidden_layers = nn.ModuleList([selfAttentionBlock1(hidden_size=hidden_size, act_fun=act_fun) for _ in range(num_hidden_layers)])
    self.first_layer = nn.Linear(in_features=input_size, out_features=hidden_size)
    if last_act is None:
      self.last_layer = nn.Sequential(
        act_fun(),
        nn.Linear(hidden_size, output_size)
      )
    else:
      self.last_layer = nn.Sequential(
        act_fun(),
        nn.Linear(hidden_size, output_size),
        last_act()
      )
  def forward(self, input):
    out = self.first_layer(input)
    for layer in self.hidden_layers:
      out = layer(out)
    return self.last_layer(out)

class ComplexSelfAttentionBlock(nn.Module):
  def __init__(self, hidden_size, act_fun=None):
    super().__init__()
    self.value_linear = ComplexLinear(hidden_size, hidden_size)
    self.weight_linear = ComplexLinear(hidden_size, hidden_size)
    self.ff_linear1 = ComplexLinear(hidden_size, hidden_size)
    self.ff_linear2 = ComplexLinear(hidden_size, hidden_size)
    if act_fun is not None:
      self.act_fun=act_fun()
    else:
      self.act_fun = Identity
  def forward(self, input):
    attended = self.value_linear(input) * self.weight_linear(input) + input
    out = self.ff_linear2( self.act_fun( self.ff_linear1(attended) ) ) + input
    return out

class ComplexSelfAttentionBlock1(nn.Module):
  def __init__(self, hidden_size, act_fun=None):
    super().__init__()
    self.value_linear = ComplexLinear(hidden_size, hidden_size)
    self.weight_linear = ComplexLinear(hidden_size, hidden_size)
    self.ff_linear1 = ComplexLinear(hidden_size, hidden_size)
    self.ff_linear2 = ComplexLinear(hidden_size, hidden_size)
    if act_fun is not None:
      self.act_fun=act_fun()
    else:
      self.act_fun = Identity
  def forward(self, input):
    out = self.value_linear(self.act_fun(input)) * self.weight_linear(self.act_fun(input)) + input
    out = self.ff_linear2( self.act_fun( self.ff_linear1(out) ) ) + out
    return out

class ComplexSelfAttention(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, num_hidden_layers, act_fun, last_act=None):
    super().__init__()
    self.hidden_layers = nn.ModuleList([ComplexSelfAttentionBlock(hidden_size=hidden_size, act_fun=act_fun) for _ in range(num_hidden_layers)])
    self.first_layer = nn.Sequential(
      ComplexLinear(in_features=input_size, out_features=hidden_size),
      act_fun(),
    )
    if last_act is None:
      self.last_layer = ComplexLinear(hidden_size, output_size)
    else:
      self.last_layer = nn.Sequential(
        ComplexLinear(hidden_size, output_size),
        last_act()
      )
  def forward(self, input):
    out = self.first_layer(input)
    for layer in self.hidden_layers:
      out = layer(out)
    return self.last_layer(out)

class ComplexSelfAttention1(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, num_hidden_layers, act_fun, last_act=None):
    super().__init__()
    self.hidden_layers = nn.ModuleList([ComplexSelfAttentionBlock1(hidden_size=hidden_size, act_fun=act_fun) for _ in range(num_hidden_layers)])
    self.first_layer = ComplexLinear(in_features=input_size, out_features=hidden_size)
    if last_act is None:
      self.last_layer = nn.Sequential(
        act_fun(),
        ComplexLinear(hidden_size, output_size)
      )
    else:
      self.last_layer = nn.Sequential(
        act_fun(),
        ComplexLinear(hidden_size, output_size),
        last_act()
      )
  def forward(self, input):
    out = self.first_layer(input)
    for layer in self.hidden_layers:
      out = layer(out)
    return self.last_layer(out)

class combination_block(nn.Module):
  def __init__(self, mult_size, out_size, act_fun):
    super().__init__()
    self.linear = ComplexLinear( int(mult_size / 2), out_size)
    self.act_fun = act_fun()
    self.rad_phase = rad_phase_act()
  def forward(self, input1, input2):
    return self.act_fun( self.linear( self.rad_phase( input1 * input2 )))

class DumbCombinationBlock(nn.Module):
  def __init__(self, mult_size, out_size, act_fun):
    super().__init__()
    self.linear = ComplexLinear( mult_size, out_size)
    self.act_fun = act_fun()
    self.to_complex = to_complex()
  def forward(self, input1, input2):
    return self.act_fun( self.linear( self.to_complex( input1 * input2 )))

class LinearBlock(nn.Module):
  def __init__(self, hidden_size, act_fun=None):
      super().__init__()
      self.ff_linear1 = nn.Linear(hidden_size, hidden_size)
      self.ff_linear2 = nn.Linear(hidden_size, hidden_size)
      self.ff_norm = nn.LayerNorm(hidden_size)
      if act_fun is not None:
        self.act_fun=act_fun()
      else:
        self.act_fun = Identity
  def forward(self, input):
    out = self.act_fun( self.ff_norm(input) )
    out = self.ff_linear2( self.act_fun( self.ff_linear1( out ))) + input
    return out

class ComplexLinearBlock(nn.Module):
  def __init__(self, hidden_size, act_fun=None):
      super().__init__()
      self.ff_linear1 = ComplexLinear(hidden_size, hidden_size)
      self.ff_linear2 = ComplexLinear(hidden_size, hidden_size)
      if act_fun is not None:
        self.act_fun=act_fun()
      else:
        self.act_fun = Identity
  def forward(self, input):
    out = self.ff_linear2( self.act_fun( self.ff_linear1( self.act_fun( input )) )) + input
    return out

class FeedForward(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, num_hidden_layers, act_fun, last_act=None):
    super().__init__()
    self.hidden_layers = nn.ModuleList([LinearBlock(hidden_size=hidden_size, act_fun=act_fun) for _ in range(num_hidden_layers)])
    self.first_layer = nn.Linear(in_features=input_size, out_features=hidden_size)
    if last_act is None:
      self.last_layer = nn.Linear(hidden_size, output_size)
    else:
      self.last_layer = nn.Sequential(
        nn.Linear(hidden_size, output_size),
        last_act()
      )
  def forward(self, input):
    out = self.first_layer(input)
    for layer in self.hidden_layers:
      out = layer(out)
    return self.last_layer(out)

class ComplexFeedForward(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, num_hidden_layers, act_fun, last_act=None):
    super().__init__()
    self.hidden_layers = nn.ModuleList([ComplexLinearBlock(hidden_size=hidden_size, act_fun=act_fun) for _ in range(num_hidden_layers)])
    self.first_layer = ComplexLinear(in_features=input_size, out_features=hidden_size)
    if last_act is None:
      self.last_layer = ComplexLinear(hidden_size, output_size)
    else:
      self.last_layer = nn.Sequential(
        ComplexLinear(hidden_size, output_size),
        last_act()
      )
  def forward(self, input):
    out = self.first_layer(input)
    for layer in self.hidden_layers:
      out = layer(out)
    return self.last_layer(out)


class lattice_block(nn.Module):
  def __init__(self, kernel_size, out_size, num_features, num_conv_layers, act_fun, lattice_sites, padding=1, padding_mode='circular'):
    super().__init__()
    conv_out_size = lattice_sites + ( 2 * padding + 1 - kernel_size ) * num_conv_layers
    self.conv_first = nn.Sequential(
      nn.Conv1d(1, num_features, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
      act_fun()
    )
    self.conv_layers = nn.ModuleList([nn.Sequential(
      nn.Conv1d(num_features, num_features, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
      act_fun()
    ) for _ in range(num_conv_layers - 1)])
    self.last_linear = nn.Linear(conv_out_size * num_features, out_size)
    self.flatten_conv = nn.Flatten(start_dim=1, end_dim=-1)
  def forward(self, input):
    conv_out = self.conv_first(input)
    for layer in self.conv_layers:
      conv_out = layer(conv_out)
    out = self.last_linear( self.flatten_conv(conv_out) )
    return out

class ResConv1d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, act_fun, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None):
    super().__init__()
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
    self.norm = nn.GroupNorm(1, in_channels)
    self.act_fun = act_fun()
  def forward(self, input):
    out = self.act_fun( self.norm(input) )
    return self.conv( out ) + input

class ResLatticeBlock(nn.Module):
  def __init__(self, kernel_size, out_size, num_features, num_conv_layers, act_fun, lattice_sites, padding=1, padding_mode='circular'):
    super().__init__()
    conv_out_size = lattice_sites + ( 2 * padding + 1 - kernel_size ) * num_conv_layers
    self.conv_first = nn.Sequential(
      nn.Conv1d(1, num_features, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode),
    )
    self.conv_layers = nn.ModuleList([ResConv1d(num_features, num_features, kernel_size, act_fun, padding='same') for _ in range(num_conv_layers - 1)])
    self.last_linear = nn.Sequential(
      act_fun(),
      nn.Linear(conv_out_size * num_features, out_size)
    )
    self.flatten_conv = nn.Flatten(start_dim=1, end_dim=-1)
  def forward(self, input):
    conv_out = self.conv_first(input)
    for layer in self.conv_layers:
      conv_out = layer(conv_out)
    out = self.last_linear( self.flatten_conv(conv_out) )
    return out
