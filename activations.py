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
    return (fr(input.real)-fi(input.imag)).type(dtype) + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexConv1d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_r = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self,input):    
        return apply_complex(self.conv_r, self.conv_i, input)