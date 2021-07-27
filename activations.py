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