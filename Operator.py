import collections
import torch

class Operator:
  def __init__(self, lat_site):
    self.lat_site = lat_site
    self.switching_map = 1
    self.mat_els = torch.tensor([1., 1.])

    self.name = 'not yet named'

  def __str__(self):
    return self.name+f"_{self.lat_site}"

  #multiplication of operator: tuple
  #sum of operators: list
  def __mul__(self, other):
    if isinstance(other, (float, int)):
      self.mat_els *= other
      return self
    elif isinstance(other, Operator):
      if other.lat_site == self.lat_site:
        print('multiplication of operators on same lattice site nyi')
        return self
      else:
        return [(self,other)]
    else:
      print('multiplication for this type nyi')
      return self

  def __rmul__(self, other):
    return self.__mul__(other)

  def __add__(self, other):
    if isinstance(other, Operator):
      return [(self,), (other,)]
    elif isinstance(other, collections.Sequence):
      if (type(other) == list):
        if (len(other) == 0 or isinstance(other[0][0], Operator)):
          return  [(self,)] + other
        else:
          print('cant add to sequence, wrong type')
      elif (type(other) == tuple):
        if (isinstance(other[0], Operator)):
          return [(self,), other]
        else:
          print('cant add to sequence, wrong type')
    else:
      print('add for this type nyi')
      return self

  def __radd__(self, other):
    return self.__add__(other)
  


class Sx(Operator):
  def __init__(self, lat_site):
    super().__init__(lat_site)
    self.switching_map = -1
    self.mat_els = torch.tensor([1., 1.], dtype=torch.complex64)
    self.name = 'S^x'

class Sy(Operator):
  def __init__(self, lat_site):
    super().__init__(lat_site)
    self.switching_map = -1
    self.mat_els = torch.tensor([1.j, -1.j], dtype=torch.complex64)
    self.name = 'S^y'

class Sz(Operator):
  def __init__(self, lat_site):
    super().__init__(lat_site)
    self.switching_map = 1
    self.mat_els = torch.tensor([1., -1.], dtype=torch.complex64)
    self.name = 'S^z'

def print_op_list(op_list):
  out_str = "Hamiltonan = "
  for j, op_tuple in enumerate(op_list):
    if (j>0 and j<len(op_list)):
      out_str += (" + ")
    for i, op in enumerate(op_tuple):
      out_str += str(op)
      if (i<(len(op_tuple)-1)):
        out_str += (" * ")
  print(out_str)