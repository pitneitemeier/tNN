import torch
print('testing wether cuda is availible')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
