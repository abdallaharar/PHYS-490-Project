
import torch
import torch.nn as nn
import torch.nn.functional as func

'''class Encode(nn.module):
  def __init__(self, **kargs):
    super(Scinet,  self).__init__()
    self.fc1 = nn.Linear(krags['observation_size', kargs['h1'])
    self.fc2 = nn.Linear(kargs['h1'], kargs['h2'])
    self.fc3 = nn.Linear(kargs['h2'], kargs['n_latent_var'])
  def forward(self, x):
    x = func.ELU(self.fc1)
    x = func.ELU(self.fc2(x))
    x = func.ELU(self.fc3(x)
    return x
    
  
class Decoder(nn.module):
  def __init__(self, input_size, **kargs):
    super(Scinet,  self).__init__()
    self_fc1 = nn.Linear(krags['decoder_input', kargs['h1'])
    self_fc2 = nn.Linear(kargs['h1'], kargs['h2'])
    self_fc3 = nn.Linear(kargs['h2'], kargs['output_size'])
   def forward(self, x):
    x = func.ELU(self.fc1)
    x = func.ELU(self.fc2(x))
    x = func.ELU(self.fc3(x)
    return x'''

class SciNet(nn.module):
  def __init__(self, **kargs):
    super(SciNet,  self).__init__()

    self.n_latent = kargs['n_latent_var']
    self.question = kargs['question']

    self.encode = nn.Sequential() 
    self.encode.add(nn.Linear(krags['observation_size', kargs['encode_h1']))
    self.encode.add(func.ELU())
    self.encode.add(nn.Linear(kargs['encode_h1'], kargs['encode_h2']))
    self.encode.add(func.ELU())
    self.encode.add(nn.Linear(kargs['encode_h2'], self.n_latent*2)
    
    self.decode = nn.Sequential()
    self.encode.add(nn.Linear(krags['decoder_input', kargs['decode_h1']))
    self.encode.add(func.ELU())
    self.encode.add(nn.Linear(kargs['decode_h1'], kargs['decode_h2']))
    self.encode.add(func.ELU())
    self.encode.add(nn.Linear(kargs['decode_h2'], kargs['output_size'])
    
  def _encode(self, x):
    return self.encode(x) 

  def _decode(self, x):
    return self.decode(x)

  def forward(self, x):
    dist = self._encode(x)

    mu = [:, :n_latent]
    sigma = [:, n_latent:]
    std_ = sigma.divid(2).exp()
    eps = torch.normal(mean=0, std=std_)
    z = mu = std_*eps

    z = torch.cat((z, self.question), 0)

    x_ = self._decode(z)
    
  return x_, mu, sigma

