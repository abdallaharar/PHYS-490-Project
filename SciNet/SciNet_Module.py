
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable


class SciNet(nn.Module):
  def __init__(self, **kargs):
    super(SciNet,  self).__init__()

    self.n_latent = kargs['n_latent_var']
    

    self.encode = nn.Sequential(nn.Linear(kargs['observation_size'],
                                          kargs['encode_h1'],bias=False), nn.ELU(),
                                nn.Linear(kargs['encode_h1'],
                                          kargs['encode_h2'],bias=False), nn.ELU(),
                                nn.Linear(kargs['encode_h2'], self.n_latent*2
                                          ,bias=False))
    
    self.decode = nn.Sequential(nn.Linear(kargs['decoder_input'],
                                          kargs['decode_h1'],bias=False),nn.ELU(),
                                nn.Linear(kargs['decode_h1'],
                                          kargs['decode_h2'],bias=False),nn.ELU(),
                                nn.Linear(kargs['decode_h2'],
                                          kargs['output_size']))
    

    
  def _encode(self, x):
    return self.encode(x) 

  def _decode(self, x):
    return self.decode(x)

    
  def reparam(self, mu, logsigma):
    
    std = torch.exp(0.5*logsigma)
    eps = torch.randn_like(std) 
    sample = mu + (eps * std) 
    
    return sample


  def forward(self, x, q, output=0):
    dist = self._encode(x)

    
    mu = dist[:, :self.n_latent]
    sigma = dist[:, self.n_latent:]
    
    z = self.reparam(mu, sigma)
    
    
     
    if(output):
        latent_out = z
        
        
    q = q.reshape(-1,1)
        
    z = torch.cat((z, q), -1)


    out = self._decode(z)
  
    

    
    if(output):
        return out, mu, sigma, latent_out
    else:
        return out, mu, sigma

