import json, argparse, torch, sys
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('../SciNet')
from SciNet import SciNet
import generator
import torch
import torch.nn as nn
import numpy as np
import Scinet
import json
import random



def kl_div(mu, sigma, beta):
  kld = sigma.pow(2).log - mu.pow(2) - sigma.pow(2) 
  return beta/2 * kld.sum()


def loss_BVAE(y, target, mu, sigma, beta):
  mseloss = nn.MSELoss(y, target, reduction='sum')
  kld_loss = kl_div(mu, sigma, beta)
  return mseloss + kld_loss


def train(net, epoch, learning_rate, batch=0, input_vars): 
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)

if batch:
#do later
else:
  x = Variable(torch.from_numpy(input_vars[0]).to(torch.float32)
  target = Variable(torch.from_numpy(input_vars[2]).to(torch.float32)
  q = Variable(torch.from_numpy(input_vars[1]).to(torch.float32)
  
  for i in range(epoch):
    out, mu, sigma = net(x)
    loss = loss_BVAE(out, target, mu, sigma)
    optimizer.zero_grad()
    loss.backwards()
    optimizer.step()
    if (e + 1) % 20 == 0:
      print(loss)



def main():

  #Import HyperParams from json
  with open('params.json') as json_data:
    d = json.loads(json_data)
    json_data.close()
    learning_rate = d['learning rate']
    num_epochs = d['num epochs']
    beta = = d['beta']

  #Specifies to run on GPU if possible
   device = torch.device('cuda' if torch.cuda.is.available() else 'cpu')
  
  #Initializing the network
  model = Scinet(observation_size = 50,encode_h1 = 500,encode_h2 = 100,decoder_input = 100, decode_h1 = 100, decode_h2 = 50,output_size = 1).to(device)

  #Loss and optimizer
  #loss = loss_BVAE()
 

if __name__ == '__main__':
  main()
