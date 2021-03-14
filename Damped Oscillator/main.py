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


def main():
  data = []
  for i in range(10000):
    if (i % 100) == 0:
        print(i)
    batch = []
    Gamma = random.uniform(0.5,1)
    Omega = random.uniform(5,10)
    Alpha = random.uniform(5,10)
    print(Gamma, Omega, Alpha)
    damped_oscillator = oscillator(Gamma, 1, Omega, Alpha, 10, 100)
    damped_oscillator.iterate()
    batch = damped_oscillator.y_points[0:50]
    question = random.randint(0,100)
    data.append([batch,[damped_oscillator.x_points[question],damped_oscillator.y_points[question]]])
  data = np.array(data)

  #Import HyperParams from json
  with open('params.json') as json_data:
    d = json.loads(json_data)
    json_data.close()
    learning_rate = d[model][learning rate]
    num_epochs = d[model][num epochs]

    
  #Specifies to run on GPU if possible
   device = torch.device('cuda' if torch.cuda.is.available() else 'cpu')
  

  #Initializing the network
  model = Scinet(input_size = 50).to(device)

  #Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr = learning_rate)

  #learning
  for epoch in range(num_epochs):


if __name__ == '__main__':
  main()
