import json, argparse, torch, sys
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as func
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
y_list = []

sys.path.append('.//SciNet')
from SciNet_Module import SciNet
sys.path.append('.//Damped_Oscillator')
from generator import oscillator

def kl_div(mu, sigma):
  kld =  0.5 * torch.mean(torch.sum(mu.pow(2) + (2 * sigma).exp() - 2 * sigma - 1, dim = 1)) 
  return kld


def loss_BVAE(y, target, mu, sigma): 
  mseloss = func.mse_loss(y, target)
  kld_loss = kl_div(mu, sigma)
  #print(mseloss)
  return mseloss, kld_loss


def train(net, epoch, learning_rate, beta, data, question, target, batch = 0):
    
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    x = Variable(torch.from_numpy(data).to(device, torch.float32))
    target =  Variable(torch.from_numpy(target)).to(device, torch.float32)
    q = Variable(torch.from_numpy(question)).to(device, torch.float32)
    x =x.reshape(-1, 50)
    q =q.reshape(-1, 1)
    target = target.reshape(-1, 1)
    
    if batch:
        for j in range(epoch):
            permutations = torch.randperm(x.shape[0])
            tot_loss = 0
            
            for i in range(0, x.shape[0], batch):
                
                indices = permutations[i:i + batch]
                batch_x = x[indices]
                batch_target =  target[indices]
                batch_q = q[indices]
                out, mu, sigma = net(batch_x, batch_q)
                Mseloss, kldloss = loss_BVAE(out, batch_target, mu, sigma)
                optimizer.zero_grad()
                net.zero_grad()
                
                #print(mu.size())
                #print(Mseloss)
                #print(kldloss)
                
                loss = (Mseloss + beta * kldloss)
                tot_loss += loss
                 
                loss.backward()
                torch.nn.utils.clip_grad_value_(net.parameters(), 10)
                optimizer.step()
                
            if (j + 1) % 50 == 0:
                print(j)
                #print(kldloss)
                #print(Mseloss)
                #print(loss)
            y_list.append(tot_loss)
        
    else:
      net.train()
      for i in range(epoch):
      
        out, mu, sigma = net(x,q)
        Mseloss, kldloss = loss_BVAE(out, target, mu, sigma)
        optimizer.zero_grad()
        net.zero_grad()
        
        #print(mu.size())
        #print(Mseloss)
        #print(kldloss)
    
        loss = (Mseloss + beta * kldloss)
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), 10)
        optimizer.step()
        if (i + 1) % 50 == 0:
          print(i + 1)
          #print(kldloss)
          #print(Mseloss)
          #print(loss)
        y_list.append(loss)

def main():
    samples = 950000
    data = np.zeros((50, samples))
    q = np.zeros(samples)
    target = np.zeros(samples)
    for i in range(samples):
      if (i % 50) == 0:
        print(i,"out of: ", samples)
      
      Gamma = random.uniform(0.5, 1)
      Omega = random.uniform(5, 10)
      Alpha = random.uniform(5, 10)
      Velocity = random.uniform(-1, 1)
      #print(Gamma, Omega, Alpha)
      damped_oscillator = oscillator(Gamma, 1, Omega, 0, 10, Velocity, 100)
      damped_oscillator.iterate_underdamped()
      data[:, i] = np.array(damped_oscillator.y_points[0:50])
      
      question = random.randint(0, 100)
      q[i] = damped_oscillator.x_points[question]
      target[i] = damped_oscillator.y_points[question]
    #data = np.array(data)
    
    #Import HyperParams from json
    with open('params.json') as json_data:
      d = json.load(json_data)
      json_data.close()
    learning_rate = d['learning rate']
    num_epochs = d['num epochs']
    beta = d['beta']
    
    #Specifies to run on GPU if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    #Initializing the network
    model = SciNet(observation_size = 50, encode_h1 = 500, encode_h2 = 100, 
                  decoder_input = (1 + 3), decode_h1 = 100, decode_h2 = 100, 
                  output_size = 1, n_latent_var = 3).to(device)
    
    #Loss and optimizer
    #loss = loss_BVAE()
    
    train(model, num_epochs,learning_rate, beta, data, q, target, 5120)
    plt.clf
    plt.cla
    x_list = [i for i in range(0,len(y_list))]
    plt.plot(x_list,y_list)
    plt.title("loss")
    plt.show()
    
    model.eval()
    torch.no_grad()
    testloss = 0
    output= np.zeros(100)

    Gamma_list = []
    Velocity_list = []
    latent_out_1 = []
    latent_out_2 = []
    latent_out_3 = []
    k_values = []
    w_values = []

    for j in range(100):
        Gamma = random.uniform(0.5, 1)
        Omega = random.uniform(5, 10)
        Alpha = random.uniform(5, 10)
        Velocity = random.uniform(-1, 1)
        damped_oscillator = oscillator(Gamma, 1, Omega, 0, 10, Velocity, 100)
        damped_oscillator.iterate_underdamped()
        question = random.randint(0, 100)
        newdata = np.array(damped_oscillator.y_points[0:50])
        
        q = np.array(damped_oscillator.x_points[0:100])
        target = damped_oscillator.y_points[50:100]
        newx = Variable(torch.from_numpy(newdata).to(device,dtype = torch.float32))
        values = np.array([target])
        values =  Variable(torch.from_numpy(values)).to(device,dtype = torch.float32)
        q = Variable(torch.from_numpy(q)).to(device,torch.float32)
        
        newx = newx.reshape(-1, 50)
        values =values.reshape(-1, 50)
        q = q.reshape(-1, 1)

        Gamma_list.append(Gamma)
        Velocity_list.append(Velocity)
    
        out, mu, sigma, latent_out = model(newx, q[j], 1)
        out = out.cpu()
        out = out.detach().numpy()

        # print(latent_out)
        
        output[j] = out[0][0]

        latent_out_1.append(latent_out[0][0].item())
        latent_out_2.append(latent_out[0][1].item())
        latent_out_3.append(latent_out[0][2].item())
        k_values.append(Gamma)
        w_values.append(Omega)
        
        q = q.cpu().detach().numpy()
      
    # LATENT_OUT_1
    plt.clf
    plt.cla
    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Customize the z axis.
    ax.scatter3D(latent_out_1, k_values, w_values, color = "green")
    plt.title("simple 3D scatter plot")
    plt.show()

     # LATENT_OUT_2
    plt.clf
    plt.cla
    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Customize the z axis.
    ax.scatter3D(latent_out_2, k_values, w_values, color = "green")
    plt.title("simple 3D scatter plot")
    plt.show()

   # LATENT_OUT_3
    plt.clf
    plt.cla
    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Customize the z axis.
    ax.scatter3D(latent_out_3, k_values, w_values, color = "green")
    plt.title("simple 3D scatter plot")
    plt.show()


if __name__ == '__main__':
  main()
