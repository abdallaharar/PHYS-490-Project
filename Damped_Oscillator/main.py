import json, argparse, torch, sys, pathlib
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as func
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
PATH = Path(__file__).parent
sys.path.append(str(PATH.parent / "SciNet"))
#deals with pathing issues, start from root
from SciNet_Module import SciNet
from generator import oscillator  

#Defines device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Y-list must be global so that multiple training sequences can append to it.
y_list = []

#KL_div function
def kl_div(mu, sigma):
  kld =  0.5 * torch.mean(torch.sum(mu.pow(2) + (2 * sigma).exp() - 2 * sigma - 1, dim = 1)) 
  return kld


#Loss calc function
def loss_BVAE(y, target, mu, sigma): 
  mseloss = func.mse_loss(y, target)
  kld_loss = kl_div(mu, sigma)
  return mseloss, kld_loss


#main training function
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
                loss = (Mseloss + beta * kldloss)
                tot_loss += loss
                
                loss.backward()
                torch.nn.utils.clip_grad_value_(net.parameters(), 10)
                optimizer.step()
                
            if (j + 1) % 50 == 0:
                print(j + 1)
            y_list.append(tot_loss)
        
    else:
      net.train()
      for i in range(epoch):
      
        out, mu, sigma = net(x,q)
        Mseloss, kldloss = loss_BVAE(out, target, mu, sigma)
        optimizer.zero_grad()
        net.zero_grad()
        loss = (Mseloss + beta * kldloss)
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), 10)
        optimizer.step()
        if (i + 1) % 50 == 0:
          print(i + 1)
        y_list.append(loss)

def main():
  #95k comes from paper
    samples = 100
    data = np.zeros((50, samples))
    q = np.zeros(samples)
    target = np.zeros(samples)
    print("Generating Samples")
    for i in range(samples):
      if ((i+1) % 5000) == 0:
        print(i+1,"out of: ", samples)
      #defining spring variables
      Gamma = random.uniform(0.5, 1)
      Omega = random.uniform(5, 10)
      damped_oscillator = oscillator(Gamma, 1, Omega, 0, 10, 0, 100)
      damped_oscillator.iterate_underdamped()
      data[:, i] = np.array(damped_oscillator.y_points[0:50])
      question = random.randint(0, 100)
      q[i] = damped_oscillator.x_points[question]
      target[i] = damped_oscillator.y_points[question]
    
    #Import HyperParams from json
    with open(PATH/'params.json') as json_data:
      d = json.load(json_data)
      json_data.close()
    learning_rate = d['learning rate']
    num_epochs = d['num epochs']
    beta = d['beta']

    #Initializing the network
    model = SciNet(observation_size = 50, encode_h1 = 500, encode_h2 = 250, 
                  decoder_input = (1 + 3), decode_h1 = 250, decode_h2 = 100, 
                  output_size = 1, n_latent_var = 3).to(device)

              
    #call training function
    print("Training Net")
    train(model, 500,learning_rate * 1, beta, data, q, target, 512)
    train(model, 500,learning_rate * 0.1, beta, data, q, target, 512)
    plt.clf
    plt.cla
    x_list = [i for i in range(0,len(y_list))]
    plt.plot(x_list,y_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("loss")
    plt.show()
    
    #set up parameters for data testing
    model.eval()
    torch.no_grad()
    testloss = 0
    output= np.zeros(100)
    print("Creating Testing Plot")
    #range specifies how many testing plot code prints out
    for j in range(3):
      Gamma = random.uniform(0.5,1)
      Omega = random.uniform(5,10)
      damped_oscillator = oscillator(Gamma, 1, Omega, 0, 10, 0, 100)
      damped_oscillator.iterate_underdamped()
      question = random.randint(0,100)
      newdata = np.array(damped_oscillator.y_points[0:50])
      q = np.array(damped_oscillator.x_points[0:100])
      target = damped_oscillator.y_points[50:100]
      newx = Variable(torch.from_numpy(newdata).to(device,dtype=torch.float32))
      values = np.array([target])
      values =  Variable(torch.from_numpy(values)).to(device,dtype=torch.float32)
      q = Variable(torch.from_numpy(q)).to(device,torch.float32)

      newx = newx.reshape(-1,50)
      values =values.reshape(-1,50)
      q = q.reshape(-1,1)
      for i in range(100):
          out, mu, sigma, _ = model(newx,q[i], 1)
          out = out.cpu()
          out = out.detach().numpy()
          output[i] = out[0][0]

      #Prints Testing plot
      q =q.cpu().detach().numpy()
      plt.clf
      plt.cla
      plt.title("Output Confirmation Plot")
      plt.plot(damped_oscillator.x_points,damped_oscillator.y_points, color = 'blue')
      plt.plot(q,output, 'r--')
      plt.show()

    #sets up scatter plot
    Gamma_list = []
    Velocity_list = []
    latent_out_1 = np.zeros((100,100))
    latent_out_2 = np.zeros((100,100))
    latent_out_3 = np.zeros((100,100))
    k_values = np.linspace(0.5,1,100)
    w_values = np.linspace(5,10,100)
    print("starting 3d scatter splot")
    for k in range(100):
      for j in range(100):
          if (j+1%50 == 0):
            print(j,"/",samples)
          Gamma = k_values[k]
          Omega = w_values[j]
          damped_oscillator = oscillator(Gamma, 1, Omega, 0, 10, 0, 100)
          damped_oscillator.iterate_underdamped()
          question = 50
          newdata = np.array(damped_oscillator.y_points[0:50])         
          q = np.array(damped_oscillator.x_points[question])
          target = damped_oscillator.y_points[question]
          newx = Variable(torch.from_numpy(newdata).to(device,dtype = torch.float32))
          values = np.array([target])
          values =  Variable(torch.from_numpy(values)).to(device,dtype = torch.float32)
          q = Variable(torch.from_numpy(q)).to(device,torch.float32)
          
          newx = newx.reshape(-1, 50)
          values =values.reshape(-1, 1)
          q = q.reshape(-1, 1)
          out, mu, sigma, latent_out = model(newx, q, 1)
          
          latent_out_1[k,j] = latent_out[0][0].item()
          latent_out_2[k,j] = latent_out[0][1].item()
          latent_out_3[k,j] = latent_out[0][2].item()  

    # LATENT_OUT_1
    plt.clf
    plt.cla
    # Plot the surface.
    X, Y = np.meshgrid(k_values,w_values)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('${b \ [kg/s]}$')
    ax.set_ylabel('${k \ [kg/s^{2}]}$')
    ax.set_zlabel('latent variable')

    # Customize the z axis.
    ax.plot_surface(X, Y, latent_out_1,cmap='inferno')
    plt.title("Latent 1")
    plt.show()

    # LATENT_OUT_2
    plt.clf
    plt.cla
    # Plot the surface.
    X, Y = np.meshgrid(k_values,w_values)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('${b \ [kg/s]}$')
    ax.set_ylabel('${k \ [kg/s^{2}]}$')
    ax.set_zlabel('latent variable')

    # Customize the z axis.
    ax.plot_surface(X, Y, latent_out_2, cmap='inferno')
    plt.title("Latent 2")
    plt.show()

    # LATENT_OUT_3
    plt.clf
    plt.cla
    # Plot the surface.
    X, Y = np.meshgrid(k_values,w_values)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('${b \ [kg/s]}$')
    ax.set_ylabel('${k \ [kg/s^{2}]}$')
    ax.set_zlabel('latent variable')

    # Customize the z axis.
    ax.plot_surface(X, Y, latent_out_3, cmap='inferno')
    plt.title("Latent 3")
    plt.show()

if __name__ == '__main__':
  main()
