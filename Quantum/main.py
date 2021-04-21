import json, argparse, torch, sys, pathlib, random
import torch.optim as optim
import torch.nn.functional as func
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from pathlib import Path

PATH = Path(__file__).parent

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device( 'cpu')

sys.path.append('..//SciNet')
from SciNet_Module import SciNet

from generator import create_data

def kl_div(mu, sigma):
    
 
  kld =  0.5 * torch.mean(torch.sum(mu.pow(2)+ (2*sigma).exp() -2*sigma -1, dim=1)) 
  
  return kld


def loss_BVAE(y, target, mu, sigma): 
  mseloss = func.mse_loss((y), target)
  
  kld_loss = kl_div(mu, sigma)
  #print(mseloss)
  return mseloss, kld_loss


def train(net, epoch, learning_rate, beta, data, question, target, err,batch=0):
    
      
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    x = Variable(torch.from_numpy(data).to(device,torch.float32))
    target =  Variable(torch.from_numpy(target)).to(device,torch.float32)
    q = Variable(torch.from_numpy(question)).to(device,torch.float32)
    x =x.reshape(-1,10)
    q =q.reshape(-1,10)
    target = target.reshape(-1,1)
    

    if batch:
        for j in range(epoch):
            
            permutations = torch.randperm(x.shape[0])
            tot_loss = 0
            
            for i in range(0,x.shape[0], batch):
                
                indices = permutations[i:i+batch]
                
                batch_x = x[indices]
                batch_target =  target[indices]
                batch_q = q[indices]
                
                out, mu, sigma = net(batch_x,batch_q)
                # if j % 1000 == 0 and i % 1000 == 0:
                #   print("mu:",mu)
                #   print("sigma:",sigma)
                Mseloss, kldloss = loss_BVAE(out, batch_target, mu, sigma)
                
                optimizer.zero_grad()
                net.zero_grad()
                
                #print(mu.size())
                
                
                #print(Mseloss)
                #print(kldloss)
                
                loss = (Mseloss+ beta* kldloss)
                tot_loss += loss
                 
                loss.backward()
                #torch.nn.utils.clip_grad_value_(net.parameters(), 10)
                optimizer.step()
                
            if (j + 1) % 50 == 0:
                # print(j)
                #print(kldloss)
                #print(Mseloss)
                # print(tot_loss)
                err.append(tot_loss.item())
        
        
        
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
    
        loss = (Mseloss+ beta* kldloss)
    
     
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), 10)
             
      
        optimizer.step()
        if (i + 1) % 50 == 0:
          print(i+1)
          #print(kldloss)
          #print(Mseloss)
          print(loss)
        err.append(loss.cpu().numpy())
        
        
def test(model, test_set):

 
    
    
    observations = test_set[0]
    question =test_set[1]
    target = test_set[2]
    
    x = Variable(torch.from_numpy(observations).to(device,torch.float32))
    target =  Variable(torch.from_numpy(target)).to(device,torch.float32)
    q = Variable(torch.from_numpy(question)).to(device,torch.float32)
    x =x.reshape(-1,10)
    q =q.reshape(-1,10)
    target = target.reshape(-1,1)
    
    out, mu, sigma = model(x,q)
    
    Mseloss, kldloss = loss_BVAE(out, target, mu, sigma)
    
    return Mseloss
    
    
    


def main():
    
      
<<<<<<< HEAD
    samples = 950
=======
    samples = 100000
>>>>>>> 69574b795c6a67a18fd34a36d16dd163c3dee5a8
    n_observation= 10
    n_questions = 10
    test_samples= 5000
    a, b, c = create_data(1, n_observation, n_questions, samples)
    
    #train_set =a[:,:-test_samples]
    
    observations = a[0][test_samples:]
    question =a[1][test_samples:]
    target = a[2][test_samples:]
    
    
    a, b, c = create_data(1, n_observation, n_questions, test_samples)
    
    test_set = (a[0][:test_samples],a[1][:test_samples], a[2][:test_samples])
                    
        
    #Import HyperParams from json
    with open('params.json') as json_data:
      d = json.load(json_data)
      json_data.close()
    learning_rate = d['learning rate1']
    learning_rate2 = d['learning rate2']
    num_epochs1 = d['num epochs1']
    mum_epochs2 = d['num epochs2']
    beta = d['beta']
    latent_max = d['latent max']+1
    
    error = np.zeros(latent_max)
    for k in range(latent_max):
    
        n_latent = k
        #Specifies to run on GPU if possible
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        #Initializing the network
        model = SciNet(observation_size=n_observation, encode_h1=100, encode_h2=100, 
                    decoder_input=(n_questions+n_latent), decode_h1=100, decode_h2=100, 
                    output_size=1,n_latent_var=n_latent).to(device)
        
        #Loss and optimizer
        #loss = loss_BVAE()
        err = []
        train(model, num_epochs1,learning_rate, beta, observations, question,
              target, err, 512)
        train(model, mum_epochs2,learning_rate2, beta, observations, question,
              target,err, 512)
        plt.clf
        plt.cla
        x_list = [i for i in range(0,len(err))]
        print(x_list)
        print(err)
        plt.plot(x_list,err)
        plt_name = "%i_latent.png" % k
        plt_path = PATH / "plots" / plt_name
        Path.mkdir(plt_path.parent, exist_ok=True)
        plt.savefig(plt_path)
        plt.close()

        # Save model
        model_name = "%i_latent.pt" % k
        model_path = PATH / "models" / model_name
        Path.mkdir(model_path.parent, exist_ok=True)
        torch.save(model.state_dict(), model_path)

        
     
        error[k] = test(model, test_set)

    
    plt.bar(np.arange(0,latent_max),error, color ='maroon', width = 0.4)
    plt_path = PATH / "plots" / "final_bar.png"
    plt.savefig(plt_path)
    plt.show()
    
    #validating data
    
    
    

    # testloss += np.sum(np.pow(target - out[0,:],2))
    # testloss = testloss/25
    # print(testloss)
  
   



if __name__ == '__main__':
  main()
