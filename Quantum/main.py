import json, argparse, torch, sys, pathlib, random, pickle
import torch.optim as optim
import torch.nn.functional as func
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from pathlib import Path
PATH = Path(__file__).parent
sys.path.append(str(PATH.parent / "SciNet"))
# sys.path.append('..//SciNet')
from SciNet_Module import SciNet
from generator import create_data

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device( 'cpu')


def kl_div(mu, sigma):
  kld = 0.5 * torch.mean(torch.sum(mu.pow(2) +
                         (2*sigma).exp() - 2*sigma - 1, dim=1))
  return kld


def loss_BVAE(y, target, mu, sigma):
  mseloss = func.mse_loss((y), target)

  kld_loss = kl_div(mu, sigma)
  #print(mseloss)
  return mseloss, kld_loss


def train(net, epoch, learning_rate, beta, data, question, target, err, batch=0):

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    x = Variable(torch.from_numpy(data).to(device, torch.float32))
    target = Variable(torch.from_numpy(target)).to(device, torch.float32)
    q = Variable(torch.from_numpy(question)).to(device, torch.float32)
    x = x.reshape(-1, 10)
    q = q.reshape(-1, 10)
    target = target.reshape(-1, 1)

    if batch:
        for j in range(epoch):

            permutations = torch.randperm(x.shape[0])
            tot_loss = 0

            for i in range(0, x.shape[0], batch):

                indices = permutations[i:i+batch]

                batch_x = x[indices]
                batch_target = target[indices]
                batch_q = q[indices]

                out, mu, sigma = net(batch_x, batch_q)
                # if j % 1000 == 0 and i % 1000 == 0:
                #   print("mu:",mu)
                #   print("sigma:",sigma)
                Mseloss, kldloss = loss_BVAE(out, batch_target, mu, sigma)

                optimizer.zero_grad()
                net.zero_grad()

                #print(mu.size())

                #print(Mseloss)
                #print(kldloss)

                loss = (Mseloss + beta * kldloss)
                tot_loss += loss

                loss.backward()
                #torch.nn.utils.clip_grad_value_(net.parameters(), 10)
                optimizer.step()

            if (j + 1) % 1 == 0:
                # print(j)
                #print(kldloss)
                #print(Mseloss)
                # print(tot_loss)
                err.append(tot_loss.item())

    else:

      net.train()
      for i in range(epoch):

        out, mu, sigma = net(x, q)

        Mseloss, kldloss = loss_BVAE(out, target, mu, sigma)

        optimizer.zero_grad()
        net.zero_grad()

        loss = (Mseloss + beta * kldloss)

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
    question = test_set[1]
    target = test_set[2]

    x = Variable(torch.from_numpy(observations).to(device, torch.float32))
    target = Variable(torch.from_numpy(target)).to(device, torch.float32)
    q = Variable(torch.from_numpy(question)).to(device, torch.float32)
    x = x.reshape(-1, 10)
    q = q.reshape(-1, 10)
    target = target.reshape(-1, 1)

    out, mu, sigma = model(x, q)

    Mseloss, kldloss = loss_BVAE(out, target, mu, sigma)

    return Mseloss
    
    
def plot_paper_results():
    samples = 100000
    n_observation= 10
    n_questions = 10
    test_samples= 5000

    # load saved datasets
    print("loading pre-generated datasets")
    compl_dataset_path = PATH / "complete" / "data" / "dataset.pk1"
    with open(compl_dataset_path,'rb') as f: a = pickle.load(f)
    compl_set_test = (a[0][:test_samples],a[1][:test_samples], a[2][:test_samples])

    incompl_dataset_path = PATH / "incomplete" / "data" / "dataset.pk1"
    with open(incompl_dataset_path,'rb') as f: a = pickle.load(f)
    incompl_set_test = (a[0][:test_samples],a[1][:test_samples], a[2][:test_samples])
    
    #Import HyperParams from json
    with open(PATH / 'params.json') as json_data:
      d = json.load(json_data)
      json_data.close()
    latent_max = d['latent max']+1

    # plot complete tomography
    print("loading pre-trained models")
    error_c = np.zeros(latent_max)
    for k in range(latent_max):
      n_latent = k
      model = SciNet(observation_size=n_observation, encode_h1=100, encode_h2=100, 
                  decoder_input=(n_questions+n_latent), decode_h1=100, decode_h2=100, 
                  output_size=1,n_latent_var=n_latent).to(device)
      model_name = "%i_latent.pt" % k
      model_path = PATH / "complete" / "models" / model_name
      model.load_state_dict(torch.load(model_path))
      model.eval()
      error_c[k] = test(model, compl_set_test)
    # plt.bar(np.arange(0,latent_max),error, color ='blue', width = 0.4)

    # plot incomplete tomography
    error_i = np.zeros(latent_max)
    for k in range(latent_max):
      n_latent = k
      model = SciNet(observation_size=n_observation, encode_h1=100, encode_h2=100, 
                  decoder_input=(n_questions+n_latent), decode_h1=100, decode_h2=100, 
                  output_size=1,n_latent_var=n_latent).to(device)
      model_name = "%i_latent.pt" % k
      model_path = PATH / "incomplete" / "models" / model_name
      model.load_state_dict(torch.load(model_path))
      model.eval()
      error_i[k] = test(model, incompl_set_test)
    
    p1 = plt.bar(np.arange(0, latent_max), np.sqrt(error_c),
                 color='blue', width=0.8, align="center")
    p2 = plt.bar(np.arange(0, latent_max), np.sqrt(error_i),
                 color='orange', width=0.4, align="edge")
    plt.ylabel("Error")
    plt.title("One qubit.")
    plt.xlabel("Number of latent neurons")
    plt.legend((p1[0],p2[0]),("Tom. complete", "Tom. incomplete"))
    plt.show()

def run_all_scenarios():
  #init
  samples = 100000
  n_observation= 10
  n_questions = 10
  test_samples= 5000
  scenario = "one qubit complete"
  n_qubits = 1
  incomplete_tomography=[False,False]
  path = PATH / "1 qubit" / "complete"
  latent_max = 6
  
  for i in range(5):
    if i == 0:
      samples = 100000
      n_observation = 10
      n_questions = 10
      test_samples = 5000
      scenario = "one qubit complete"
      n_qubits = 1
      incomplete_tomography = [False, False]
      path = PATH / "1 qubit" / "complete"
      latent_max = 6
    elif i == 1: 
      samples = 100000
      n_observation = 10
      n_questions = 10
      test_samples = 5000
      scenario = "one qubit incomplete"
      n_qubits = 1
      incomplete_tomography=[2,False]
      path = PATH / "1 qubit" / "incomplete"
      latent_max = 6
    elif i == 2:
      samples = 500000
      n_observation = 30
      n_questions = 30
      test_samples = 10000
      scenario = "two qubit complete"
      n_qubits = 2
      incomplete_tomography=[False,False]
      path = PATH / "2 qubit" / "complete"
      latent_max = 10
    elif i == 3:
      samples = 500000
      n_observation = 30
      n_questions = 30
      test_samples = 10000
      scenario = "two qubit 2-dim subspace"
      n_qubits = 2
      incomplete_tomography=[2,False]
      path = PATH / "2 qubit" / "2 dim"
      latent_max = 10
    elif i == 4:
      samples = 500000
      n_observation = 30
      n_questions = 30
      test_samples = 10000
      scenario = "two qubit 3-dim subspace"
      n_qubits = 2
      incomplete_tomography=[3,False]
      path = PATH / "2 qubit" / "3 dim"
      latent_max = 10

    print("Generating dataset for", scenario, "with", samples, "samples")
    a, b, c = create_data(n_qubits, n_observation, n_questions, samples, incomplete_tomography=incomplete_tomography)
    Path.mkdir(path, exist_ok=True)
    dataset_path = path / "dataset.pk1"
    print("Saving dataset to", str(dataset_path))
    with open(dataset_path,'wb') as f: pickle.dump(a, f)
    
    observations = a[0][test_samples:]
    question =a[1][test_samples:]
    target = a[2][test_samples:]
    test_set = (a[0][:test_samples],a[1][:test_samples], a[2][:test_samples])
                    
    #Import HyperParams from json
    with open(PATH / 'params.json') as json_data:
      d = json.load(json_data)
      json_data.close()
    learning_rate1 = d['learning rate1']
    learning_rate2 = d['learning rate2']
    num_epochs1 = d['num epochs1']
    num_epochs2 = d['num epochs2']
    beta = d['beta']

    
    error = np.zeros(latent_max)
    for k in range(latent_max):
    
        n_latent = k
        print("Training model for", scenario, "with", n_latent, "latent nuerons")
        #Initializing the network
        model = SciNet(observation_size=n_observation, encode_h1=100, encode_h2=100, 
                    decoder_input=(n_questions+n_latent), decode_h1=100, decode_h2=100, 
                    output_size=1,n_latent_var=n_latent).to(device)
        
        err = []
        train(model, num_epochs1, learning_rate1, beta, observations, question,
              target, err, 512)
        train(model, num_epochs2, learning_rate2, beta, observations, question,
              target, err, 512)
        plt.clf
        plt.cla
        x_list = [u for u in range(0,len(err))]
        print(x_list)
        print(err)
        plt.plot(x_list,err)
        print("Plotting training error for " + scenario + " with " + str(k) + " latent neurons")
        plt.title("Training error for " + scenario + " with " + str(k) + " latent neurons")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt_name = "%i_latent.png" % k
        plt_path = path / "plots" / plt_name
        Path.mkdir(plt_path.parent, exist_ok=True)
        plt.savefig(plt_path)
        plt.close()

        # Save model
        model_name = "%i_latent.pt" % k
        model_path = path / "models" / model_name
        Path.mkdir(model_path.parent, exist_ok=True)
        torch.save(model.state_dict(), model_path)

        print("Testing model for", scenario, "with", n_latent, "latent nuerons")
        error[k] = test(model, test_set)
        print("Error", error[k])

    plt.bar(np.arange(0,latent_max), np.sqrt(error), color ='maroon', width = 0.4)
    plt_path = path / "plots" / "final_bar.png"
    print("Saving bar plot for", scenario, "at", plt_path)
    plt.savefig(plt_path)
    # plt.show()
    

def main():
    # uncomment to generate report plot
    # plot_paper_results()
    # return()

    # uncomment to run for all scenarios
    # run_all_scenarios()
    # return
    
    samples = 100000
    n_observation= 10
    n_questions = 10
    test_samples= 5000
    regenerate = True

    a = []
    dataset_path = PATH / "data" / "dataset.pk1"
    if regenerate:
      print("Generating dataset with",samples,"samples")
      a, b, c = create_data(1, n_observation, n_questions, samples, incomplete_tomography=[False,False])
      Path.mkdir(dataset_path.parent, exist_ok=True)
      with open(dataset_path,'wb') as f: pickle.dump(a, f)
    else:
      print("Loading pre-generated dataset")
      with open(dataset_path,'rb') as f: a = pickle.load(f)

        
    observations = a[0][test_samples:]
    question =a[1][test_samples:]
    target = a[2][test_samples:]
    test_set = (a[0][:test_samples],a[1][:test_samples], a[2][:test_samples])
                    
        
    #Import HyperParams from json
    with open(PATH / 'params.json') as json_data:
      d = json.load(json_data)
      json_data.close()
    learning_rate1 = d['learning rate1']
    learning_rate2 = d['learning rate2']
    num_epochs1 = d['num epochs1']
    num_epochs2 = d['num epochs2']
    beta = d['beta']
    latent_max = d['latent max']+1
    
    print("Training...")
    error = np.zeros(latent_max)
    for k in range(latent_max):
      print("Training with", k, "latent neurons")
      n_latent = k
      #Initializing the network
      model = SciNet(observation_size=n_observation, encode_h1=100, encode_h2=100, 
                  decoder_input=(n_questions+n_latent), decode_h1=100, decode_h2=100, 
                  output_size=1,n_latent_var=n_latent).to(device)
      
      #Loss and optimizer
      err = []
      train(model, num_epochs1, learning_rate1, beta, observations, question,
            target, err, 512)
      train(model, num_epochs2,learning_rate2, beta, observations, question,
            target,err, 512)
      plt.clf
      plt.cla
      x_list = [i for i in range(0,len(err))]
      print(x_list)
      print(err)
      plt.plot(x_list,err)
      plt.title("Training error with %i latent neurons" % k)
      plt.xlabel("Epochs")
      plt.ylabel("Error")
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

    plt.bar(np.arange(0, latent_max), np.sqrt(error), color='maroon', width = 0.4)
    plt_path = PATH / "plots" / "final_bar.png"
    plt.savefig(plt_path)
    plt.show()
    
    #validating data
    
    
    

    # testloss += np.sum(np.pow(target - out[0,:],2))
    # testloss = testloss/25
    # print(testloss)
  
   



if __name__ == '__main__':
  main()
