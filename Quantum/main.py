
def main():

  #Import HyperParams from json
  with open('params.json') as json_data:
    d = json.loads(json_data)
    json_data.close()
    learning_rate = d[model][learning rate]
    num_epochs = d[model][num epochs]

    
  #Specifies to run on GPU if possible
   device = torch.device('cuda' if torch.cuda.is.available() else 'cpu')
  

  #Initializing the network
  model = Scinet(input_size = TODO, num_classes = TODO).to(device)

  #Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr = learning_rate)

  #learning
  for epoch in range(num_epochs):


if __name__ == '__main__':
  main()
