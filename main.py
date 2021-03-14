import torch
import numpy as np
import Scinet
import json

def main():
  with open('params.json') as json_data:
    d = json.loads(json_data)
    json_data.close()
    pprint(d)
  
  device = torch.device('cuda' if torch.cuda.is.available() else 'cpu')
  




if __name__ == '__main__':
  main()
