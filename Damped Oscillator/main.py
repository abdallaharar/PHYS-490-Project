import json, argparse, torch, sys
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append('../SciNet')
from nn_gen import Net
from data_gen import Data