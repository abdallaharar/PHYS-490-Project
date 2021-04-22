# Discovering Physical Concepts with Neural Networks
PHYS 490 - Winter 2021

Group 1: Abdallah Arar, Arjun Bakshi, Mitchell Faguy, Nathan Fischer

Based on: [R. Iten, T. Metger, H. Wilming, L. del Rio, and R. Renner, (2020), arXiv:1807.10300v3](https://arxiv.org/pdf/1807.10300.pdf)

## Structure
Project is split into three main components: SciNet and two physical examples. SciNet's architecture is defined in the paper, and is implemented as a pytorch neural network that is imported and used in each example. Each example's directory contains:
* `generator.py`, a program which simulates a physical problem and generates a dataset of observations,
* `main.py`, a program which trains SciNet based on the generated dataset and given parameter
* `params.json`, where parameters used in `main.py` are defined
* results from various scenarios

### SciNet
Defined in `./SciNet/SciNet_Module.py`, neural network used to learn physical examples. 

### Quantum Tomography
One of the examples outlined in the paper and found in `./Quantum`.

To run, use:
```bash
python ./Quantum/main.py
```

Results from 1 and 2 qubit scenarios are stored in `./Quantum/1 qubit/` and `./Quantum/2 qubit/` respectively, with each containing subdirectories corresponding to tomographically complete and incomplete scenarios. The trained generated dataset, trained model, and plots for each scenario can be found in the scenario's subdirectory. The saved models and data can be used for further analysis without needing to retrain. 

### Damped Oscillator
Another example outlined in the paper and found in `./Damped_Oscillator`. Plots for various scenarios can be found in subdirectories. 

To run, use:
```bash
python ./Damped_Oscillator/main.py
```

## Dependencies
* json
* argparse
* torch
* sys
* pathlib
* numpy
* matplotlib
