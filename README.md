# prog-synth-SAC

**Lucas Kabela**

_A research project investigating applications of Soft Actor Critic in Neural Program Synthesis_

[Final Writeup](./writeups/nps_with_sac.pdf)
---

## References:
This project is largely following the work of Robustfill by [Delvin et al. 17](https://arxiv.org/abs/1703.07469)

DSL and base model was largerly derived from [here](https://github.com/yeoedward/Robust-Fill/)

SAC imlementation derived from [here](https://github.com/kengz/SLM-Lab)

## Getting Started

### Prerequisites
The following base packages were used to run this repository: 

 - [Python](https://www.python.org/) - 3.6+
 - [Numpy](https://numpy.org/) - 1.16.4+
 - [PyTorch](https://pytorch.org/) - Lastest (1.2.0+)

### First Steps
This repository contains code for training a model for neural program synthesis.  We provide 
supervised learning and reinforcement learing algorithms REINFORCE and SAC.  To train a model, run
    ` python train.py [--sac] [--reinforce] `
and to evaluate 
    ` python train.py  --eval -c --checkpoint_filename [policy to evaluate]`

See train.py for more command line arguments related to hyperparameters.  Alternatively, we have provided
a notebook, train.ipynb which requires the code in the src folder to run.  This code, with supervised 
training from scratch should reach a loss of 4 within minutes, and below 2 in about 30 hours with hyperparameters
provided

## Repository Structure

    .
    ├── model_zoo              # Pretrained models used to produce results
    |
    ├── results                # Results of experiments in csv
    |
    ├── src                    # source code for the project
    |
    ├── writeups               # Project proposal and final report
    |
    ├── LICENSE
    | 
    └── README.md


### Src
Contains the code required for training the models and running experiments, as well as 
executable notebooks.

#### Training in Colab:
Note, this program has long training times, so if you are running in colab, to avoid disconnection from
inactivity and take advantage of full training time, please add the following
to the console:
```
function ClickConnect(){
    console.log("Clicking");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect,60000)
```

### Results
[Contains the raw data from experiments](./results/NPS_with_SAC_data.csv)

### Writeup
This folder contains the project proposal and final report/writeup describing the 
methodology, intuitions, related works and results.

## License:
This project is licensed under the terms of the MIT license.
