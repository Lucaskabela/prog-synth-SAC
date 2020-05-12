## Src

This file contains source code required to run this project.  There are currently 4 "modes" of execution, 
3 of which are training, and 1 of which is evaluation.  These modes are:
	
	- --eval, evaluates a trained model (requires -c and --checkpoint_filename)

	- --reinforce, uses REINFORCE to train a model

	- --sac, uses Soft actor critic to train a model

	- supervised, uses Supervised learning to train a model (default, no action required)

For the command line arguments, see main.py.  Other files of interest are:

	- dsl.py, which defines the domain specific language, operators, and token tables

	- env.py, which defines the markov decision process / gym like environment for RL

	- models.py, which defines the architectures used

	- train.py, which defines the training loops used to learn the program decoder

### Notebooks
A notebook for colab training is included in *./notebooks*, but requires the contents of this folder
to be executed.  Furthermore, we supply two notebooks for our RL algorithms operating in cartpole to 
demonstrate that they are implemented properly.