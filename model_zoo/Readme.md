This model zoo contains saved pytorch files for different models

vanilla_nsp.pth - vanilla neural synthesis programming approach.  Trained with SGD on ~ 33,000 samples

checkpoint_100k_lr1e2.pth - vanilla neural synthesis programming.  Trained with SGD, lr 1e-2, minibatch size 1, hidden size 256, embedding size 32, for 100k minibatches

checkpoint_adam_lr1e2_loss3_0.pth - vanilla neural synthesis programming.  Trained with Adam, lr 1e-2, minibatch size 128, hidden size 256, embedding size 128, for 20k minibatches. Loss consistently around 3.1