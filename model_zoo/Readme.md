This model zoo contains saved pytorch files for different models

**checkpoint_her.pth** - A policy/Robustfill model trained with SAC + HER for another 1.28 million programs.  Optimizer was Adam with lr 1e-6, hyper parameters were 
hidden size 512, embedding size of 128 and gradient clipping of .25.  This model was trained with *checkpoint_network_pretrained.pth* as a pretrained policy

**checkpoint_network_pretrained.pth** - A policy/Robustfill model trained with supervised learning for 20k batches, 128 samples per batch with 4
examples per sample or 2.56 million programs.  Optimizer was Adam with lr 5e-4.  Hyper parameters are a hidden size of 512, embedding sizes of 128, and gradient clipping of .25 to prevent exploding gradients.  This model serves as a pretrained policy network for RL methods.

**checkpoint_reinforce.pth** - A policy/Robustfill model trained with REINFORCE for another 1.28 million programs.  Optimizer was Adam with lr 1e-6, hyper parameters were hidden size 512, embedding size of 128 and gradient clipping of .25.  This model was trained with *checkpoint_network_pretrained.pth* as a pretrained policy

**checkpoint_sac.pth** - A policy/Robustfill model trained with SAC for another 1.28 million programs.  Optimizer was Adam with lr 1e-6, hyper parameters were 
hidden size 512, embedding size of 128 and gradient clipping of .25.  This model was trained with *checkpoint_network_pretrained.pth* as a pretrained policy

**checkpoint_supervised_baseline.pth** - A policy/Robustfill model trained with supervised learning for 30k batches, 128 samples per batch with 4
examples per sample or 3.84 million progams.  Optimizer was Adam with lr 5e-4.  Hyper parameters are a hidden size of 512, 	embedding sizes of 128, and gradient clipping of .25 to prevent exploding gradients.  This model serves as a baseline supervised network for RL method comparisson, and was continued training from *checkpoint_network_pretrained.pth*

*q_1_checkpoint_her.pth* - one of two q networks for SAC + HER.

*q_1_checkpoint_sac.pth* - one of two q networks for SAC.

*q_2_checkpoint_her.pth* - one of two q networks for SAC.

*q_2_checkpoint_sac.pth* - one of two q networks for SAC.

*val_checkpoint_reinforce.pth* - the value baseline network for REINFORCE + Baseline.