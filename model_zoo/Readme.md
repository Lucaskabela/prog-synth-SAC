This model zoo contains saved pytorch files for different models

**checkpoint_adam_lr5e4_loss2_2_20k.pth** - A policy/Robustfill model trained with supervised learning for 20k batches, 128 samples per batch with 4
examples per sample.  Optimizer was Adam with lr 5e-4.  Hyper parameters are a hidden size of 512, 	embedding sizes of 128, and gradient clipping of 
.25 to prevent exploding gradients.  This model serves as a pretrained policy network for RL methods.

**checkpoint_adam_lr5e4_loss2_2_40k.pth** - A policy/Robustfill model trained with supervised learning for 40k batches, 128 samples per batch with 4
examples per sample.  Optimizer was Adam with lr 5e-4.  Hyper parameters are a hidden size of 512, 	embedding sizes of 128, and gradient clipping of 
.25 to prevent exploding gradients.  This model serves as a baseline supervised network for RL method comparisson, and was continued training from *checkpoint_adam_lr5e4_loss2_2_20k.pth*

**checkpoint_adam_lr3e4_loss2_0.pth** - Ignore this, not important