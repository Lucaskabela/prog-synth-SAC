
'''
This file defines the primary driver for the experiments, as well as the arg parser
'''
from train import train_reinforce, train_supervised, train_sac, run_eval
import argparse


def main():
    parser = argparse.ArgumentParser(description='RobustFill arguments')

    #----------- Training and saving arguments ------------#
    parser.add_argument('-c', '--continue_training', action='store_true',
        help='Continue training all of the networks in the model'
    )
    parser.add_argument('-cp', '--continue_training_policy', action='store_true',
        help='Continue training just the policy (RobustFill) network in the model'

    )
    parser.add_argument('--checkpoint_filename', type=str, default='./checkpoint.pth',
        help="Name of file to save and load RobustFill from"
    )
    parser.add_argument('--val_checkpoint_filename', type=str, default='./val_checkpoint.pth',
        help="Name of file to save and load value network from (REINFORCE only)"
    )
    parser.add_argument('--q1_checkpoint_filename', type=str, default='./q1_checkpoint.pth',
        help="Name of file to save and load q network from (SAC only)"
    )
    parser.add_argument('--q2_checkpoint_filename', type=str, default='./q2_checkpoint.pth',
        help="Name of file to save and load q network from (SAC only)"
    )
    parser.add_argument('--checkpoint_step_size', default=8)


    #----------- Logging and output arguments ------------#
    parser.add_argument('--log_dir', type=str, default=None, help="Directory to log torchboard")
    parser.add_argument('--print_tensors', default=True, help="Print partial programs")


    #----------- Optimizer arguments ------------#
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for networks")
    parser.add_argument('--optimizer', type=str, default='adam',
        help = "Name of optimizer to use (sgd or adam)"
    )

    #----------- Architecture/model arguments ------------#
    parser.add_argument('--embedding_size', type=int, default=128, 
        help="Dimension of embeddings for char tokens and decoder tokens"
    )
    parser.add_argument('--hidden_size', type=int, default=512, help="Size of hidden layers")
    parser.add_argument('--batch_size', type=int, default=16, help="Size of batches for training")
    parser.add_argument('--grad_clip', type=float, default=.25,  help="Value to clip gradients above")
    parser.add_argument('--number_progs', type=int, default=1000, 
        help="Number of programs to train on for supervised"
    )


    #----------- Mode arguments ------------#
    parser.add_argument('--eval', action='store_true', 
        help="Perform evaluation (requires continue_training and checkpoint_filename"
    )
    parser.add_argument('--beam_size', type=int, default=10, help="Beam size for decoder")
    parser.add_argument('--reinforce', action='store_true', help="Train with REINFORCE")
    parser.add_argument('--sac', action='store_true',  help="Train with SAC")
    parser.add_argument('--her', action='store_true', help="Train with SAC + HER")
    


    args = parser.parse_args()
    if (args.eval):
        run_eval(args)
    elif (args.reinforce):
        train_reinforce(args)
    elif (args.sac):
        train_sac(args)
    else:
        train_supervised(args)

if __name__ == '__main__':
    main()