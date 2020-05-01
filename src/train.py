import argparse
import dsl as op
import pprint as pp
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard as tb

from env import to_program, RobustFillEnv
from models import RobustFill, ReinforceRobustFill
from utils import sample_example
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical

def max_program_length(expected_programs):
    return max([len(program) for program in expected_programs])


def train_reinforce(args, policy, optimizer, env, checkpoint_filename,
    checkpoint_step_size, checkpoint_print_tensors):
    

    # Get logger if available!
    from os import path
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'),
            flush_secs=1)

    if args.continue_training:
        policy.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), checkpoint_filename))
        )
    
    def select_action(inp, hidden, output_all_hidden, out_len, num_examples):
        program_embedding, hidden = policy.predict_next(inp, hidden, 
            output_all_hidden, out_len, num_examples)
        probs = F.softmax(program_embedding, dim=1).squeeze(0)
        m = Categorical(probs)
        action = m.sample()
        policy.saved_log_probs.append(m.log_prob(action))
        return action.item(), program_embedding, hidden

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    policy.set_device(device)
    policy = policy.to(device)
    policy.train()
    global_iter = 0
    num_examples = 4
    running_reward = 0
   
    # Get an minibatch by interacting with the environment
    # no limit on performance: keep interacting with env!
    while True:
    # Train from the results of the minibatch
        
        for i_episode in range(args.batch_size):
            state, ep_reward = env.reset(), 0
            output_all_hidden, hidden, out_len = policy.encode_io(state)
            # initial input is zero vector
            decoder_input = torch.stack([torch.zeros(1, policy.program_size) for _ in range(hidden[0].size()[1])])
            decoder_input = decoder_input.to(hidden[0].device)

            # here we should decide to use teacher or not - take trg and make a one-hot encoding
            done=False
            while not done: # not end of sequence yet
                action, program_embedding, hidden = select_action(decoder_input, 
                    hidden, output_all_hidden, out_len, num_examples)
                state, reward, done, _ = env.step(action)
                policy.rewards.append(reward)
                ep_reward += reward
                decoder_input = torch.stack([F.softmax(p, dim=1) 
                    for p in program_embedding.split(1) for _ in range(num_examples)])

            if (ep_reward > 0):
                print("I did it yay!!!!")
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            finish_episode(policy, optimizer)
            if i_episode % args.checkpoint_step_size == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, ep_reward, running_reward))
            if running_reward > .995:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                break

def finish_episode(policy, optimizer, eps=1, gamma=.95):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to(policy.device)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

    
def train(args, robust_fill, optimizer, dataloader, checkpoint_filename, 
        checkpoint_step_size, checkpoint_print_tensors):

    # Get logger if available!
    from os import path
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'),
            flush_secs=1)


    if args.continue_training:
        robust_fill.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), checkpoint_filename))
        )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    robust_fill.set_device(device)
    robust_fill = robust_fill.to(device)
    robust_fill.train()
    token_tables = op.build_token_tables()
    global_iter = 0
    # No number of iterartions here - just train for a real long time
    while True:
        for b in dataloader:

            optimizer.zero_grad()

            expected_programs, examples = b
            max_length = max_program_length(expected_programs)

            # teacher learning
            padded_tgt_programs = torch.LongTensor([
                    [program[i] if i < len(program) else 0 for i in range(max_length)]
                    for program in expected_programs
            ]).to(device)

            # Output: program_size x b x #ops, need to turn b x #ops x #p_size
            actual_programs = robust_fill(examples, padded_tgt_programs).permute(1, 2, 0)  
            padding_index = -1
            padded_expected_programs = torch.LongTensor([
                    [program[i] if i < len(program) else padding_index for i in range(max_length)]
                    for program in expected_programs
            ]).to(device)


            loss = F.cross_entropy(actual_programs, padded_expected_programs, 
                ignore_index=padding_index
            )
            loss.backward()
            if args.grad_clip > 0.:
                torch.nn.utils.clip_grad_norm_(robust_fill.parameters(), args.grad_clip)
            optimizer.step()

            # Debugging information
            if train_logger is not None:
                train_logger.add_scalar('loss', loss.item(), global_iter)

            if global_iter % checkpoint_step_size == 0:
                print('Checkpointing at batch {}'.format(global_iter))
                print('Loss: {}'.format(loss.item()))

                # note this code will not print correct if more than 1 printed
                if checkpoint_print_tensors:
                    print_programs(expected_programs[0],
                        actual_programs.permute(2, 0, 1)[:len(expected_programs[0]), :1, :],
                        train_logger, token_tables.token_op_table, global_iter
                    )


                if checkpoint_filename is not None:
                    print('Saving to file {}'.format(checkpoint_filename))
                    torch.save(robust_fill.state_dict(), checkpoint_filename)
                print('Done checkpointing model')
            global_iter += 1


def print_programs(expected_programs, actual_programs, train_logger, tok_op, global_iter):
    tokens = torch.argmax(actual_programs.permute(1, 0, 2), dim=-1)
    tokens = tokens[0].tolist()
    if train_logger is not None:
        train_logger.add_text('Expected program', 
            str(expected_programs), global_iter)
        train_logger.add_text('Actual program', str(tokens), global_iter)
    else:
        print("Expected program: ")
        print(expected_programs)
        print("Actual program: ")
        print(str(tokens))

    try:
        prog = str(to_program(expected_programs, tok_op))
    except Exception:
        prog = "Could not parse expected program"
    if train_logger is not None:
        train_logger.add_text('Expected Parsed Program', prog, global_iter)
    else:
        print("Expected parsed program:")
        print(prog)

    try:
        prog = str(to_program(tokens, tok_op))
    except Exception:
        prog = "Could not parse program"

    if train_logger is not None:
        train_logger.add_text('Actual Parsed program', prog, global_iter)
    else:
        print("Actual parsed program:")
        print(prog)

# Kinda an odd 
class RobustFillDataset(Dataset):

    # idea - use sample to get number of desired programs, store in a list
    def __init__(self, token_tables, padding_index=-1, max_exp=3, max_characters=50, d=100):
        self.token_tables = token_tables
        self.padding_index = padding_index
        self.max_exp = max_exp
        self.max_characters = max_characters
        self.programs = [self._sample() for _ in range(d)]


    def __len__(self):
        return len(self.programs)

    def __getitem__(self, idx):
        return self.programs[idx]

    def _sample(self):
        example = sample_example(self.max_exp, self.max_characters)
        program = example.program.to_tokens(self.token_tables.op_token_table)
        strings = [
            (op.tokenize_string(input_, self.token_tables.string_token_table),
             op.tokenize_string(output, self.token_tables.string_token_table))
            for input_, output in example.strings
        ]
        return (program, strings)

# a simple custom collate function, just put them into a list!
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return (data, target)

def sample_full(token_tables, batch_size, max_expressions, max_characters):
    program_batch, strings_batch = [], []

    for _ in range(batch_size):
        example = sample_example(max_expressions=max_expressions, max_characters=max_characters)
        program = example.program.to_tokens(token_tables.op_token_table)
        strings = [
            (op.tokenize_string(input_, token_tables.string_token_table),
             op.tokenize_string(output, token_tables.string_token_table))
            for input_, output in example.strings
        ]
        program_batch.append(program)
        strings_batch.append(strings)
    return program_batch, strings_batch


def train_full(args):

    token_tables = op.build_token_tables()
    robust_fill = RobustFill(string_size=len(op.CHARACTER), 
        string_embedding_size=args.embedding_size, hidden_size=args.hidden_size, 
        program_size=len(token_tables.op_token_table),
    )

    if (args.optimizer == 'sgd'):
        optimizer = optim.SGD(robust_fill.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(robust_fill.parameters(), lr=args.lr)

    train_dataset = RobustFillDataset(token_tables, d=args.number_progs)
    prog_dataloaer = DataLoader(dataset=train_dataset,
                      batch_size=args.batch_size,
                      shuffle=True,
                      collate_fn=my_collate,
                      num_workers=4
                    )

    train(args, robust_fill=robust_fill, optimizer=optimizer, 
        dataloader=prog_dataloaer, 
        checkpoint_filename=args.checkpoint_filename,
        checkpoint_step_size=args.checkpoint_step_size, 
        checkpoint_print_tensors=args.print_tensors,
    )


def train_rl(args):
    token_tables = op.build_token_tables()
    robust_fill = ReinforceRobustFill(string_size=len(op.CHARACTER), 
        string_embedding_size=args.embedding_size, hidden_size=args.hidden_size, 
        program_size=len(token_tables.op_token_table),
    )

    if (args.optimizer == 'sgd'):
        optimizer = optim.SGD(robust_fill.parameters(), lr=args.lr)
    else:
        optimizer = optim.Adam(robust_fill.parameters(), lr=args.lr)

    env = RobustFillEnv()
    train_reinforce(args, policy=robust_fill, optimizer=optimizer, 
        env=env, 
        checkpoint_filename=args.checkpoint_filename,
        checkpoint_step_size=args.checkpoint_step_size, 
        checkpoint_print_tensors=args.print_tensors,
    )
# run gives passable argparser interface, no random seed!
def run(args):
    train_full(args)

def run_rl(args):
    train_rl(args)

def main():
    parser = argparse.ArgumentParser(description='Train RobustFill.')
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('--log_dir')
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--hidden_size', default=512)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--embedding_size', default=128)
    parser.add_argument('--checkpoint_filename', default='./checkpoint.pth')
    parser.add_argument('--checkpoint_step_size', default=8)
    parser.add_argument('--print_tensors', default=True)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--grad_clip', default=.25)
    parser.add_argument('--number_progs', default=1000)
    parser.add_argument('--rl', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1337)
    random.seed(420)
    if (args.rl):
        train_rl(args)
    else:
        train_full(args)


if __name__ == '__main__':
    main()