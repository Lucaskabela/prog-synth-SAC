import argparse
import dsl as op
import pprint as pp
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard as tb
import numpy as np

from env import to_program, RobustFillEnv, num_consistent
from models import RobustFill, ValueNetwork, SoftQNetwork
from utils import sample_example, Beam, ReplayBuffer
from torch.utils.data import Dataset, DataLoader

def max_program_length(expected_programs):
    return max([len(program) for program in expected_programs])

def train_sac(args, policy, value, tgt_value, q_1, q_2, policy_opt, value_opt,
    q_1_opt, q_2_opt, replay_buffer, env, 
    checkpoint_filename, checkpoint_step_size, checkpoint_print_tensors):
    
    max_frames = 10_000_000
    max_steps = 25
    frame_idx = 0
    num_examples = 4
    batch_size = args.batch_size
    rewards = []
    while frame_idx < max_frames:
        state = env.reset()
        output_all_hidden, hidden = policy.encode_io(state)
        episode_reward = 0
        decoder_input = [policy.decoder_embedding(torch.tensor([policy.program_size], 
                        device=policy.device, dtype=torch.long)) for _ in range(hidden[0].size()[1])
        ]
        for step in range(max_steps):

            action, program_embedding, output_all_hidden, next_hidden = policy.select_action(decoder_input, hidden, output_all_hidden)
            next_state, reward, done, _ = env.step(action)
            idx_next = [action]
            index_input = torch.tensor(idx_next, device=policy.device, dtype=torch.long)
            next_decoder_input = [policy.decoder_embedding(p) for p in index_input.split(1) for _ in range(num_examples)]
            replay_buffer.push((decoder_input, hidden, output_all_hidden), action, reward, (next_decoder_input, next_hidden), done)
            hidden = next_hidden
            decoder_input = next_decoder_input
            state = next_state
            episode_reward += reward
            frame_idx += 1
            
            if len(replay_buffer) > batch_size:
                update(policy, value, tgt_value, q_1, q_2, policy_opt, value_opt, q_1_opt, q_2_opt,
                    replay_buffer, batch_size)
            
            if done:
                break
        print("Reward: ")
        print(episode_reward)
        rewards.append(episode_reward)

def update(policy, value, tgt_value, q_1, q_2, policy_opt, value_opt, q1_opt, q2_opt,
    replay_buffer, batch_size, gamma=0.99, soft_tau=1e-2,):
    
    q_1obj = torch.nn.MSELoss()
    q_2obj = torch.nn.MSELoss()
    v_obj = torch.nn.MSELoss()

    device = policy.device
    states, action, reward, next_states, done = replay_buffer.sample(batch_size)
    decoder_input = [inp_ for inp_, _, _ in states]
    hidden = [hidden for _, hidden, _ in states]
    output_all_hidden = [output_all_hidden for _, _, output_all_hidden in states]
    next_decoder_input = [inp_ for inp_, _ in next_states]
    next_hidden = [hidden for _, hidden in next_states]
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    new_action, log_prob, z, mean, log_std = policy.evaluate(decoder_input, hidden, output_all_hidden)

    p_q1 = q_1(decoder_input, hidden, action)
    p_q2 = q_2(decoder_input, hidden, action)
    p_v  = value(decoder_input, hidden)

    # Training Q Function
    target_value = tgt_value(next_decoder_input, next_hidden)
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = q_1obj(p_q1, target_q_value.detach())
    q_value_loss2 = q_2obj(p_q2, target_q_value.detach())

    # Note this is probably wrong
    q1_opt.zero_grad()
    q_value_loss1.backward(retain_graph=True)
    q1_opt.step()
    q2_opt.zero_grad()
    q_value_loss2.backward(retain_graph=True)
    q2_opt.step()  

    # Training V networks
    predicted_new_q_value = torch.min(q_1(decoder_input, hidden, new_action), q_2(decoder_input, hidden, new_action))
    target_value_func = predicted_new_q_value - log_prob
    value_loss = v_obj(p_v, target_value_func.detach())

    value_opt.zero_grad()
    value_loss.backward(retain_graph=True)
    value_opt.step()

    # Train actor network
    policy_loss = (log_prob - predicted_new_q_value).mean()
    policy_opt.zero_grad()
    policy_loss.backward(retain_graph=True)
    policy_opt.step()
    
    # Copy parameters
    for target_param, param in zip(tgt_value.parameters(), value.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
    print("Update succesfully!")

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
            output_all_hidden, hidden = policy.encode_io(state)
            decoder_input = [policy.decoder_embedding(torch.tensor([policy.program_size], 
                        device=policy.device, dtype=torch.long)) for _ in range(hidden[0].size()[1])
            ]
            done=False
            while not done: # not end of sequence yet
                try:
                    action, program_embedding, output_all_hidden, hidden = policy.select_action(decoder_input, 
                        hidden, output_all_hidden)
                except RuntimeError:
                    print("Got some whack stuff again")
                    print("state starts off as: ")
                    print(state)
                    print("Decoder input: ")
                    print(decoder_input)
                    print("The index input was: ")
                    print(index_input)
                    print("Hidden is: ")
                    print(hidden)
                    print("The hidden from the encoding: ")
                    print(output_all_hidden)
                state, reward, done, _ = env.step(action)
                policy.rewards.append(reward)
                ep_reward += reward
                idx_next = [action]
                index_input = torch.tensor(idx_next, device=policy.device, dtype=torch.long)
                decoder_input = [policy.decoder_embedding(p) for p in index_input.split(1) for _ in range(num_examples)]

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
        string_embedding_size=args.embedding_size, decoder_inp_size=128,
        hidden_size=args.hidden_size, 
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


def train_dqn(args):
    token_tables = op.build_token_tables()
    policy = RobustFill(string_size=len(op.CHARACTER), 
        string_embedding_size=args.embedding_size, decoder_inp_size=128,
        hidden_size=args.hidden_size, 
        program_size=len(token_tables.op_token_table),
    )
    q_1 = SoftQNetwork(128, len(token_tables.op_token_table), args.hidden_size)
    q_2 = SoftQNetwork(128, len(token_tables.op_token_table), args.hidden_size)
    value = ValueNetwork(128, args.hidden_size)
    tgt_value = ValueNetwork(128, args.hidden_size)

    if (args.optimizer == 'sgd'):
        policy_opt = optim.SGD(policy.parameters(), lr=args.lr)
        val_opt = optim.SGD(value.parameters(), lr=args.lr)
        q_1_opt = optim.SGD(q_1.parameters(), lr=args.lr)
        q_2_opt = optim.SGD(q_2.parameters(), lr=args.lr)
    else:
        policy_opt = optim.Adam(policy.parameters(), lr=args.lr)
        val_opt = optim.Adam(value.parameters(), lr=args.lr)
        q_1_opt = optim.Adam(q_1.parameters(), lr=args.lr)
        q_2_opt = optim.Adam(q_2.parameters(), lr=args.lr)

    env = RobustFillEnv()
    replay_buffer_size = 1000000
    replay_buffer = ReplayBuffer(replay_buffer_size)
    train_sac(args, policy, value, tgt_value, q_1, q_2, policy_opt, val_opt,
        q_1_opt, q_2_opt, replay_buffer, env, 
        args.checkpoint_filename, args.checkpoint_step_size, args.print_tensors
    )


def train_rl(args):
    token_tables = op.build_token_tables()
    robust_fill = RobustFill(string_size=len(op.CHARACTER), 
        string_embedding_size=args.embedding_size, decoder_inp_size=128,
        hidden_size=args.hidden_size, 
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

def sample(token_tables, max_expressions=3, max_characters=50):
    example = sample_example(max_expressions, max_characters)
    program = example.program.to_tokens(token_tables.op_token_table)
    strings = [
        (op.tokenize_string(input_, token_tables.string_token_table),
         op.tokenize_string(output, token_tables.string_token_table))
        for input_, output in example.strings
    ]
    return (program, strings)


def eval(model, token_tables, num_samples=1000, beam_size=1, em=True):
    model.eval()
    num_match = 0
    for _ in range(num_samples):
        expected_programs, examples = sample(token_tables)
        max_len = len(expected_programs[0] + 5) # do not allow generating progams longer than max_len! 
        beam = Beam(beam_size)
        res_beam = Beam(beam_size)
        # Elements in beam are tuples ([sequence], output_all_hidden, hidden)
        # Scores are log probs.  Start with SOS, 
        output_all_hidden, hidden = model.encode_io(examples)
        beam.add(([model.program_size], output_all_hidden, hidden), 0)
        iteration = 0
        while (iteration < max_len):
            next_beam = Beam(beam_size)     
            for elt, score in beam.get_elts_and_scores():
                sequence, output_all_hidden, hidden = elt
                inp_idx = sequence[-1]
                decoder_input = [model.decoder_embedding(torch.tensor([inp_idx], 
                    device=model.device, dtype=torch.long)) for _ in range(hidden[0].size()[1])
                ]
                probs, output_all_hidden, hidden = model.next_probs(decoder_input, hidden, output_all_hidden)
                probs = F.log_softmax(probs.squeeze(0), dim=-1)
                scored, idx = torch.topk(probs, dim=-1)
                for next_score, next_idx in zip(scored, idx):
                    if (next_idx == 0): #EOS!
                        next_sequence = copy.deepcopy(sequence)
                        next_sequence.append(next_idx)
                        res_beam.add(next_sequence, score + next_score)
                    else:
                        next_sequence = copy.deepcopy(sequence)
                        next_sequence.append(next_idx)
                        next_beam.add((next_sequence, output_all_hidden, hidden), score + next_score)
                beam = next_beam

        for sequence, _ in res_beam.get_elts_and_scores():
            if em:
                if (sequence == expected_programs):
                    num_match+=1
                    break
            else:
                if (num_consistent((expected_programs, examples), actual_programs) == 4):
                    num_match+=1
                    break

    print('{}\% Accuracy!'.format((num_match/num_samples) * 100))



# run gives passable argparser interface, no random seed!
def run(args):
    train_full(args)

def run_rl(args):
    train_rl(args)

def run_sac(args):
    train_sac(args)

def run_eval(args):
    token_tables = op.build_token_tables()
    model = RobustFill(string_size=len(op.CHARACTER), 
        string_embedding_size=args.embedding_size, hidden_size=args.hidden_size, 
        program_size=len(token_tables.op_token_table),
    )
    if args.continue_training:
        model.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), checkpoint_filename))
        )

    eval_em(model, token_tables)

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
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--sac', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1337)
    random.seed(420)
    if (args.rl):
        train_rl(args)
    elif (args.eval):
        run_eval(args)
    elif (args.sac):
        train_dqn(args)
    else:
        train_full(args)

if __name__ == '__main__':
    main()