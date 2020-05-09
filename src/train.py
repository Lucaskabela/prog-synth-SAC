import argparse
import dsl as op
import pprint as pp
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard as tb
import numpy as np
import copy
import math

from env import to_program, RobustFillEnv, num_consistent
from models import RobustFill, ValueNetwork, SoftQNetwork
from utils import sample, sample_example, Beam, Replay_Buffer, HER
from torch.utils.data import Dataset, DataLoader

def max_program_length(expected_programs):
    return max([len(program) for program in expected_programs])

def guard_q_actions(actions, dim):
    '''Guard to convert actions to one-hot for input to Q-network'''
    actions = F.one_hot(actions.long(), dim).float()
    return actions

def calculate_critic_loss(policy, q_1, q_2, tqt_q_1, tgt_q_2, states, i_os, actions, rewards, next_states, done, gamma=.99):
    with torch.no_grad():

        embed_states = [policy.decoder_embedding(torch.LongTensor(state).to(policy.device)) for state in states]
        embed_next_states = [policy.decoder_embedding(torch.LongTensor(next_state).to(policy.device)) for next_state in next_states]

        next_probs, next_actions = policy.calc_log_prob_action(next_states, i_os)
        next_actions = guard_q_actions(next_actions, tqt_q_1.action_space)
        next_q1 = tqt_q_1(embed_next_states, next_actions)
        next_q2 = tgt_q_2(embed_next_states, next_actions)

        min_q_next = (torch.min(next_q1, next_q2) - policy.alpha * next_probs)
        target_q_value = rewards + (1 - done) * gamma * min_q_next

    p_q1 = q_1(embed_states, actions)
    p_q2 = q_2(embed_states, actions)
    q_value_loss1 = F.mse_loss(p_q1, target_q_value)
    q_value_loss2 = F.mse_loss(p_q2, target_q_value)
    return q_value_loss1, q_value_loss2

def calculate_actor_loss(policy, q_1, q_2, states, i_os):
     # Train actor network
    embed_states = [policy.decoder_embedding(torch.LongTensor(state).to(policy.device)) for state in states]
    
    log_probs, actions = policy.calc_log_prob_action(states, i_os, reparam=True)
    q1 = q_1(embed_states, actions)
    q2 = q_2(embed_states, actions)
    min_q = torch.min(q1, q2)
    policy_loss = (policy.alpha * log_probs - min_q).mean()
    return policy_loss, log_probs

def calculate_entropy_tuning_loss(policy, log_pi):
    """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
    is True."""
    alpha_loss = -(policy.log_alpha * (log_pi.detach() + policy.target_entropy)).mean()
    return alpha_loss

def update(args, replay, policy, q_1, q_2, tgt_q_1, tgt_q_2, policy_opt, q_1_opt, q_2_opt, entropy_opt, i_episode, soft_tau=5e-3):
    # Soft Q networks are the problem, detach gradient?  Not sure why they are the cause
    states, i_os, action, reward, next_states, done = replay.sample(args.batch_size)

    action = action.squeeze(1)
    reward = reward.squeeze(1)
    done = done.squeeze(1)

    if not math.isnan(reward.std()):
        reward = (reward - reward.mean()) / (reward.std() + np.finfo(np.float32).eps)

    action = guard_q_actions(action, q_1.action_space)
    q_value_loss1, q_value_loss2 = calculate_critic_loss(policy, q_1, q_2, tgt_q_1, tgt_q_2,
        states, i_os, action, reward, next_states, done) 
    q_1_opt.zero_grad()
    q_value_loss1.backward()
    torch.nn.utils.clip_grad_norm_(q_1.parameters(), .5)
    q_1_opt.step()
    q_2_opt.zero_grad()
    q_value_loss2.backward()
    torch.nn.utils.clip_grad_norm_(q_2.parameters(), .5)
    q_2_opt.step()  

    policy_loss, log_action_probabilities = calculate_actor_loss(policy, q_1, q_2, states, i_os)
    policy_opt.zero_grad()

    # Give q network's time to stableize
    if i_episode > 2_500:
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), .25)
        policy_opt.step()

    alpha_loss = calculate_entropy_tuning_loss(policy, log_action_probabilities) 
    entropy_opt.zero_grad()
    alpha_loss.backward()
    entropy_opt.step()
    
    loss = [policy_loss.item(), q_value_loss1.item(), q_value_loss2.item(), alpha_loss.item()]
    policy.alpha = policy.log_alpha.detach().exp()

    for target_param, param in zip(tgt_q_1.parameters(), q_1.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
    
    for target_param, param in zip(tgt_q_2.parameters(), q_2.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
    # Save and intialize episode history counters
    policy.loss_history.append(policy_loss.item())
    policy.reset()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    return loss

def train_sac_(args, policy, q_1, q_2, tgt_q_1, tgt_q_2, policy_opt, q_1_opt, q_2_opt, 
    entropy_opt, replay_buffer, her, env, train_logger,
    checkpoint_filename, checkpoint_step_size, checkpoint_print_tensors):
    
    scores = []
    global_step = 0
    i_episode = 0
    loss = None
    if args.her:
        her.reset()
    while True:
        # Reset environment and record the starting state
        (state, i_o), ep_reward = env.reset(), 0
        output_all_hidden, hidden = policy.encode_io(i_o)
        decoder_input = [policy.decoder_embedding(torch.tensor([state[-1]], 
                    device=policy.device, dtype=torch.long)) for _ in range(hidden[0].size()[1])
        ]
        done=False
        while not done: # not end of sequence yet
            action, log_prob, output_all_hidden, hidden = policy.select_action(decoder_input, 
                    hidden, output_all_hidden)
            if (global_step < 1_000):
                action = int(random.random() * policy.program_size)

            (next_state, next_i_o), reward, done, _ = env.step(action)
            replay_buffer.add_experience(state, i_o, action, reward, next_state, done)
            if args.her:
                her.add_experience(state, i_o, action, reward, next_state, done)
            state = next_state
            i_o = next_i_o
            policy.rewards.append(reward)
            ep_reward += reward

            decoder_input = [policy.decoder_embedding(torch.tensor([state[-1]], 
                        device=policy.device, dtype=torch.long)) for _ in range(hidden[0].size()[1])
            ]
            global_step+=1

            if len(replay_buffer) > 1_000 + args.batch_size and global_step % 3 == 0:
                loss = update(args, replay_buffer, policy, q_1, q_2, tgt_q_1, tgt_q_2, policy_opt, q_1_opt, q_2_opt,
                    entropy_opt, i_episode)
            if done:
                break

        if args.her:
            her_list = her.backward()
            for exp in her_list:
                replay_buffer.add_experience(*exp)

        # Calculate score to determine when the environment has been solved
        policy.reward_history.append(ep_reward)

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        if train_logger is not None and loss is not None:
            train_logger.add_scalar('average reward', np.mean(policy.reward_history[-100:]), i_episode)
            train_logger.add_scalar('policy loss', loss[0], i_episode)
            train_logger.add_scalar('q_1 loss', loss[1], i_episode)
            train_logger.add_scalar('q_2 loss', loss[2], i_episode)
            train_logger.add_scalar('entropy loss', loss[3], i_episode)

        if i_episode % checkpoint_step_size == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            print('Checkpointing at batch {}'.format(global_iter))
            print('Policy Loss: {}'.format(loss[0]))
            print('Q_1 Loss: {}'.format(loss[1]))
            print('Q_2 Loss: {}'.format(loss[2]))
            print('Entropy Loss: {}'.format(loss[3]))

            if checkpoint_filename is not None:
                print('Saving policy to file {}'.format(checkpoint_filename))
                torch.save(policy.state_dict(), args.checkpoint_filename)
            if q1_checkpoint_filename is not None:
                print('Saving value network to file {}'.format(args.q1_checkpoint_filename))
                torch.save(q1.state_dict(), args.q1_checkpoint_filename)
            if q2_checkpoint_filename is not None:
                print('Saving value network to file {}'.format(args.q2_checkpoint_filename))
                torch.save(q2.state_dict(), args.q2_checkpoint_filename)
            print('Done checkpointing model')

        i_episode += 1
        if running_reward > .995:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break



def update_reinforce(replay, policy, value, policy_opt, value_opt, i_episode, gamma=.99):
    R = 0
    policy_loss = []
    returns = []
    for _, _, r in replay[: :-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns).float().to(policy.device)
    if not math.isnan(returns.std()):
        returns = (returns - returns.mean()) / \
        (returns.std() + np.finfo(np.float32).eps)

    states = [policy.decoder_embedding(torch.LongTensor(state).to(policy.device))
        for state, _, _ in replay]

    vals = value(states)
    with torch.no_grad():
        advantage = returns - vals

    for log_prob, R in zip(policy.saved_log_probs, advantage):
        policy_loss.append(-log_prob * R)


    policy_opt.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    # Give value network time to stableize
    if i_episode > 2_500:
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), .25)
        policy_opt.step()

    value_opt.zero_grad()
    val_loss = F.mse_loss(vals, returns)
    val_loss.backward()
    torch.nn.utils.clip_grad_norm_(value.parameters(), .25)
    value_opt.step()

    # Save and intialize episode history counters
    policy.loss_history.append(policy_loss.item())
    policy.reward_history.append(np.sum(policy.rewards))
    policy.reset()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    return [policy_loss.item(), val_loss.item()]


def train_reinforce_(args, policy, value, pol_opt, value_opt, env, train_logger,
    checkpoint_filename, checkpoint_step_size, checkpoint_print_tensors):
    
    global_iter = 0
    num_examples = 4
    running_reward = 0
    i_episode = 0
    # Get an minibatch by interacting with the environment
    # no limit on performance: keep interacting with env!
    while True:
        (state, i_o), ep_reward = env.reset(), 0
        output_all_hidden, hidden = policy.encode_io(i_o)
        decoder_input = [policy.decoder_embedding(torch.tensor([state[-1]], 
                    device=policy.device, dtype=torch.long)) for _ in range(hidden[0].size()[1])
        ]
        replay = []
        done=False
        while not done: # not end of sequence yet
            action, log_prob, output_all_hidden, hidden = policy.select_action(decoder_input, 
                    hidden, output_all_hidden)

            (next_state, i_o), reward, done, _ = env.step(action)
            replay.append((state, action, reward))
            state = next_state
            policy.rewards.append(reward)
            policy.saved_log_probs.append(log_prob)
            ep_reward += reward
            index_input = torch.tensor([state[-1]], device=policy.device, dtype=torch.long)
            decoder_input = [policy.decoder_embedding(p) for p in index_input.split(1) for _ in range(num_examples)]
            global_iter+=1


        loss = update_reinforce(replay, policy, value, pol_opt, value_opt, i_episode)

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        if train_logger is not None:
            train_logger.add_scalar('average reward', np.mean(policy.reward_history[-100:]), i_episode)
            train_logger.add_scalar('policy loss', loss[0], i_episode)
            train_logger.add_scalar('value loss', loss[1], i_episode)

        if i_episode % checkpoint_step_size == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            print('Checkpointing at batch {}'.format(i_episode))
            print('Policy Loss: {}'.format(loss[0]))
            print('Value Loss: {}'.format(loss[1]))

            if checkpoint_filename is not None:
                print('Saving policy to file {}'.format(checkpoint_filename))
                torch.save(policy.state_dict(), checkpoint_filename)
            if args.val_checkpoint_filename is not None:
                print('Saving value network to file {}'.format(args.val_checkpoint_filename))
                torch.save(value.state_dict(), args.val_checkpoint_filename)
                print('Done checkpointing model')

        i_episode += 1
        if running_reward > .995:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

    
def train_supervised_(args, robust_fill, optimizer, dataloader, train_logger,
        checkpoint_filename, checkpoint_step_size, checkpoint_print_tensors):

    token_tables = op.build_token_tables()
    device = robust_fill.device
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


#-------------------Debugging code, shows whats up via tensorboard---------------------#
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

#----------------Data Loader---------------------------#
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
        return sample(self.token_tables, self.max_exp, self.max_characters)

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


#---------------------------Training Drivers--------------------------#
def train_supervised(args):
    '''
    Parse arguments and build objects for supervised training approach
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    token_tables = op.build_token_tables()
    from os import path
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'),
            flush_secs=1)

    # init model
    robust_fill = RobustFill(string_size=len(op.CHARACTER), 
        string_embedding_size=args.embedding_size, decoder_inp_size=128,
        hidden_size=args.hidden_size, 
        program_size=len(token_tables.op_token_table),
    )
    if args.continue_training:
        robust_fill.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), args.checkpoint_filename))
        )
    robust_fill = robust_fill.to(device)
    robust_fill.set_device(device)

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

    train_supervised_(args, robust_fill=robust_fill, optimizer=optimizer, 
        dataloader=prog_dataloaer, train_logger=train_logger,
        checkpoint_filename=args.checkpoint_filename,
        checkpoint_step_size=args.checkpoint_step_size, 
        checkpoint_print_tensors=args.print_tensors,
    )


def train_reinforce(args):
    '''
    Parse arguments and construct objects for training reinforce model, with no baseine
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    token_tables = op.build_token_tables()

    # initialize tensorboard for logging output
    from os import path
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'),
            flush_secs=1)

    # Load Models
    policy = RobustFill(string_size=len(op.CHARACTER), 
        string_embedding_size=args.embedding_size, decoder_inp_size=128,
        hidden_size=args.hidden_size, 
        program_size=len(token_tables.op_token_table), device=device
    )
    value = ValueNetwork(128, args.hidden_size).to(device)
    if args.continue_training_policy:
        policy.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), args.checkpoint_filename),
            map_location=device)
        )
    elif args.continue_training:
        policy.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), args.checkpoint_filename),
            map_location=device)
        )
        value.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), args.val_checkpoint_filename),
            map_location=device)
        )
    policy = policy.to(device)
    value = value.to(device)
    # Initialize Optimizer
    if (args.optimizer == 'sgd'):
        pol_opt = optim.SGD(policy.parameters(), lr=args.lr)
        val_opt = optim.SGD(value.parameters(), lr=args.lr)
    else:
        pol_opt = optim.Adam(policy.parameters(), lr=args.lr)
        val_opt = optim.Adam(value.parameters(), lr=args.lr)


    # Load Environment
    env = RobustFillEnv()
    train_reinforce_(args, policy=policy, value=value, pol_opt=pol_opt, 
        value_opt=val_opt, env=env, train_logger=train_logger,
        checkpoint_filename=args.checkpoint_filename,
        checkpoint_step_size=args.checkpoint_step_size, 
        checkpoint_print_tensors=args.print_tensors,
    )


def train_sac(args):
    '''
    Parse arguments and construct objects for training reinforce model, with no baseine
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    token_tables = op.build_token_tables()
    # initialize tensorboard for logging output
    from os import path
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'),
            flush_secs=1)

    # Load Models
    policy = RobustFill(string_size=len(op.CHARACTER), 
        string_embedding_size=args.embedding_size, decoder_inp_size=128,
        hidden_size=args.hidden_size, 
        program_size=len(token_tables.op_token_table),
    )
    q_1 = SoftQNetwork(128, len(token_tables.op_token_table), args.hidden_size)
    q_2 = SoftQNetwork(128, len(token_tables.op_token_table), args.hidden_size)

    tgt_q_1 = SoftQNetwork(128, len(token_tables.op_token_table), args.hidden_size).eval()
    tgt_q_2 = SoftQNetwork(128, len(token_tables.op_token_table), args.hidden_size).eval()


    if args.continue_training_policy:
        policy.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), args.checkpoint_filename),
            map_location=device)
        )
    elif args.continue_training:
        policy.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), args.checkpoint_filename),
            map_location=device)
        )
        q_1.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), args.q1_checkpoint_filename),
            map_location=device)
        )
        q_2.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), args.q2_checkpoint_filename),
            map_location=device)
        )

    for target_param, param in zip(tgt_q_1.parameters(), q_1.parameters()):
        target_param.data.copy_( param.data)
    for target_param, param in zip(tgt_q_2.parameters(), q_2.parameters()):
        target_param.data.copy_(param.data)
    for param in tgt_q_1.parameters():
        param.requires_grad = False
    for param in tgt_q_2.parameters():
        param.requires_grad = False

    policy = policy.to(device)
    q_1 = q_1.to(device)
    q_2 = q_2.to(device)
    tgt_q_1 = tgt_q_1.to(device)
    tgt_q_2 = tgt_q_2.to(device)
    
    # Initialize optimizers
    if (args.optimizer == 'sgd'):
        policy_opt = optim.SGD(policy.parameters(), lr=args.lr)
        q_1_opt = optim.SGD(q_1.parameters(), lr=args.lr)
        q_2_opt = optim.SGD(q_2.parameters(), lr=args.lr)
        entropy_opt = optim.SGD([policy.log_alpha], lr=args.lr)
    else:
        policy_opt = optim.Adam(policy.parameters(), lr=args.lr)
        q_1_opt = optim.Adam(q_1.parameters(), lr=args.lr)
        q_2_opt = optim.Adam(q_2.parameters(), lr=args.lr)
        entropy_opt = optim.Adam([policy.log_alpha], lr=args.lr)


    # Other necessary objects
    env = RobustFillEnv()
    replay_buffer_size = 1_000_000
    replay_buffer = Replay_Buffer(replay_buffer_size, args.batch_size)
    her = HER()
    train_sac_(args, policy, q_1, q_2, tgt_q_1, tgt_q_2, policy_opt, q_1_opt, q_2_opt, 
        entropy_opt, replay_buffer, her, env, train_logger,
        args.checkpoint_filename, args.checkpoint_step_size, args.print_tensors
    )


def run_eval(args):
    token_tables = op.build_token_tables()
    model = RobustFill(string_size=len(op.CHARACTER), 
        string_embedding_size=args.embedding_size, decoder_inp_size=128,
        hidden_size=args.hidden_size, 
        program_size=len(token_tables.op_token_table),
    )
    from os import path
    if args.continue_training:
        model.load_state_dict(torch.load(
            path.join(path.dirname(path.abspath(__file__)), args.checkpoint_filename),
            map_location=torch.device('cpu'))
        )

    eval(model, token_tables)



def eval(model, token_tables, num_samples=200, beam_size=10, em=True):
    model.eval()
    num_match = 0
    print("Evaluatng Examples...")
    for idx in range(num_samples):
        if (idx % 10 == 0):
            print("On example {}".format(idx))
        expected_programs, examples = sample(token_tables)
        max_len = len(expected_programs) + 5 # do not allow generating progams longer than max_len! 
        beam = Beam(beam_size)
        res_beam = Beam(beam_size)
        # Elements in beam are tuples ([sequence], output_all_hidden, hidden)
        # Scores are log probs.  Start with SOS, 
        output_all_hidden, hidden = model.encode_io([examples])
        beam.add(([model.program_size], output_all_hidden, hidden), 0)
        iteration = 0
        while (len(beam) > 0 and iteration < max_len):
            next_beam = Beam(beam_size)  
            for elt, score in beam.get_elts_and_scores():
                sequence, output_all_hidden, hidden = elt
                inp_idx = sequence[-1]
                decoder_input = [model.decoder_embedding(torch.tensor([inp_idx], 
                    device=model.device, dtype=torch.long)) for _ in range(hidden[0].size()[1])
                ]
                probs, output_all_hidden, hidden = model.next_probs(decoder_input, hidden, output_all_hidden)
                probs = F.log_softmax(probs.squeeze(0), dim=-1)
                scored, idx = torch.topk(probs, dim=-1, k=beam_size)
                for next_score, next_idx in zip(scored, idx):
                    if (next_idx == 0): #EOS!
                        next_sequence = copy.deepcopy(sequence)
                        next_sequence.append(next_idx.item())
                        res_beam.add(next_sequence, score + next_score)
                    else:
                        next_sequence = copy.deepcopy(sequence)
                        next_sequence.append(next_idx.item())
                        next_beam.add((next_sequence, output_all_hidden, hidden), score + next_score)
                beam = next_beam
            iteration+=1

        for sequence, _ in res_beam.get_elts_and_scores():
            sequence = sequence[1:]
            if em:
                if (sequence == expected_programs):
                    num_match+=1
                    break
            else:
                if (num_consistent((expected_programs, examples), actual_programs) == 4):
                    num_match+=1
                    break

    print('{}\% Accuracy!'.format((num_match/num_samples) * 100))


# run gives passable argparser interface for colab, no random seed!
def run(args):
    train_supervised(args)

def run_rl(args):
    train_reinforce(args)

def run_sac(args):
    train_sac(args)


def main():
    parser = argparse.ArgumentParser(description='Train RobustFill.')
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-cp', '--continue_training_policy', action='store_true')
    parser.add_argument('--log_dir')
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--hidden_size', default=512)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--embedding_size', default=64)

    parser.add_argument('--checkpoint_filename', default='./checkpoint.pth')
    parser.add_argument('--val_checkpoint_filename', default='./val_checkpoint.pth')
    parser.add_argument('--q1_checkpoint_filename', default='./q1_checkpoint.pth')
    parser.add_argument('--q2_checkpoint_filename', default='./q2_checkpoint.pth')

    parser.add_argument('--checkpoint_step_size', default=8)
    parser.add_argument('--print_tensors', default=True)
    parser.add_argument('--optimizer', default='adam')
    parser.add_argument('--grad_clip', default=.25)
    parser.add_argument('--number_progs', default=1000)
    parser.add_argument('--reinforce', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--sac', action='store_true')
    parser.add_argument('--her', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(1337)
    random.seed(420)
    if (args.reinforce):
        train_reinforce(args)
    elif (args.eval):
        run_eval(args)
    elif (args.sac):
        train_sac(args)
    else:
        train_supervised(args)

if __name__ == '__main__':
    main()