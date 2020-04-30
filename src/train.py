import argparse
import pprint as pp
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.tensorboard as tb

import dsl as op
from models import RobustFill
from utils import sample_example
from env import to_program


def max_program_length(expected_programs):
    return max([len(program) for program in expected_programs])


def train(args, robust_fill, optimizer, sample, checkpoint_filename, 
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

        optimizer.zero_grad()

        expected_programs, examples = sample()
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
                print_programs(expected_programs[1],
                    actual_programs.permute(2, 0, 1)[:len(expected_programs[0]), :1, :],
                    train_logger, token_tables.token_op_table
                )


            if checkpoint_filename is not None:
                print('Saving to file {}'.format(checkpoint_filename))
                torch.save(robust_fill.state_dict(), checkpoint_filename)
            print('Done checkpointing model')
        global_iter += 1


def print_programs(expected_programs, actual_programs, train_logger, tok_op):
    tokens = torch.argmax(actual_programs.permute(1, 0, 2), dim=-1)
    tokens = tokens[0].tolist()
    if train_logger is not None:
        train_logger.add_text('Expected program', 
            expected_programs, global_iter)
        train_logger.add_text('Actual program', str(tokens), global_iter)
    else:
        print("Expected program: ")
        print(expected_programs)
        print("Actual program: ")
        print(str(tokens))

    try:
        prog = to_program(tokens, tok_op)
    except Exception:
        prog = "Could not parse program"

    if train_logger is not None:
        train_logger.add_text('Parsed program', prog, global_iter)
    else:
        print(prog)


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

    def sample():
        return sample_full(token_tables, batch_size=args.batch_size, max_expressions=3, max_characters=50)

    train(args, robust_fill=robust_fill, optimizer=optimizer, sample=sample, 
        checkpoint_filename=args.checkpoint_filename,
        checkpoint_step_size=args.checkpoint_step_size, 
        checkpoint_print_tensors=args.print_tensors,
    )

# run gives passable argparser interface, no random seed!
def run(args):
    train_full(args)


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
    args = parser.parse_args()

    torch.manual_seed(1337)
    random.seed(420)
    train_full(args)


if __name__ == '__main__':
    main()