import argparse
import pprint as pp
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import RobustFill
from utils import sample_example
import dsl as op

import torch.utils.tensorboard as tb

def max_program_length(expected_programs):
    return max([len(program) for program in expected_programs])


def train(
        args, 
        robust_fill,
        optimizer,
        sample,
        checkpoint_filename,
        checkpoint_step_size,
        checkpoint_print_tensors):

    from os import path
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Beginning training.")
    print(train_logger)
    if args.continue_training:
        robust_fill.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), checkpoint_filename)))
    robust_fill = robust_fill.to(device)
    robust_fill.set_device(device)
    robust_fill.train()
    example_idx = 0
    while True:
        optimizer.zero_grad()

        expected_programs, examples = sample()
        max_length = max_program_length(expected_programs)
        actual_programs = robust_fill(examples, max_length)
        program_size = actual_programs.size()[2]
        padding_index = -1

        # Output: program_size x b x #ops
        # need to turn b x #ops x #p_size
        reshaped_actual_programs = (
            actual_programs.permute(1, 2, 0)
        )
        # B x program_size
        padded_expected_programs = torch.LongTensor([
                [program[i] if i < len(program) else padding_index 
                    for i in range(max_length)]
                for program in expected_programs
        ]).to(device)
        loss = F.cross_entropy(
            reshaped_actual_programs,
            padded_expected_programs,
            ignore_index=padding_index,
        )

        loss.backward()
        optimizer.step()
        if train_logger is not None:
            train_logger.add_scalar('loss', loss.detach(), example_idx)

        if example_idx % checkpoint_step_size == 0:
            print('Checkpointing at example {}'.format(example_idx))
            print('Loss: {}'.format(loss))

            if checkpoint_print_tensors:
                print_batch_limit = 3

                # print('Examples:')
                # pp.pprint(examples[:print_batch_limit])

                print('Expected programs:')
                print(expected_programs[0])

                print('Actual programs:')
                print(
                    torch.argmax(actual_programs.permute(1, 0, 2)[0, 0:len(expected_programs[0])], dim=-1)
                )

            if checkpoint_filename is not None:
                print('Saving to file {}'.format(checkpoint_filename))
                torch.save(robust_fill.state_dict(), checkpoint_filename)

            print('Done')

        example_idx += 1


def generate_program(batch_size):
    return [
        [0] if random.randint(0, 1) == 0 else [1, 0]
        for _ in range(batch_size)
    ]


def generate_data(program_batch, num_examples, string_size):
    # Batch is a:
    # list (batch_size) of tuples (input, output) of list (sequence_length)
    # of token indices
    batch = []
    for program in program_batch:
        examples = []
        for _ in range(num_examples):
            input_sequence = [random.randint(0, string_size-1)]

            if program == [0]:
                output_sequence = input_sequence
            elif program == [1, 0]:
                output_sequence = input_sequence * 2
            else:
                raise ValueError('Invalid program {}'.format(program))

            examples.append((input_sequence, output_sequence))

        batch.append(examples)

    return batch


def sample_easy(batch_size, string_size, num_examples):
    programs = generate_program(batch_size)
    examples = generate_data(programs, num_examples, string_size)
    return programs, examples


def train_easy(args):
    string_size = 3
    robust_fill = RobustFill(
        string_size=string_size,
        string_embedding_size=2,
        hidden_size=8,
        program_size=2,
    )
    optimizer = optim.SGD(robust_fill.parameters(), lr=0.01)

    def sample():
        return sample_easy(
            batch_size=32,
            string_size=string_size,
            num_examples=2,
        )

    train(
        args,
        robust_fill=robust_fill,
        optimizer=optimizer,
        sample=sample,
        checkpoint_filename=None,
        checkpoint_step_size=100,
        checkpoint_print_tensors=True,
    )


def sample_full(token_tables, batch_size, max_expressions, max_characters):
    program_batch, strings_batch = [], []

    for _ in range(batch_size):
        example = sample_example(
            max_expressions=max_expressions,
            max_characters=max_characters,
        )
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

    checkpoint_filename = './checkpoint.pth'
    robust_fill = RobustFill(
        string_size=len(op.CHARACTER),
        string_embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        program_size=len(token_tables.op_token_table),
    )
    optimizer = optim.SGD(robust_fill.parameters(), lr=args.lr)

    def sample():
        return sample_full(
            token_tables,
            batch_size=args.batch_size,
            max_expressions=3,
            max_characters=50,
        )

    train(
        args, 
        robust_fill=robust_fill,
        optimizer=optimizer,
        sample=sample,
        checkpoint_filename=checkpoint_filename,
        checkpoint_step_size=128,
        checkpoint_print_tensors=True,
    )

def run(args):
    torch.manual_seed(1337)
    random.seed(420)
    train_full(args)

def main():
    parser = argparse.ArgumentParser(description='Train RobustFill.')
    parser.add_argument(
        '--dry',
        action='store_true',
        help='run smaller network on easier version of the problem',
    )
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('--log_dir')
    parser.add_argument('--lr', default=1e-3)
    parser.add_argument('--hidden_size', default=512)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--embedding_size', default=128)


    args = parser.parse_args()

    torch.manual_seed(1337)
    random.seed(420)

    if args.dry:
        train_easy(args)
    else:
        train_full(args)


if __name__ == '__main__':
    main()