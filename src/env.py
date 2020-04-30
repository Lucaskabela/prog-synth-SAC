'''
This file defines the MDP for the problem
'''
import dsl as op
from utils import sample_example

# Defines an MDP for robust fill
# This mdp is fairly simple - episodes are a sequence of predictions from the model
# which terminates when the model predicts 0. Reward is given all or none (0 or 1)
# based on consistency with the examples observed
class RobustFillEnv():

    def __init__(self, max_expressions=3, max_characters=50):
        self.max_expressions = max_expressions
        self.max_characters = max_characters
        self.token_tables = op.build_token_tables()
        self.reference_prog = None
        self.examples = None


    # Step takes an action from the agent, and returns
    # next_state, reward, done, info
    def step(self, action):
        # check action == reference_prog

        # example = sample_example(max_expressions=max_expressions, max_characters=max_characters)
        #program = example.program.to_tokens(token_tables.op_token_table)
        #strings = [
        #    (op.tokenize_string(input_, token_tables.string_token_table),
        #     op.tokenize_string(output, token_tables.string_token_table))
        #    for input_, output in example.strings
        #]
        reward = 0
        if (action == reference_prog[0]):
            reward = 1

        done = False
        info = None
        self.reference_prog = sample(self.token_tables, -1, self.max_expressions, self.max_characters)
        return self.reference_prog[1], reward, reference_prog # return the i/o strings only
    
    # Reset resets the environment to the initial state, returning the first state
    def reset(self):
        self.reference_prog = sample(self.token_tables, -1, self.max_expressions, self.max_characters)
        return self.reference_prog[1] # return the i/o strings only

def sample(token_tables, max_expressions, max_characters):

    example = sample_example(max_expressions=max_expressions, max_characters=max_characters)
    program = example.program.to_tokens(token_tables.op_token_table)
    strings = [
        (op.tokenize_string(input_, token_tables.string_token_table),
         op.tokenize_string(output, token_tables.string_token_table))
        for input_, output in example.strings
    ]
    return program, strings


# Should this be in the utils???
def to_program(tokens, token_op_table):

    # [40, 1, 3, 1052, 1, 4, 864, 869, 0]
    # [3, 1113, 1, 562, 0]
    # [5, 819, 936, 829, 1097, 930, 829, 1, 6, 1, 157, 0]
    # [776, 0]
    # [696, 1, 682, 5, 1110, 932, 829, 1108, 929, 829, 1, 4, 1013, 928, 0]
    sub_progs = []
    while(len(tokens) != 0):
        sub_progs.append(op_to_prog(tokens, token_op_table))
        if (sub_progs[-1] == 'EOS'):
            del sub_progs[-1]
            break
    return op.Concat(*sub_progs)

def op_to_prog(tokens, token_op_table):
    res = None
    curr_token = tokens.pop(0)
    curr_op = token_op_table[curr_token]
    if curr_token != 0 and (curr_token < 818): # then it is an operator, recurisve!
        if (curr_token == 1): # Concat
            args = []
            while (len(tokens) > 0):
                args.append(op_to_prog(tokens, token_op_table))
                if (args[-1] == 'EOS'):
                    del args[-1]
                    break

            res = curr_op(*args)
        elif (curr_token == 2 or curr_token == 4):
            arg1 = token_op_table[tokens.pop(0)]
            arg2 = token_op_table[tokens.pop(0)]
            arg1 = arg1 if arg1 != 'EOS' else ''
            arg2 = arg2 if arg2 != 'EOS' else ''
            res = curr_op(arg1, arg2)
        elif (curr_token == 3):
            arg1 = token_op_table[tokens.pop(0)]
            arg1 = arg1 if arg1 != 'EOS' else ''
            res = curr_op(arg1) # should be a constant string!
        elif (curr_token == 5):
            args = [token_op_table[tokens.pop(0)] for _ in range(6)]
            args = [arg if arg != 'EOS' else '' for arg in args]
            res = curr_op(*args)
        else:
            ops = None
            if (type(curr_op) == tuple):
                ops = list(curr_op)
            else:
                ops = list()
                ops.append(curr_op)

            if(len(ops) > 1):
                res = ops[0](*ops[1:])
            else:
                res = ops[0]()
    else:
        return curr_op

    return res
