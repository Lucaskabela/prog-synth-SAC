'''
This file defines the MDP, (S, A, R) where:

    - S, states, are pairs <x, y>, where x is the i/o examples 
        fed into the network, and y is the current sequence predicted

    - A, actions, are operators to be appended to the sequence

    - R, the reward function, is 0 at non terminal states, and 1 at terminal
        states if and only if the behavior of the program constructed, 
        y, is consistent with the behavior of the reference program on the i/o
        examples.

    - terminal states are all <x, y> such that y[-1] == 0, the EOS tag.

Note, the interface for the environment follows the gym environment.
'''
import dsl as op
from utils import sample_example

class RobustFillEnv():

    def __init__(self, SOS=10_000, max_expressions=3, max_characters=50, EOS=0, curriculum=True):
        self.max_expressions = max_expressions
        self.max_characters = max_characters
        self.token_tables = op.build_token_tables()
        self.reference_prog = None
        self.examples = None
        self.user_prog = []
        self.SOS = len(self.token_tables.op_token_table)
        self.EOS = EOS
        self.correct = 0     
        self.curriculum=curriculum # Determine if use curriculum learning - every 10_000, increase size

    # Step takes an action from the agent, and returns
    # next_state, reward, done, info
    def step(self, action):
        self.user_prog.append(action)
        if (self.user_prog[-1] == self.EOS):
            self.user_prog = self.user_prog[1:] # Get rid of SOS
            reward = 0
            if (self.reference_prog[0] == self.user_prog):
                reward = 1
                self.correct += 1
            old_prog = self.reference_prog[0]

            i_o = self.reset()
            return (self.user_prog, [self.reference_prog[1]]), reward, True, {'reference_prog' : old_prog} 
        elif(len(self.user_prog) > int(self.correct / 1_000) + 3): # curriculum
            # User aint gonna get it if already on char 25..
            old_prog = self.reference_prog[0]
            i_o = self.reset()
            return (self.user_prog, [self.reference_prog[1]]), 0, True, {'reference_prog' : old_prog} 

        else:
            # If not end of sequnce, return 0
            return (self.user_prog, [self.reference_prog[1]]), 0, False, None
    
    # Reset resets the environment to the initial state, returning the first state
    def reset(self):
        self.user_prog = [self.SOS]
        self.reference_prog = self._sample()
        return (self.user_prog, [self.reference_prog[1]]) # return the i/o strings only

    def _sample(self):
        example = sample_example(self.max_expressions, self.max_characters)
        program = example.program.to_tokens(self.token_tables.op_token_table)
        while len(program) > (self.correct / 10_000) + 3:
            example = sample_example(self.max_expressions, self.max_characters)
            program = example.program.to_tokens(self.token_tables.op_token_table)
        strings = [
            (op.tokenize_string(input_, self.token_tables.string_token_table),
             op.tokenize_string(output, self.token_tables.string_token_table))
            for input_, output in example.strings
        ]
        return (program, strings)


def num_consistent(reference_prog, user_prog, token_tables):

    num_consistent = 0
    expected, examples = reference_prog
    try:
        ref = to_program(expected, token_tables.token_op_table)
    except Exception:
        print("I couldnt parse, sorry mom :(")
        return -1
    try:
        user = to_program(user_prog, token_tables.token_op_table)
    except Exception:
        # include logging here
        # print("I couldn't parse it sorry dad :(")
        return -1 # program was unparsable bro, wtf

    # print("Parser thinks the reference program is: ")
    # print(ref)
    # print("And the user program is: ")
    # print(user)
    # print("\n")
    for i, o in examples:
        i = op.stringify_tokens(i, token_tables.string_token_table)
        o = op.stringify_tokens(o, token_tables.string_token_table)
        ref_res = ref.eval(i)
        assert(ref_res == o)
        try:
            user_res = user.eval(i)
        except Exception:
            continue # If user prog fails on some string, not consistent, move on
        # print("Input was {}".format(i))
        # print("Reference output: ")
        # print(ref_res)
        # print("Policy output: ")
        # print(user_res)
        # print("\n")
        if (ref_res == user_res):
            num_consistent += 1
    return num_consistent

# Should this be in the utils???
def to_program(tokens, token_op_table):
    sub_progs = []
    while(len(tokens) != 0):
        sub_progs.append(op_to_prog(tokens, token_op_table))
        if (sub_progs[-1] == 'EOS'):
            del sub_progs[-1]
            break
        elif (type(sub_progs[-1]) == int or type(sub_progs[-1]) == str):
            # Need operators in top level!
            raise Exception
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
        elif (curr_token == 2):
            arg1 = op_to_prog(tokens, token_op_table)
            arg2 = op_to_prog(tokens, token_op_table)
            arg1 = arg1 if arg1 != 'EOS' else ''
            arg2 = arg2 if arg2 != 'EOS' else ''
            res = curr_op(arg1, arg2)
        elif (curr_token == 3):
            arg1 = token_op_table[tokens.pop(0)]
            if (type(arg1) != str):
                raise Exception
            arg1 = arg1 if arg1 != 'EOS' else ''
            res = curr_op(arg1) # should be a constant string!
        elif (curr_token == 4):
            arg1 = token_op_table[tokens.pop(0)]
            arg2 = token_op_table[tokens.pop(0)]
            arg1 = arg1 if arg1 != 'EOS' else ''
            arg2 = arg2 if arg2 != 'EOS' else ''
            res = curr_op(arg1, arg2)
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
