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

    def __init__(self, max_expressions=3, max_characters=50, EOS=0, curriculum=True):
        self.max_expressions = max_expressions
        self.max_characters = max_characters
        self.token_tables = op.build_token_tables()
        self.reference_prog = None
        self.examples = None
        self.user_prog = []
        self.EOS = EOS
        self.correct = 0     
        self.curriculum=curriculum # Determine if use curriculum learning - every 10_000, increase size

    # Step takes an action from the agent, and returns
    # next_state, reward, done, info
    def step(self, action):

        self.user_prog.append(action)
        if (self.user_prog[-1] == self.EOS):
            reward = 0
            # all or nothing rn, can modify later.
            if (num_consistent(self.reference_prog, self.user_prog) == len(self.reference_prog[1])):
                reward = 1
                correct += 1
            old_prog = self.reference_prog[0]

            i_o = self.reset()
            return i_o, reward, True, {'reference_prog' : old_prog} 
        elif(len(self.user_prog) > self.correct % 10_000 + 2):
            # User aint gonna get it if already on char 25..
            old_prog = self.reference_prog[0]

            i_o = self.reset()
            return i_o, 0, True, {'reference_prog' : old_prog} 

        else:
            # If not end of sequnce, return 0
            return [self.reference_prog[1]], 0, False, None
    
    # Reset resets the environment to the initial state, returning the first state
    def reset(self):
        self.user_prog = []
        self.reference_prog = self._sample()
        return [self.reference_prog[1]] # return the i/o strings only

    def _sample(self):

        example = sample_example(self.max_expressions, self.max_characters)
        program = example.program.to_tokens(self.token_tables.op_token_table)
        while len(program) > (self.correct % 10_000) + 2:
            example = sample_example(self.max_expressions, self.max_characters)
            program = example.program.to_tokens(self.token_tables.op_token_table)
        strings = [
            (op.tokenize_string(input_, self.token_tables.string_token_table),
             op.tokenize_string(output, self.token_tables.string_token_table))
            for input_, output in example.strings
        ]
        return (program, strings)


def num_consistent(reference_prog, user_prog):
    num_consistent = 0
    examples, expected = reference_prog
    try:
        ref = to_program(expected)
    except Exception:
        assert(False, "Reference program should not error")

    try:
        user = to_program(user_prog)
    except Exception:
        # include logging here
        return -1 # program was unparsable bro, wtf

    for i, o in examples:
        ref_res = ref.eval(i)
        assert(ref_res == o)
        try:
            user_res = user.eval(o)
        except Exception:
            continue # If user prog fails on some string, not consistent, move on\

        if (ref_res == user_res):
            num_consistent += 1
    return num_consistent

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
