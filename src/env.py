'''
This file defines the environemnt for the MDP, (S, A, R) where:

    - S, states, are pairs <x, y>, where x is the current sequence predicted
        and y is the i/o examples

    - A, actions, are the operators to be appended to the sequence (0 - 1132) in 
        this version of the program

    - R, the reward function, is 0 at non terminal states, and 1 at terminal
        states if and only if the behavior of the program constructed, 
        x, is consistent with the behavior of the reference program on the i/o
        examples.

    - terminal states are all <x, y> such that x[-1] == 0, the EOS tag.

Note, the interface for the environment follows the gym environment.
'''
import dsl as op

from utils import sample_example


# TODO: Extend Gym interface
class RobustFillEnv():
    '''
    An rl environment for RobustFill
    '''
    def __init__(self, max_expressions=3, max_characters=50, num_examples=4, EOS=0, curriculum=True):

        # used for sampling programs
        self.max_expressions = max_expressions
        self.max_characters = max_characters
        self.num_examples = num_examples
        self.token_tables = op.build_token_tables()

        # special tokens
        self.SOS = len(self.token_tables.op_token_table)
        self.EOS = EOS

        # Reference for the current trajectory
        self.reference_prog = None
        self.examples = None
        self.user_prog = []

        # Reference for curriculum learning
        self.correct = 0     
        self.curriculum=curriculum # Determine if use curriculum learning 

        # attributes
        self.num_actions = self.SOS
        self.observation_space = None
        self.reward_space = [0, 1]

    def step(self, action):
        '''
        Step takes an action from the agent, and returns (next_state, reward, done, info)
        '''
        self.user_prog.append(action)

        if (self.user_prog[-1] == self.EOS):
            self.user_prog = self.user_prog[1:] # Get rid of SOS
            reward = 0
            if (self.reference_prog[0] == self.user_prog):
                reward = 1
                self.correct += 1
            elif(num_consistent(self.reference_prog, self.user_prog, self.token_tables) == self.num_examples):
                reward = 1
                self.correct += 1
            old_prog = self.reference_prog[0]
            self.reset()
            return (self.user_prog, [self.reference_prog[1]]), reward, True, {'reference_prog' : old_prog} 
        elif(len(self.user_prog) > int(self.correct / 1_000) + 3): # curriculum
            # User aint gonna get it, to far over
            old_prog = self.reference_prog[0]
            self.reset()
            return (self.user_prog, [self.reference_prog[1]]), 0, True, {'reference_prog' : old_prog} 

        else:
            # If not end of sequnce, return 0 reward
            return (self.user_prog, [self.reference_prog[1]]), 0, False, None
    
    def reset(self):
        '''
        Reset the environment by returning new sequence [SOS], and new i/o examples
        '''
        self.user_prog = [self.SOS]
        self.reference_prog = self._sample()
        return (self.user_prog, [self.reference_prog[1]]) 

    def _sample(self):
        '''
        Samples a single program according to parameters of environment
        '''
        example = sample_example(self.max_expressions, self.max_characters, num_strings=self.num_examples)
        program = example.program.to_tokens(self.token_tables.op_token_table)

        if(self.curriculum):
            # Keep looking of too long! ~ curriculum learning
            while len(program) > (self.correct / 10_000) + 3:
                example = sample_example(self.max_expressions, self.max_characters, 
                    num_strings=self.num_examples)
                program = example.program.to_tokens(self.token_tables.op_token_table)

        # Seperate out the i/o examples to run them into the network
        strings = [
            (op.tokenize_string(input_, self.token_tables.string_token_table),
             op.tokenize_string(output, self.token_tables.string_token_table))
            for input_, output in example.strings
        ]
        return (program, strings)


def num_consistent(reference_prog, user_prog, token_tables):
    '''
    Given a reference program, (prog, i/o), user program tokens, and the token tables,
    convert the progams from tokens to operators, then return number of i/o examples they are 
    consistent on.
    '''

    expected, examples = reference_prog
    try:
        ref = to_program(expected, token_tables.token_op_table)
    except Exception:
        print("I couldnt parse, sorry mom :(")
        return -1
    try:
        user = to_program(user_prog, token_tables.token_op_table)
    except Exception:
        return -1 # program was unparsable bro, wtf

    # Evaluate on the programs to find number which match
    num_consistent = 0
    for i, o in examples:
        i = op.stringify_tokens(i, token_tables.string_token_table)
        o = op.stringify_tokens(o, token_tables.string_token_table)
        ref_res = ref.eval(i)
        assert(ref_res == o)
        try:
            user_res = user.eval(i)
        except Exception:
            continue # If user prog fails on some string, not consistent, move on
        if (ref_res == user_res):
            num_consistent += 1
    return num_consistent


def to_program(tokens, token_op_table):
    '''
    Converts a program, which is a list of tokens into operator based programs by recursively applying
    rules defined in dsl.py, using token_op_table
    '''
    sub_progs = []
    while(len(tokens) != 0):
        sub_progs.append(op_to_prog(tokens, token_op_table))
        if (sub_progs[-1] == 'EOS'):
            del sub_progs[-1]
            break
    return op.Concat(*sub_progs)


def op_to_prog(tokens, token_op_table):
    '''
    Recursively converts the first token in tokens, then the rest based on what operator.
    '''
    res = None
    curr_token = tokens.pop(0)
    curr_op = token_op_table[curr_token]

    if curr_token != 0 and (curr_token < 818): # then it is an operator, recurisve!
        if (curr_token == 1): # Concat, so parse rest as a list
            args = []
            while (len(tokens) > 0):
                args.append(op_to_prog(tokens, token_op_table))
                if (args[-1] == 'EOS'):
                    del args[-1]
                    break
            res = curr_op(*args)
        elif (curr_token == 2): # Compose, so parse next 2 as programs
            arg1 = op_to_prog(tokens, token_op_table)
            arg2 = op_to_prog(tokens, token_op_table)
            arg1 = arg1 if arg1 != 'EOS' else ''
            arg2 = arg2 if arg2 != 'EOS' else ''
            res = curr_op(arg1, arg2)
        elif (curr_token == 3): # Constant string, so get next arg (better be a string!!)
            arg1 = token_op_table[tokens.pop(0)]
            arg1 = arg1 if arg1 != 'EOS' else ''
            res = curr_op(arg1)
        elif (curr_token == 4): # Get Substring, so get next two args (better be ints)
            arg1 = token_op_table[tokens.pop(0)]
            arg2 = token_op_table[tokens.pop(0)]
            arg1 = arg1 if arg1 != 'EOS' else ''
            arg2 = arg2 if arg2 != 'EOS' else ''
            res = curr_op(arg1, arg2)
        elif (curr_token == 5): # GetSpan, pop next 6 arguments
            args = [token_op_table[tokens.pop(0)] for _ in range(6)]
            args = [arg if arg != 'EOS' else '' for arg in args]
            res = curr_op(*args)
        else: # After the first 5, we just have enumerated, so simply put them in the output
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
