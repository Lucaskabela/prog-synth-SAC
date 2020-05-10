'''
This file defines the utilities for creating sample programs from the DSL,
Beam, HER, and ReplayBuffer as well
'''

from collections import namedtuple, deque
import logging
import random
import numpy as np

import dsl as op
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

Example = namedtuple(
    'Example',
    ['program', 'strings', 'num_discarded_programs'],
)

def sample(token_tables, max_expressions=3, max_characters=50):
    '''
    Samples a program and i/o examples using token_table and returns 
    in the form (program, [(inputs, outputs)])
    '''
    example = sample_example(max_expressions, max_characters)
    program = example.program.to_tokens(token_tables.op_token_table)
    strings = [
        (op.tokenize_string(input_, token_tables.string_token_table),
         op.tokenize_string(output, token_tables.string_token_table))
        for input_, output in example.strings
    ]
    return (program, strings)


def sample_example(max_expressions=10, max_characters=100, max_empty_strings=0,
        num_strings=4, discard_program_num_empty=100,
        discard_program_num_exceptions=100):

    ''' Does the heavy lifting of sampling'''
    num_discarded = 0
    while True:
        program = sample_program(max_expressions)

        num_empty, num_exception = 0, 0
        sampled_strings = []

        while True:
            string = sample_string(max_characters)
            try:
                transformed = program.eval(string)

                assert isinstance(transformed, str)

                if len(transformed) == 0:
                    num_empty += 1
                    if num_empty <= max_empty_strings:
                        sampled_strings.append((string, transformed))
                else:
                    sampled_strings.append((string, transformed))

            except IndexError:
                num_exception += 1

            if len(sampled_strings) == num_strings:
                return Example(program, sampled_strings, num_discarded)

            # We have to throw programs away because some of them always
            # throw IndexError's or produce empty strings.
            if (num_empty > discard_program_num_empty
                    or num_exception > discard_program_num_exceptions):
                LOGGER.debug('Throwing program away')
                LOGGER.debug('Empty: %s, exception: %s', num_empty, num_exception)
                LOGGER.debug(program)
                num_discarded += 1
                break

def sample_string(max_characters):
    num_characters = random.randint(1, max_characters)
    random_string = ''.join(random.choices(op.CHARACTER, k=num_characters))
    return random_string


def sample_program(max_expressions):
    num_expressions = random.randint(1, max_expressions)
    return op.Concat(*[sample_expression() for _ in range(num_expressions)])


def sample_from(*samplers):
    choice = random.choice(samplers)
    return choice()


def sample_expression():
    return sample_from(sample_substring, sample_nesting, sample_Compose,
        	sample_ConstStr)


def sample_substring():
    return sample_from( sample_SubStr, sample_GetSpan)


def sample_nesting():
    return sample_from(sample_GetToken, sample_ToCase, sample_Replace, sample_Trim,
        	sample_GetUpto, sample_GetFrom, sample_GetFirst, sample_GetAll)


def sample_Compose():
    nesting = sample_nesting()
    nesting_or_substring = sample_from(sample_nesting, sample_substring)
    return op.Compose(nesting, nesting_or_substring)


def sample_ConstStr():
    char = random.choice(op.CHARACTER)
    return op.ConstStr(char)


def sample_position():
    return random.choice(op.POSITION)


def sample_SubStr():
    pos1 = sample_position()
    pos2 = sample_position()
    return op.SubStr(pos1, pos2)


def sample_Boundary():
    return random.choice(list(op.BOUNDARY))


def sample_GetSpan():
    return op.GetSpan(sample_dsl_regex(), sample_index(), sample_Boundary(),
        	sample_dsl_regex(), sample_index(), sample_Boundary())


def sample_Type():
    return random.choice(list(op.Type))


def sample_index():
    return random.choice(op.INDEX)


def sample_GetToken():
    type_ = sample_Type()
    index = sample_index()
    return op.GetToken(type_, index)


def sample_ToCase():
    case = random.choice(list(op.CASE))
    return op.ToCase(case)


def sample_delimiter():
    return random.choice(op.DELIMITER)


def sample_Replace():
    delim1 = sample_delimiter()
    delim2 = sample_delimiter()
    return op.Replace(delim1, delim2)


def sample_Trim():
    return op.Trim()


def sample_dsl_regex():
    return random.choice(list(op.Type) + list(op.DELIMITER))


def sample_GetUpto():
    dsl_regex = sample_dsl_regex()
    return op.GetUpto(dsl_regex)


def sample_GetFrom():
    dsl_regex = sample_dsl_regex()
    return op.GetFrom(dsl_regex)


def sample_GetFirst():
    type_ = sample_Type()
    index = random.choice([i for i in op.INDEX if i > 0])
    return op.GetFirst(type_, index)


def sample_GetAll():
    type_ = sample_Type()
    return op.GetAll(type_)


# Beam from an undergraduate NLP course
class Beam(object):
    """
    Beam data structure. Maintains a list of scored elements, but only keeps the top n
    elements after every insertion operation. Insertion is O(n) (list is maintained in
    sorted order), access is O(1).
    """
    def __init__(self, size):
        self.size = size
        self.elts = []
        self.scores = []

    def __repr__(self):
        return "Beam(" + repr(list(self.get_elts_and_scores())) + ")"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.elts)

    def add(self, elt, score):
        """
        Adds the element to the beam with the given score if the beam has room or if the score
        is better than the score of the worst element currently on the beam

        :param elt: element to add
        :param score: score corresponding to the element
        """
        if len(self.elts) == self.size and score < self.scores[-1]:
            # Do nothing because this element is the worst
            return
        # If the list contains the item with a lower score, remove it
        i = 0
        while i < len(self.elts):
            if self.elts[i] == elt and score > self.scores[i]:
                del self.elts[i]
                del self.scores[i]
            i += 1
        # If the list is empty, just insert the item
        if len(self.elts) == 0:
            self.elts.insert(0, elt)
            self.scores.insert(0, score)
        # Find the insertion point with binary search
        else:
            lb = 0
            ub = len(self.scores) - 1
            # We're searching for the index of the first element with score less than score
            while lb < ub:
                m = (lb + ub) // 2
                # Check > because the list is sorted in descending order
                if self.scores[m] > score:
                    # Put the lower bound ahead of m because all elements before this are greater
                    lb = m + 1
                else:
                    # m could still be the insertion point
                    ub = m
            # lb and ub should be equal and indicate the index of the first element with score less than score.
            # Might be necessary to insert at the end of the list.
            if self.scores[lb] > score:
                self.elts.insert(lb + 1, elt)
                self.scores.insert(lb + 1, score)
            else:
                self.elts.insert(lb, elt)
                self.scores.insert(lb, score)
            # Drop and item from the beam if necessary
            if len(self.scores) > self.size:
                self.elts.pop()
                self.scores.pop()

    def get_elts(self):
        return self.elts

    def get_elts_and_scores(self):
        return zip(self.elts, self.scores)

    def head(self):
        return self.elts[0]


class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""
    
    def __init__(self, buffer_size, batch_size, seed=0):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "i_o", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add_experience(self, states, i_o, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        if type(dones) == list:
            assert type(dones[0]) != list, "A done shouldn't be a list"
            experiences = [self.experience(states, i_o, actions, rewards, next_states, dones)
                           for  states, i_o, actions, rewards, next_states, dones in
                           zip( states, i_o, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience( states, i_o, actions, rewards, next_states, dones)
            self.memory.append(experience)
   
    def sample(self, num_experiences=None, separate_out_data_types=True):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        if separate_out_data_types:
            states, i_o, actions, rewards, next_states, dones = self.separate_out_data_types(experiences)
            return states, i_o, actions, rewards, next_states, dones
        else:
            return experiences
            
    def separate_out_data_types(self, experiences):
        """Puts the sampled experience into the correct format for a PyTorch neural network"""
        states = [e.state for e in experiences if e is not None]
        i_os = [e.i_o[0] for e in experiences if e is not None] # Handle wierd format from environment
        next_states = [e.next_state for e in experiences if e is not None]
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([int(e.done) for e in experiences if e is not None])).float().to(self.device)
        
        return states, i_os, actions, rewards, next_states, dones
    
    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None: batch_size = num_experiences
        else: batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


# Simple HER for this domain - actual goal will be last state
class HER(object):

    def __init__(self):
        self.episode_history = deque()


    def __len__(self):
        return len(self.episode_history)

    def add_experience(self, states, i_o, actions, rewards, next_states, dones):
        self.episode_history.append((states, i_o, actions, rewards, next_states, dones))

    def backward(self):
        '''
        Apply the hindsight - make end of episode state goal state, so change the reward to 1!
        '''
        # Can't really change the state here though...
        self.episode_history[-1][-3] = 1
        episodes_ret = copy.deepcopy(self.episode_history)
        self.reset()
        return episodes_ret

    def reset(self):
        self.episode_history = deque()

# https://stackoverflow.com/questions/56226133/soft-actor-critic-with-discrete-action-space
# ... for discrete action, GumbelSoftmax distribution
class GumbelSoftmax(torch.distributions.RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size()):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        return torch.argmax(noisy_logits, dim=-1)

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value.long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)