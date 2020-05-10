'''
This file defines the models for the project - largely based off 
    https://github.com/yeoedward/Robust-Fill/

Not the most efficent implementation, but upon trying to rewrite, I failed to reach the same
loss, so reverted to this verion with modifications.
'''
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.distributions.categorical import Categorical
from utils import GumbelSoftmax
import numpy as np

class ValueNetwork(nn.Module):
    '''
    Value network takes in a user program, i/o hidden and passes through an lstm
    to get a hidden state, which is tne used to produce a single value out.
    '''
    def __init__(self, state_dim, hidden_dim, num_examples=4, dropout=.05):
        super(ValueNetwork, self).__init__()

        self.seq_encoder = AttentionLSTM.lstm(input_size=state_dim, hidden_size=hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_normal_(self.linear1.weight, gain=nn.init.calculate_gain('relu')) 
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.xavier_normal_(self.linear2.weight, gain=nn.init.calculate_gain('relu')) 
        nn.init.constant_(self.linear2.bias, 0.0)


    def forward(self, embedded_seq, hidden):
        '''
        Takes the embedded user program and hidden state from i/o produced by policy network
        and assigns a scalar score.

        NOTE: embedded_seq is a list of tensors, where the list is batch size large.
        '''
        _, hidden_out = self.seq_encoder(embedded_seq, hidden)
        hidden_state = hidden_out[0] 
        x = F.relu(self.dropout(self.linear1(hidden_state)))
        x = self.linear2(x)
        return x.view(-1)
        
        
class SoftQNetwork(nn.Module):
    '''
    SoftQNetwork takes in a user program, action, and i/o hidden state and determines a 
    scalar value
    '''
    def __init__(self, state_dim, action_dim, hidden_dim, num_examples=4, dropout=.05):
        super(SoftQNetwork, self).__init__()

        self.action_space = action_dim
        self.seq_encoder = AttentionLSTM.lstm(input_size=state_dim, 
            hidden_size=hidden_dim)
        self.linear1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_normal_(self.linear1.weight, gain=nn.init.calculate_gain('relu')) 
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.xavier_normal_(self.linear2.weight, gain=nn.init.calculate_gain('relu')) 
        nn.init.constant_(self.linear2.bias, 0.0)
        
    def forward(self, embedded_seq, action, hidden):
        '''
        Takes the embedded user program, action and hidden state from i/o produced by policy network
        and assigns a scalar score.

        NOTE: embedded_seq is a list of tensors, where the list is batch size large.
        '''
        _, hidden = self.seq_encoder(embedded_seq, hidden)
        hidden_state = torch.cat([hidden[0].squeeze(0), action], -1)
        x = F.relu(self.dropout(self.linear1(hidden_state)))
        x = self.linear2(x)
        return x.view(-1)

class RobustFill(nn.Module):
    '''
    Defines the primary/policy network, which consist of:
        2 embedding layers for embedding i/o strings and decoder sequences
        1 Input encoder (LSTM)
        1 Ouptut encoder (LSTM, with attention on the input encoder)
        1 Program decoder (LSTM, with attention on output encoder)

    Further, for use in REINFORCE and SAC, we have attributes such as temperature,
        reward history, etc.
    '''
    def __init__(self, string_size, string_embedding_size, decoder_inp_size, 
            hidden_size, program_size, device='cpu'):

        super().__init__()

        # set the device of this robustfill model and size of chacters in string setting
        self.device = device
        self.string_size = string_size
        self.program_size = program_size


        # converts character_ngrams to string embeddings, add 1 for padding char
        self.embedding = nn.Embedding(self.string_size + 1, string_embedding_size)
        nn.init.uniform_(self.embedding.weight, -.1, .1)
        self.decoder_embedding = nn.Embedding(program_size+1, decoder_inp_size)
        nn.init.uniform_(self.embedding.weight, -.1, .1)

        # input encoder uses lstm to embed input
        self.input_encoder = AttentionLSTM.lstm(input_size=string_embedding_size, 
            hidden_size=hidden_size)

        # output decoder with attention, default to single attention architecture
        self.output_encoder = AttentionLSTM.single_attention(
            input_size=string_embedding_size,
            hidden_size=hidden_size,
        )

        # program decoder, computes attention over output_encoder hidden.
        self.program_decoder = ProgramDecoder(inp_size=decoder_inp_size ,hidden_size=hidden_size, 
            program_size=program_size)

        # RL attributes
        self.saved_log_probs = []
        self.rewards = []
        self.loss_history = []
        self.reward_history = []
        
        # params for SAC
        target_entropy_ratio = .95
        self.target_entropy = -np.log(1.0/program_size) * target_entropy_ratio
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.detach().exp()

    def set_device(self, device):
        self.device = device
    
    # ensures all programs in batch have same number examples
    @staticmethod
    def _check_num_examples(batch):
        assert len(batch) > 0
        num_examples = len(batch[0])
        assert all([len(examples) == num_examples for examples in batch])
        return num_examples

    def reset(self):
        # Episode policy and reward history cleared
        self.saved_log_probs = []
        self.rewards = []

    def _split_flatten_examples(self, batch):
        '''
        seperates i/o into two different list
        '''
        input_batch = [input_sequence for examples in batch for input_sequence, _ in examples]
        output_batch = [output_sequence for examples in batch for _, output_sequence in examples]
        return input_batch, output_batch

    def _embed_batch(self, batch):
        '''
        embeds a list of input or output examples into a list of tensors
        '''
        return [self.embedding(torch.LongTensor(sequence).to(self.device)) for sequence in batch]

    def encode_io(self, batch):
        '''
        Takes batch, a list of i/o example pairs, and produces the hidden states and (hidden, cell) from
        encoding the i/o examples
        '''
        num_examples = RobustFill._check_num_examples(batch)
        input_batch, output_batch = self._split_flatten_examples(batch)

        input_batch = self._embed_batch(input_batch)
        output_batch = self._embed_batch(output_batch)

        input_all_hidden, hidden = self.input_encoder(input_batch)
        output_all_hidden, hidden = self.output_encoder(output_batch, hidden=hidden, attended=input_all_hidden)
        return output_all_hidden, hidden

    def calc_log_prob_action(self, seq, i_os, reparam=False, num_examples=4):
        '''
        Takes a user program sequence, list of i/o examples and produces an action, 
        calculating the log probability.  Makes use of the GumbelSoftmax (FOR SAC)
        '''
        output_all_hidden, hidden = self.encode_io(i_os)
        seq_lengths = [len(s) for s in seq]
        max_length = max(seq_lengths)

        padded_seqs = torch.LongTensor([
            [program[i] if i < len(program) else 0 for i in range(max_length)]
            for program in seq
        ]).to(self.device)

        program_embeddings = torch.zeros([max_length, len(seq), self.program_size], dtype=torch.float, device=self.device)
        # Run through the decoder, then go back and get one corresponding to lengths!
        for i in range(padded_seqs.shape[1]):
            index_input =  padded_seqs[:, i]
            decoder_input = [self.decoder_embedding(p) for p in index_input.split(1) for _ in range(num_examples)]
            action_probs, output_all_hidden, hidden = self.next_probs(decoder_input, hidden, output_all_hidden)
            program_embeddings[i, :, :] = action_probs
        program_embeddings = program_embeddings.permute(1, 0, 2)
        action_probs = torch.stack([program_embeddings[idx, length - 1, :] for idx, length in enumerate(seq_lengths)])
        probs = F.softmax(action_probs.squeeze(0), dim=-1)
        action_pd = GumbelSoftmax(probs=probs, temperature=1)
        actions = action_pd.rsample() if reparam else action_pd.sample()
        log_probs = action_pd.log_prob(actions)
        return log_probs, actions

    def act(self, inp, hidden, output_all_hidden):
        '''
        Returns a token (action) given the previous token, hidden state, and all hidden
        states from the output decoder (for attention)
        '''
        action_probs, _, _ = self.next_probs(inp, hidden, 
            output_all_hidden)    
        probs = F.softmax(action_probs.squeeze(0), dim=-1)
        m = Categorical(probs=probs)
        action = m.sample()
        return action.cpu().squeeze()

    def select_action(self, inp, hidden, output_all_hidden):
        '''
        Pretty similar to last method, but for use with REINFORCE
        '''
        action_probs, out_all, hidden = self.next_probs(inp, hidden, 
            output_all_hidden)    
        probs = F.softmax(action_probs.squeeze(0), dim=-1)
        m = Categorical(probs=probs)
        action = m.sample()
        return action.cpu().squeeze(), m.log_prob(action), out_all, hidden

    def next_probs(self, inp, hidden, output_all_hidden, num_examples=4):
        '''
        Given the previous token, hidden and decoder hidden, produce the probabilities over
        tokens
        '''
        program_embedding, output_all_hidden, hidden = self.program_decoder.forward(inp, hidden, 
            output_all_hidden, num_examples)
        return program_embedding, output_all_hidden, hidden 

    def predict(self, batch, num_examples=4):
        ''' 
        Given a batch of i/o examples, predicts the entire output (for use with eval)
        '''
        program = []
        output_all_hidden, hidden = self.encode_io(batch)
        # initial input is padding vector
        decoder_input = [self.decoder_embedding(torch.tensor([self.program_size], 
            device=self.device, dtype=torch.long)) for _ in range(hidden[0].size()[1])
        ]

        while program[-1] != 0 and len(program) < 25:
            program_embedding, output_all_hidden, hidden = self.program_decoder.forward(decoder_input, 
                hidden, output_all_hidden, num_examples)

            program.append(torch.argmax(program_embedding, dim=-1).item())
            idx_next = [program[-1]]

            index_input = torch.tensor(idx_next, device=self.device, dtype=torch.long)
            decoder_input = [self.decoder_embedding(p) for p in index_input.split(1) for _ in range(num_examples)]
        
        return program

    def forward(self, batch, tgt_progs, num_examples=4, teacher_ratio=.5):
        '''
        Given batch of i/o examples and the padded target programs for teacher learning, 
        outputs programs.
        '''
        program_sequence = []
        output_all_hidden, hidden = self.encode_io(batch)
        # initial input is padding vector
        decoder_input = [self.decoder_embedding(torch.tensor([self.program_size], 
            device=self.device, dtype=torch.long)) for _ in range(hidden[0].size()[1])
        ]
        # here we should decide to use teacher or not - take trg and make a one-hot encoding
        for i in range(tgt_progs.shape[1]):

            program_embedding, output_all_hidden, hidden = self.program_decoder.forward(decoder_input, 
                hidden, output_all_hidden, num_examples)

            program_sequence.append(program_embedding.unsqueeze(0))

            # get input to network for next iteration:
            if random.random() < teacher_ratio: 
                # Get actual, teacher learning time!
                idx_next = tgt_progs[:, i].tolist()
            else:
                idx_next = torch.argmax(program_embedding, dim=-1).tolist()

            index_input = torch.tensor(idx_next, device=self.device, dtype=torch.long)
            decoder_input = [self.decoder_embedding(p) for p in index_input.split(1) for _ in range(num_examples)]
        

        return torch.cat(program_sequence)

class ProgramDecoder(nn.Module):
    '''
    Inner network for the program decoder.  Contains standard lstm decoder with attention over
    output encoder, as well as max pool over the number of i/o examples.
    '''
    def __init__(self, inp_size, hidden_size, program_size):
        super().__init__()
        self.program_size = program_size
        self.program_lstm = AttentionLSTM.single_attention(input_size=inp_size, hidden_size=hidden_size)
        self.max_pool_linear = nn.Linear(hidden_size, hidden_size)
        self.softmax_linear = nn.Linear(hidden_size, program_size)
        nn.init.xavier_normal_(self.max_pool_linear.weight, gain=torch.nn.init.calculate_gain('tanh')) 
        nn.init.xavier_normal_(self.softmax_linear.weight) 
        nn.init.constant_(self.max_pool_linear.bias, 0.0)
        nn.init.constant_(self.softmax_linear.bias, 0.0)

    def forward(self, decoder_input, hidden, output_all_hidden, num_examples):
        _, hidden = self.program_lstm(decoder_input, hidden=hidden, attended=output_all_hidden)
        hidden_size = hidden[0].size()[2]
        unpooled = (torch.tanh(self.max_pool_linear(hidden[0][-1, :, :]))
            .view(-1, num_examples, hidden_size)
            .permute(0, 2, 1)
        )
        pooled = F.max_pool1d(unpooled, num_examples).squeeze(2)
        program_embedding = self.softmax_linear(pooled)
        return program_embedding, output_all_hidden, hidden


class LuongAttention(nn.Module):
    '''
    Attention for the network (Luong form)
    '''
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        nn.init.xavier_normal_(self.linear.weight) 
        nn.init.constant_(self.linear.bias, 0.0)

    def create(query_size):
        return LuongAttention(nn.Linear(query_size, query_size))

    @staticmethod
    def _masked_softmax(vectors, sequence_lengths):
        pad(vectors, sequence_lengths, float('-inf'), batch_dim=0, sequence_dim=1)
        return F.softmax(vectors, dim=1)

    # attended: (other sequence length x batch size x query size)
    # Uses the "general" content-based function
    def forward(self, query, attended, sequence_lengths):
        if query.dim() != 2:
            raise ValueError(
                'Expected query to have 2 dimensions. Instead got {}'.format(
                    query.dim(),
                )
            )

        # (batch size x query size)
        key = self.linear(query)
        # (batch size x other sequence length)
        align = LuongAttention._masked_softmax(
            torch.matmul(attended.unsqueeze(2), key.unsqueeze(2))
            .squeeze(3)
            .squeeze(2)
            .transpose(1, 0),
            sequence_lengths,
        )
        # (batch_size x query size)
        context = (align.unsqueeze(1).bmm(attended.transpose(1, 0)).squeeze(1))
        return context


class LSTMAdapter(nn.Module):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    @staticmethod
    def create(input_size, hidden_size):
        return LSTMAdapter(nn.LSTM(input_size, hidden_size))

    # attended_args is here to conform to the same interfaces
    # as the attention-variants
    def forward(self, input_, hidden, attended_args):
        if attended_args is not None:
            raise ValueError('LSTM doesnt use the arg "attended"')

        _, hidden = self.lstm(input_.unsqueeze(0), hidden)
        return hidden


class SingleAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.attention = LuongAttention.create(hidden_size)
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size)

    def forward(self, input_, hidden, attended_args):
        attended, sequence_lengths = attended_args
        context = self.attention(hidden[0].squeeze(0), attended, sequence_lengths)
        input_ = input_.to(context.device)
        _, hidden = self.lstm(torch.cat((input_, context), 1).unsqueeze(0), hidden)
        return hidden


class AttentionLSTM(nn.Module):
    def __init__(self, attention_lstm):
        super().__init__()
        self.attention_lstm = attention_lstm

    @staticmethod
    def lstm(input_size, hidden_size):
        return AttentionLSTM(LSTMAdapter.create(input_size, hidden_size))

    @staticmethod
    def single_attention(input_size, hidden_size):
        return AttentionLSTM(SingleAttention(input_size, hidden_size))

    @staticmethod
    def _pack(sequence_batch):
        sorted_indices = sorted(
            range(len(sequence_batch)),
            key=lambda i: sequence_batch[i].shape[0],
            reverse=True,
        )
        packed = pack_sequence([sequence_batch[i] for i in sorted_indices])
        return packed, sorted_indices

    @staticmethod
    def _sort(hidden, attended, sorted_indices):
        if hidden is None:
            return None, None

        sorted_hidden = (hidden[0][:, sorted_indices, :], hidden[1][:, sorted_indices, :])

        sorted_attended = None
        if attended is not None:
            sorted_attended = (attended[0][:, sorted_indices, :], attended[1][sorted_indices])

        return sorted_hidden, sorted_attended

    @staticmethod
    def _unsort(all_hidden, final_hidden, sorted_indices):
        unsorted_indices = [None] * len(sorted_indices)
        for i, original_idx in enumerate(sorted_indices):
            unsorted_indices[original_idx] = i

        unsorted_all_hidden = all_hidden[:, unsorted_indices, :]
        unsorted_final_hidden = (
            final_hidden[0][:, unsorted_indices, :],
            final_hidden[1][:, unsorted_indices, :],
        )

        return unsorted_all_hidden, unsorted_final_hidden

    def _unroll(self, packed, hidden, attended):
        all_hn = []
        final_hn = []
        final_cn = []

        pos = 0
        for size in packed.batch_sizes:
            timestep_data = packed.data[pos:pos+size, :]
            pos += size

            if hidden is not None and hidden[0].size()[1] > size:
                hn, cn = hidden
                hidden = hn[:, :size, :], cn[:, :size, :]
                final_hn.append(hn[:, size:, :])
                final_cn.append(cn[:, size:, :])

                if attended is not None:
                    attended = (
                        attended[0][:, :size, :],
                        attended[1][:size],
                    )

            hidden = self.attention_lstm(
                input_=timestep_data,
                hidden=hidden,
                attended_args=attended,
            )

            all_hn.append(hidden[0].squeeze(0))

        final_hn.append(hidden[0])
        final_cn.append(hidden[1])

        final_hidden = (
            torch.cat(final_hn[::-1], 1),
            torch.cat(final_cn[::-1], 1),
        )
        # all_hn is a list (sequence_length) of
        # tensors (batch_size for timestep x hidden_size).
        # So if we set batch_first=True, we get back tensor
        # (sequence_length x batch_size x hidden_size)
        all_hidden = pad_sequence(all_hn, batch_first=True)

        return all_hidden, final_hidden

    def forward(self, sequence_batch, hidden=None, attended=None):
        if not isinstance(sequence_batch, list):
            raise ValueError(
                'sequence_batch has to be a list. Instead got {}.'.format(
                    type(sequence_batch).__name__,
                )
            )

        packed, sorted_indices = AttentionLSTM._pack(sequence_batch)
        sorted_hidden, sorted_attended = AttentionLSTM._sort(hidden, attended, sorted_indices)
        all_hidden, final_hidden = self._unroll(packed, sorted_hidden, sorted_attended)
        unsorted_all_hidden, unsorted_final_hidden = AttentionLSTM._unsort(all_hidden=all_hidden,
            final_hidden=final_hidden, sorted_indices=sorted_indices,
        )
        sequence_lengths = torch.LongTensor([s.shape[0] for s in sequence_batch])
        return (unsorted_all_hidden, sequence_lengths), unsorted_final_hidden


def expand_vector(vector, dim, num_dims):
    if vector.dim() != 1:
        raise ValueError('Expected vector of dim 1. Instead got {}.'.format(
            vector.dim(),
        ))

    return vector.view(*[vector.size()[0] if d == dim else 1 for d in range(num_dims)])


def pad(tensor, sequence_lengths, value, batch_dim, sequence_dim):
    max_length = tensor.size()[sequence_dim]
    indices = expand_vector(torch.arange(max_length), sequence_dim, tensor.dim())
    mask = indices >= expand_vector(sequence_lengths, batch_dim, tensor.dim())
    mask = mask.to(tensor.device)
    tensor.masked_fill_(mask, value)