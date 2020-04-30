'''
This file defines the models for the project
'''
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class RobustFill(nn.Module):
    def __init__( self, string_size, string_embedding_size, hidden_size, program_size, device='cpu'):
        super().__init__()

        # set the device of this robustfill model and size of chacters in string setting
        self.device = device
        self.string_size = string_size

        # converts character_ngrams to string embeddings, add 1 for padding char
        self.embedding = nn.Embedding(self.string_size + 1, string_embedding_size)
        nn.init.uniform_(self.embedding.weight, -.1, .1)

        # input encoder uses lstm to embed input
        self.input_encoder = AttentionLSTM.lstm(input_size=string_embedding_size, 
            hidden_size=hidden_size)

        # output encoder with attention, default to aligned attention 
        # computed over input_encoder's hidden
        self.output_encoder = AttentionLSTM.aligned_attention(
            input_size=string_embedding_size,
            hidden_size=hidden_size,
        )

        # program decoder, computes attention over output_encoder hidden.
        self.program_decoder = ProgramDecoder(hidden_size=hidden_size, 
            program_size=program_size)

    def set_device(self, device):
        self.device = device
    
    # ensures all programs in batch have same number examples
    @staticmethod
    def _check_num_examples(batch):
        assert len(batch) > 0
        num_examples = len(batch[0])
        assert all([len(examples) == num_examples for examples in batch])
        return num_examples

    # seperates i/o into two different list
    def _split_flatten_examples(self, batch):
        input_batch = [torch.tensor(input_sequence, device=self.device, dtype=torch.long) 
            for examples in batch for input_sequence, _ in examples]
        output_batch = [torch.tensor(output_sequence, device=self.device, dtype=torch.long) 
            for examples in batch for _, output_sequence in examples]
        return input_batch, output_batch

    def _embed_batch(self, batch):
        return self.embedding(batch)

    # Expects:
    # list (batch_size) of tuples (input, output) of list (sequence_length) of token indices
    def forward(self, batch, tgt_progs):
        num_examples = RobustFill._check_num_examples(batch)
        input_batch, output_batch = self._split_flatten_examples(batch)

        # pad the two tensor, mask, then compute the sequnce lengths
        p_input_batch = pad_sequence(input_batch, batch_first=False, 
            padding_value=self.string_size)
        p_output_batch = pad_sequence(output_batch, batch_first=False, 
            padding_value=self.string_size)
        p_mask = (p_input_batch == self.string_size)

        input_length = torch.tensor([i.shape[0] for i in input_batch], 
            dtype=torch.long, device=self.device)
        output_length = torch.tensor([o.shape[0] for o in output_batch], 
            dtype=torch.long, device=self.device)

        # embed batches, 
        input_batch = self._embed_batch(p_input_batch)
        output_batch = self._embed_batch(p_output_batch)

        input_all_hidden, hidden = self.input_encoder(input_batch, input_length, None, None)
        output_all_hidden, hidden = self.output_encoder(output_batch, output_length, hidden=hidden, attended=input_all_hidden, mask=p_mask)

        # return the forward method of the decoder
        return self.program_decoder(hidden=hidden, output_all_hidden=output_all_hidden, out_len=output_length,
            num_examples=num_examples, tgt_progs=tgt_progs)

        # TODO - add an "rl forward" method, which  encodes then single steps until
        # gets a 0.

class ProgramDecoder(nn.Module):
    def __init__(self, hidden_size, program_size):
        super().__init__()
        self.program_size = program_size
        self.program_lstm = AttentionLSTM.single_attention(input_size=program_size, hidden_size=hidden_size)
        self.max_pool_linear = nn.Linear(hidden_size, hidden_size)
        self.softmax_linear = nn.Linear(hidden_size, program_size)
        nn.init.xavier_normal_(self.max_pool_linear.weight, gain=torch.nn.init.calculate_gain('tanh')) 
        nn.init.xavier_normal_(self.softmax_linear.weight) 
        nn.init.constant_(self.max_pool_linear.bias, 0.0)
        nn.init.constant_(self.softmax_linear.bias, 0.0)

    def single_step_forward(self, inp, hidden, output_all_hidden, out_len, num_examples):

        _, hidden = self.program_lstm(inp, out_len, hidden=hidden, attended=output_all_hidden)
        hidden_size = hidden[0].size()[2]
        unpooled = (torch.tanh(self.max_pool_linear(hidden[0][-1, :, :]))
            .view(-1, num_examples, hidden_size)
            .permute(0, 2, 1)
        )
        pooled = F.max_pool1d(unpooled, num_examples).squeeze(2)
        program_embedding = self.softmax_linear(pooled)
        return program_embedding, output_all_hidden, hidden

    def forward(self, hidden, output_all_hidden, out_len, num_examples, tgt_progs, teacher_ratio=.25):

        program_sequence = []

        # initial input is zero vector
        decoder_input = torch.stack([torch.zeros(1, self.program_size) for _ in range(hidden[0].size()[1])])
        decoder_input = decoder_input.to(hidden[0].device)

        ## For teacher learning, first iter, make blank
        correct_onehot = torch.zeros(hidden[0].shape[1], self.program_size)

        # here we should decide to use teacher or not - take trg and make a one-hot encoding
        for i in range(tgt_progs.shape[1]):

            program_embedding, output_all_hidden, hidden = self.single_step_forward(decoder_input, 
                hidden, output_all_hidden, out_len, num_examples)

            program_sequence.append(program_embedding.unsqueeze(0))

            # get input to network for next iteration:
            if random.random() < teacher_ratio: 
                # Get actual, teacher learning time!
                correct_prev = tgt_progs[:, i]
                correct_prev = torch.stack([t for t in correct_prev.split(1) 
                    for _ in range(num_examples)])
                correct_onehot.zero_()
                correct_onehot.scatter(1, correct_prev, 1)
                decoder_input = correct_onehot.unsqueeze(1)
            else:
                decoder_input = torch.stack([F.softmax(p, dim=1) 
                    for p in program_embedding.split(1) for _ in range(num_examples)])

        return torch.cat(program_sequence)


class LuongAttention(nn.Module):
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
    def forward(self, input_, hidden, attended_args, mask=None):
        attended, seq_len = attended_args
        all_hidden, hidden = sorted_rnn(input_, seq_len, hidden, self.lstm)
        return all_hidden, hidden


class SequenceAttention(nn.Module):
    """
    This module returns attention scores over sequences. Details can be
    found in these papers:
        - Aligned question embedding (Chen et al. 2017):
             https://arxiv.org/pdf/1704.00051.pdf
        - Context2Query (Seo et al. 2017):
             https://arxiv.org/pdf/1611.01603.pdf
    Args:
    Inputs:
        k: key tensor (float), [batch_size, p_len, p_dim].
        q: query tensor (float), [batch_size, q_len, q_dim].
        q_mask: Query mask (bool), an elements is `True` if it's pad token
            `False` if it's original. [batch_size, q_len].
    Returns:
        Attention scores over query sequences, [batch_size, p_len, q_len].
    """
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super().__init__()

        # default to using key stuff
        key_size = hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        nn.init.xavier_normal_(self.key_layer.weight)
        nn.init.xavier_normal_(self.query_layer.weight)

        self.relu = nn.ReLU()
        # to store attention scores
        self.alphas = None

    def forward(self, k, q, q_len, q_mask):
        # Compute scores
        p_key = self.relu(self.key_layer(k)) 
        q_key = self.relu(self.query_layer(q))  
        p_key = p_key.permute(1, 0, 2)
        q_key = q_key.permute(1, 2, 0)
        scores = p_key.bmm(q_key)  # [batch_size, k_len, q_len]
        q_mask = q_mask.permute(1, 0)
        q_mask = q_mask.unsqueeze(1).repeat(1, scores.size(1), 1)

        # Assign -inf to pad tokens
        scores.data.masked_fill_(q_mask, -float('inf'))
        # Normalize along question length
        self.alphas = F.softmax(scores, 2)
        return self.alphas.bmm(q.permute(1, 0, 2)).permute(1, 0, 2)


# LSTM Wrapper for the Sequence Attention
class AlignedAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.attention = SequenceAttention(hidden_size, key_size=input_size)
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size)

    def forward(self, input_, hidden, attended_args, mask=None):
        # consistent with other foward method interfaces, attended_args
        # is tuple
        attended, sequence_lengths = attended_args
        context = self.attention(input_, attended, sequence_lengths, mask)
        all_hidden, final_hidden = sorted_rnn(torch.cat((input_, context), 2), 
            sequence_lengths, hidden, self.lstm)

        return all_hidden, final_hidden

# LSTM Wrapper for Luong attention
class SingleAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.attention = LuongAttention.create(hidden_size)
        self.lstm = nn.LSTM(input_size + hidden_size, hidden_size)

    def forward(self, input_, hidden, attended_args, mask=None):
        attended, sequence_lengths = attended_args
        context = self.attention(hidden[0].squeeze(0), attended, sequence_lengths).unsqueeze(1)

        sequence_length_result = torch.ones(input_.shape[0]).to(input_.device) #[batch len ones, bc all sequence 1 here!]
        all_hidden, final_hidden = sorted_rnn(torch.cat((input_, context), 2).permute(1, 0, 2), 
            sequence_length_result, hidden, self.lstm)
        return all_hidden, final_hidden


def _sort_batch_by_length(tensor, sequence_lengths):
    """
    Sorts input sequences by lengths. This is required by Pytorch
    `pack_padded_sequence`. Note: `pack_padded_sequence` has an option to
    sort sequences internally, but we do it by ourselves.

    Args:
        tensor: Input tensor to RNN [batch_size, len, dim].
        sequence_lengths: Lengths of input sequences.

    Returns:
        sorted_tensor: Sorted input tensor ready for RNN [batch_size, len, dim].
        sorted_sequence_lengths: Sorted lengths.
        restoration_indices: Indices to recover the original order.
    """
    # Sort sequence lengths
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    # Sort sequences
    sorted_tensor = tensor.index_select(1, permutation_index)
    # Find indices to recover the original order
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths))).long()
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices


def sorted_rnn(sequences, sequence_lengths, hidden, rnn):
    """
    Sorts and packs inputs, then feeds them into RNN.

    Args:
        sequences: Input sequences, [len, batch_size, dim].
        sequence_lengths: Lengths for each sequence, [batch_size].
        rnn: Registered LSTM or GRU.

    Returns:
        All hidden states, [len, batch_size, hid].
    """
    # Sort input sequences
    sorted_inputs, sorted_sequence_lengths, restoration_indices = _sort_batch_by_length(
        sequences, sequence_lengths
    )
    # Pack input sequences
    packed_sequence_input = pack_padded_sequence(
        sorted_inputs,
        sorted_sequence_lengths.data.long().tolist(),
    )

    # Run RNN
    packed_sequence_output, final_hidden = rnn(packed_sequence_input, hidden)

    # Unpack hidden states
    unpacked_sequence_tensor, _ = pad_packed_sequence(
        packed_sequence_output
    )
    # Restore the original order in the batch and return all hidden states
    return unpacked_sequence_tensor.index_select(1, restoration_indices), final_hidden


# General interface for the attention LSTMs
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
    def aligned_attention(input_size, hidden_size):
        return AttentionLSTM(AlignedAttention(input_size, hidden_size))

    def forward(self, sequence_batch, sequence_len, hidden=None, attended=None, mask=None):

        all_hidden, final_hidden = self.attention_lstm(sequence_batch, hidden,
            (attended, sequence_len), mask)
        return all_hidden, final_hidden


def expand_vector(vector, dim, num_dims):
    if vector.dim() != 1:
        raise ValueError('Expected vector of dim 1. Instead got {}.'.format(
            vector.dim(),
        ))

    return vector.view(*[vector.size()[0] if d == dim else 1 for d in range(num_dims)])


def pad(tensor, sequence_lengths, value, batch_dim, sequence_dim):
    max_length = tensor.size()[sequence_dim]
    indices = expand_vector(torch.arange(max_length, device=sequence_lengths.device), sequence_dim, tensor.dim())
    mask = indices >= expand_vector(sequence_lengths, batch_dim, tensor.dim())
    mask = mask.to(tensor.device)
    tensor.masked_fill_(mask, value)
