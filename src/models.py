'''
This file defines the models for the project
'''
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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
    sorted_tensor = tensor.index_select(0, permutation_index)
    # Find indices to recover the original order
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths))).long()
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices

def sorted_rnn(sequences, sequence_lengths, rnn, h0c0=None):
        """
        Sorts and packs inputs, then feeds them into RNN.

        Args:
            sequences: Input sequences, [batch_size, len, dim].
            sequence_lengths: Lengths for each sequence, [batch_size].
            rnn: Registered LSTM or GRU.
            h0c0: hidden in and cell in

        Returns:
            All hidden states, [batch_size, len, hid].
        """
        # Sort input sequences
        sorted_inputs, sorted_sequence_lengths, restoration_indices = _sort_batch_by_length(
            sequences, sequence_lengths
        )
        # Pack input sequences
        packed_sequence_input = pack_padded_sequence(
            sorted_inputs,
            sorted_sequence_lengths.data.long().tolist(),
            batch_first=True
        )
        # Run RNN
        packed_sequence_output, hiddencell = rnn(packed_sequence_input, h0c0)
        # Unpack hidden states
        unpacked_sequence_tensor, _ = pad_packed_sequence(
            packed_sequence_output, batch_first=True
        )
        # Restore the original order in the batch and return all hidden states
        return unpacked_sequence_tensor.index_select(0, restoration_indices), hiddencell

class RobustFill(nn.Module):
    def __init__( self, string_size, string_embed_dim, hidden_size, program_size, prog_embed_dim,
          bi=True, num_layers=2, dropout=0., device='cpu'):
        super().__init__()

        # set the device of this robustfill model, default to cpu
        self.device = device
        self.bi=bi
        self.num_layers=num_layers
        self.program_size=program_size
        self.string_size=string_size


        self.src_embed = nn.Embedding(string_size + 1, string_embed_dim)
        self.prog_embed = nn.Embedding(program_size + 1, prog_embed_dim)
        nn.init.uniform_(self.src_embed.weight, -.1, .1)
        nn.init.uniform_(self.prog_embed.weight, -.1, .1)

        bi_mult = 1
        if self.bi:
            bi_mult = 2

        # input encoder uses lstm to embed input
        self.input_encoder = Encoder(string_embed_dim, hidden_size, 
            bi=bi, num_layers=num_layers, dropout=dropout)

        self.output_attention = AlignedAttention(hidden_size, key_size=string_embed_dim, bi=True)
        self.output_encoder = Encoder(string_embed_dim + hidden_size*bi_mult, hidden_size=hidden_size, 
            bi=bi, num_layers=num_layers, dropout=dropout)

        self.decoder_attention = BahdanauAttention(hidden_size, bi=True)
        self.program_decoder = Decoder(prog_embed_dim, hidden_size, program_size, 
            bi=bi, num_layers=num_layers, dropout=dropout)


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
        in_list = [torch.tensor(input_sequence, device=self.device, dtype=torch.long) for examples in batch for input_sequence, _ in examples]
        out_list = [torch.tensor(output_sequence, device=self.device, dtype=torch.long) for examples in batch for _, output_sequence in examples]
        return in_list, out_list

    def encoding(self, batch):
        num_examples = RobustFill._check_num_examples(batch)
        in_list, out_list = self._split_flatten_examples(batch)

        # Embed the i/o examples
        in_batch_pad = pad_sequence(in_list, padding_value=self.string_size, batch_first=True)
        in_mask = (in_batch_pad != self.string_size) # [batch_size, seq_len]
        in_lengths = in_mask.long().sum(-1)
        in_bp_embed = self.src_embed(in_batch_pad)

        out_batch_pad = pad_sequence(out_list, padding_value=self.string_size, batch_first=True)
        out_mask = (out_batch_pad != self.string_size) # [batch_size, seq_len]
        out_lengths = out_mask.long().sum(-1)
        out_bp_embed = self.src_embed(out_batch_pad)

        # input_batch_pad [padded seq_len, batch size, embedded dimension]
        input_all_hidden, hidden = self.input_encoder(in_bp_embed, in_lengths)

        if self.bi:
            hidden_query = torch.cat([hidden[0][-1, :, :], hidden[0][-2, :, :]], dim=-1).unsqueeze(1)
        else:
            hidden_query = hidden[0].unsqueeze(1)

        att_in, _ = self.output_attention(out_bp_embed, input_all_hidden, ~in_mask)
        output_all_hidden, hidden = self.output_encoder(torch.cat([out_bp_embed, att_in], dim=2), 
            out_lengths, hidden=hidden)

        return output_all_hidden, hidden, out_mask


       
    # Expects:
    # list (batch_size) of tuples (input, output) of list (sequence_length) of token indices
    def forward(self, batch, trg, teacher_forcing_ratio=.5, num_examples=4):
        encoder_hid, hidden, out_mask = self.encoding(batch)
        # precompute key projection
        proj_key = self.decoder_attention.key_layer(encoder_hid)

        #first input to the decoder is the <sos> tokens
        batch_size_exs = hidden[0].shape[1]
        batch_size_progs = trg.shape[0]
        prog_len = trg.shape[1]
        trg_vocab_size = self.program_size

        #tensor to store decoder outputs
        results = [None] * prog_len
        inp = torch.tensor([self.program_size] * batch_size_exs, dtype=torch.long, device=self.device)
        for t in range(prog_len):

            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            if self.bi:
                hidden_query = torch.cat([hidden[0][-1, :, :], hidden[0][-2, :, :]], dim=-1).unsqueeze(1)
            else:
                hidden_query = hidden[0].unsqueeze(1)

            context, _ = self.decoder_attention(hidden_query, proj_key, value=encoder_hid, mask=~out_mask)
            output, hidden = self.program_decoder(self.prog_embed(inp), context, hidden)
            #place predictions in a tensor holding predictions for each token
            results[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            inp = trg[:, t] if teacher_force else top1
            inp = inp.view(-1, 1).repeat(1, num_examples).view(batch_size_exs)
        results = torch.stack(results)
        return results

        
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bi=False, dropout=0.):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.bi = bi
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bi, 
            batch_first=True, dropout=dropout)
        
        self.init_model_()


    def init_model_(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.uniform_(param, -.1, .1)

    def forward(self, src, src_len, hidden=None):
        # src is b x src_len x input_size
        output, hncn = sorted_rnn(
            src, src_len, self.rnn, hidden
        ) 
        # if bi is true need to manually concatanate
        #outputs = [batch size, src_len, hid dim * num directions]
        return output, hncn


class Decoder(nn.Module):
    def __init__(self, emb_size, hidden_size, out_dim, num_layers=1, dropout=0., bi=False, bridge=True):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi = bi
        self.rnn = nn.LSTM(emb_size + hidden_size*2, hidden_size, 
            bidirectional=bi, num_layers=num_layers, batch_first=True, dropout=dropout)

        bi_mult = 1
        if self.bi:
            bi_mult = 2
        self.bridge = nn.Linear(bi_mult*hidden_size, hidden_size) if bridge else None
        self.max_pool_layer = nn.Linear(bi_mult * hidden_size + bi_mult*hidden_size + emb_size, hidden_size)
        self.softmax_linear = nn.Linear(hidden_size, out_dim)
        self.init_model_()

    def init_model_(self):
        nn.init.xavier_normal_(self.max_pool_layer.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.constant_(self.max_pool_layer.bias, 0.0)

        nn.init.xavier_normal_(self.softmax_linear.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.constant_(self.softmax_linear.bias, 0.0)

        if self.bridge is not None:
            nn.init.xavier_normal_(self.bridge.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.constant_(self.bridge.bias, 0.0)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.uniform_(param, -.1, .1)

    def forward(self, embed, context, hidden, num_examples=4):

        #embedded = [1, batch size, emb dim]
        embed = embed.unsqueeze(1)
        lstm_in = torch.cat([embed, context], dim=-1)
        output, hidden = self.rnn(lstm_in, hidden)
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]

        if self.bi:
            hidden_out = torch.cat([hidden[0][-1, :, :], hidden[0][-2, :, :]], dim=-1)
        else:
            hidden_out = hidden[0]
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [batch size, dec hid dim]
        hidden_out = torch.cat([embed, hidden_out.unsqueeze(1), context], dim=2)
        unpooled = (torch.tanh(self.max_pool_layer(hidden_out))).view(-1, num_examples, hidden[0].shape[-1])
        unpooled = unpooled.permute(0, 2, 1)
        pooled = F.max_pool1d(unpooled, num_examples).squeeze(2)
        program_embedding = self.softmax_linear(pooled)
        return F.softmax(program_embedding, dim=1), hidden

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""
        if encoder_final is None:
            return None  # start with zeros
        return torch.tanh(self.bridge(encoder_final))    


class AlignedAttention(nn.Module):
    """
    This module returns attention scores over sequences. Details can be
    found in these papers:
        - Aligned question embedding (Chen et al. 2017):
             https://arxiv.org/pdf/1704.00051.pdf
        - Context2Query (Seo et al. 2017):
             https://arxiv.org/pdf/1611.01603.pdf

    Args:

    Inputs:
        p: Passage tensor (float), [batch_size, p_len, p_dim].
        q: Question tensor (float), [batch_size, q_len, q_dim].
        q_mask: Question mask (bool), an elements is `False` if it's a word
            `True` if it's a pad token. [batch_size, q_len].

    Returns:
        Attention scores over question sequences, [batch_size, p_len, q_len].
    """
    def __init__(self, hidden_size, key_size=None, query_size=None, bi=False):
        super().__init__()
        bi_mult=1
        if (bi):
            bi_mult=2

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = hidden_size if key_size is None else key_size
        query_size = bi_mult* hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        nn.init.xavier_normal_(self.key_layer.weight)
        nn.init.xavier_normal_(self.query_layer.weight)

        self.relu = nn.ReLU()
        # to store attention scores
        self.alphas = None

    def forward(self, k, q, q_mask):
        # Compute scores
        p_key = self.relu(self.key_layer(k))  # [batch_size, k_len, h_dim]
        q_key = self.relu(self.query_layer(q))  # [batch_size, q_len, h_dim]
        scores = p_key.bmm(q_key.transpose(2, 1))  # [batch_size, k_len, q_len]

        # Stack question mask k_len times
        q_mask = q_mask.unsqueeze(1).repeat(1, scores.size(1), 1)

        # Assign -inf to pad tokens
        scores.data.masked_fill_(q_mask.data, -float('inf'))
        # Normalize along question length
        self.alphas = F.softmax(scores, 2)

        return self.alphas.bmm(q), self.alphas # [batch_size, k_len, h_dim]

class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None, bi=False):
        super(BahdanauAttention, self).__init__()

        bi_mult=1
        if (bi):
            bi_mult=2

        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = bi_mult * hidden_size if key_size is None else key_size
        query_size = bi_mult * hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        nn.init.xavier_normal_(self.key_layer.weight)
        nn.init.xavier_normal_(self.query_layer.weight)
        nn.init.xavier_normal_(self.energy_layer.weight)
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
       
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask.data, -float('inf')).unsqueeze(1)
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1).unsqueeze(1)
        self.alphas = alphas        

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

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


    def forward(self, hidden, output_all_hidden, num_examples, max_program_length):
        program_sequence = []
        decoder_input = [torch.zeros(1, self.program_size) for _ in range(hidden[0].size()[1])]
        for _ in range(max_program_length):
            _, hidden = self.program_lstm(decoder_input, hidden=hidden, attended=output_all_hidden)
            hidden_size = hidden[0].size()[2]
            unpooled = (torch.tanh(self.max_pool_linear(hidden[0][-1, :, :]))
                .view(-1, num_examples, hidden_size)
                .permute(0, 2, 1)
            )
            pooled = F.max_pool1d(unpooled, num_examples).squeeze(2)
            program_embedding = self.softmax_linear(pooled)

            program_sequence.append(program_embedding.unsqueeze(0))
            decoder_input = [F.softmax(p, dim=1) for p in program_embedding.split(1) for _ in range(num_examples)]
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
