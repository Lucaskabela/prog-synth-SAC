'''
This file defines the models for the project
'''
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        print(sorted_inputs)
        packed_sequence_input = pack_padded_sequence(
            sorted_inputs,
            sorted_sequence_lengths.data.long().tolist(),
            batch_first=True
        )
        # Run RNN
        packed_sequence_output, hiddencell = rnn(packed_sequence_input, h0c0)
        print("Sequence out length is: " )
        print(packed_sequence_output)
        # Unpack hidden states
        unpacked_sequence_tensor, _ = pad_packed_sequence(
            packed_sequence_output, batch_first=True
        )
        # Restore the original order in the batch and return all hidden states
        return unpacked_sequence_tensor.index_select(0, restoration_indices), hiddencell


class RobustFill(nn.Module):
    def __init__( self, string_size, embed_dim, hidden_size, program_size, 
          bi=True, num_layers=2, device='cpu'):
        super().__init__()

        # set the device of this robustfill model, default to cpu
        self.device = device
        self.bi=bi
        self.num_layers=num_layers
        self.program_size=program_size
        self.embedding = nn.Embedding(string_size, embed_dim)
        nn.init.xavier_normal_(self.embedding.weight)

        # input encoder uses lstm to embed input
        self.input_encoder = Encoder(embed_dim=embed_dim, hidden_size=hidden_size, 
            dec_hid_dim=hidden_size*2, bi=bi, num_layers=num_layers)

        # output decoder with attention, default to single attention architecture
        self.output_encoder = Encoder(embed_dim=embed_dim, hidden_size=hidden_size, 
            dec_hid_dim=hidden_size*2, attention=True, bi=bi, num_layers=num_layers)

        self.program_decoder = Decoder(embed_dim=embed_dim, enc_hidden=hidden_size, dec_hid_dim=hidden_size*4, 
            out_dim=program_size, attention=True, bi=True)


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
        return len(in_list), in_list + out_list

    # Expects:
    # list (batch_size) of tuples (input, output) of list (sequence_length) of token indices
    def forward(self, batch, trg, max_program_length):
        print(trg.shape)
        num_examples = RobustFill._check_num_examples(batch)
        split_idx, exs = self._split_flatten_examples(batch)

        # Embed the i/o examples
        padded_batch = pad_sequence(exs, padding_value=1, batch_first=True)
        in_batch_pad = padded_batch[:split_idx]
        in_mask = (in_batch_pad != 1) # [batch_size, seq_len]
        in_lengths = in_mask.long().sum(-1)
        in_bp_embed = self.embedding(in_batch_pad)

        out_batch_pad = padded_batch[split_idx:]
        out_mask = (out_batch_pad != 1) # [batch_size, seq_len]
        out_lengths = out_mask.long().sum(-1)
        out_bp_embed = self.embedding(out_batch_pad)

        # input_batch_pad [padded seq_len, batch size, embedded dimension]
        input_all_hidden, hidden = self.input_encoder(in_bp_embed, in_lengths)
        output_all_hidden, hidden = self.output_encoder(out_bp_embed, out_lengths, hidden=hidden, pay_attn_to=input_all_hidden)

        #first input to the decoder is the <sos> tokens
        inp = trg[:, 0]
        batch_size = padded_batch.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.program_size
        
        #tensor to store decoder outputs
        results = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        for t in range(1, len(trg)):
            print(inp)
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.program_decoder(inp, out_mask, hidden, output_all_hidden)
            
            #place predictions in a tensor holding predictions for each token
            results[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            inp = trg[t] if teacher_force else top1

        return results

class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, dec_hid_dim=512, attention=False, bi=False, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, 
            bidirectional=bi, num_layers=num_layers, batch_first=True)
        self.nonlinearity = torch.nn.Tanh()

        # Initialize nonlinearity
        self.nonlinear='tanh'
        self.num_layers=num_layers
        self.bi=bi
        self.hidden_size = hidden_size
        if bi is True:
            self.hidden_size*=2
        self.attention=attention
        self.fc = nn.Linear(self.hidden_size, dec_hid_dim)

        if self.attention is True:
            self.attn = Attention(self.hidden_size)
            self.lstm = nn.LSTM(input_size=embed_dim + self.hidden_size, hidden_size=hidden_size, 
                bidirectional=bi, num_layers=num_layers, batch_first=True)        
        else:
            self.attn = None

        self.init_model_()


    def init_model_(self):
        nn.init.xavier_normal_(self.fc.weight, gain=nn.init.calculate_gain(self.nonlinear))
        nn.init.constant_(self.fc.bias, 0.0)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, src, src_mask, hidden=None, pay_attn_to=None):
        # src is embedded input padded
        print("In the encoder")
        if self.attention is True:
            assert(pay_attn_to is not None and hidden is not None)
            #pay_attn_to = [batch size, src len, enc hid dim * # directions]
            #a = [batch size, 1, src len]
            if self.bi:
                hidden_attn = torch.cat([hidden[0][-1, :, :], hidden[0][-2, :, :]], -1)
            else:
                hidden_attn = hidden[0][-1, :, :]
            a = self.attn(hidden_attn, pay_attn_to, src_mask)
            a = a.repeat(1, src.shape[1], 1)
            print("Shapes: ")
            print(src.shape)
            print(a.shape)
            #weighted = [batch size, 1, enc hid dim * #directions]
            lstm_in = torch.cat((src, a), dim=2)
            print(lstm_in.shape)
            print("Size going in: ")
            print(lstm_in.shape)
            print(hidden[0].shape)
            output, hncn = sorted_rnn(
                lstm_in, src_mask, self.lstm, hidden
            )
            print("Size coming out: ")
            print(output.shape)
            print(hncn[0].shape)

        else:
            print("Going in: ")
            print(src.shape)
            output, hncn = sorted_rnn(
                src, src_mask, self.lstm
            ) 

        # hidden_out = hncn[0]
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...], so outputs are always from the last layer
        # if (self.bi):
        #     print(torch.cat((hidden_out[-2,:,:], hidden_out[-1,:,:]), dim = 1).shape)
        #     hidden_out = self.nonlinearity(self.fc(torch.cat((hidden_out[-2,:,:], hidden_out[-1,:,:]), dim = 1)))
        # else:
        #     hidden_out = self.nonlinearity(self.fc(hidden_out[-1,:,:]))

        print("Output: ")
        print(output.shape)
        print(hncn[0].shape)
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [batch size, dec hid dim]
        return output, hncn


class Attention(nn.Module):
    def __init__(self, query):
        super().__init__()
        
        self.attn = nn.Linear(query, query)
        nn.init.xavier_normal_(self.attn.weight)
        nn.init.constant_(self.attn.bias, 0.0)

    def forward(self, query, attend_to, mask):
        
        print("Attention shapes: ")
        key = self.attn(query)
        print(key.shape)
        attended = attend_to.permute(1, 0, 2)
        print(attend_to.shape)

        prod = torch.matmul(attended.unsqueeze(2), key.unsqueeze(2))
        align = prod.squeeze(3).squeeze(2)
        align.masked_fill_(mask.bool(), -float('inf'))
        align = align.transpose(1, 0)
        print(align.shape)
        align = F.softmax(align, dim=1)
        context = (align.unsqueeze(1).bmm(attended.transpose(1, 0)))
        return context


class Decoder(nn.Module):
    def __init__(self, embed_dim, enc_hidden, dec_hid_dim, out_dim, 
            nonlinear='tanh', attention=False, bi=False, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(out_dim, embed_dim)
        self.lstm = nn.LSTM(input_size=enc_hidden + embed_dim, hidden_size=dec_hid_dim, 
            bidirectional=bi, num_layers=num_layers)

        # Initialize nonlinearity
        self.nonlinear=nonlinear
        if nonlinear == 'relu':
            self.nonlinearity = torch.nn.ReLU()
        elif nonlinear == 'tanh':
            self.nonlinearity = torch.nn.Tanh()
        else:
            self.nonlinearity = None

        self.bi=bi
        self.hidden_size = enc_hidden
        if bi is True:
            self.hidden_size*=2
        self.attention=attention

        if attention is True:
            self.attn = Attention(self.hidden_size)

        self.max_pool_linear = nn.Linear(enc_hidden + dec_hid_dim + embed_dim, enc_hidden + dec_hid_dim + embed_dim)
        self.softmax_linear = nn.Linear(enc_hidden + dec_hid_dim + embed_dim, out_dim)
        self.init_model_()

    def init_model_(self):
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.xavier_normal_(self.max_pool_linear.weight, gain=nn.init.calculate_gain(self.nonlinear))
        nn.init.constant_(self.max_pool_linear.bias, 0.0)

        nn.init.xavier_normal_(self.softmax_linear.weight, gain=nn.init.calculate_gain(self.nonlinear))
        nn.init.constant_(self.softmax_linear.bias, 0.0)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, src, src_mask, hidden, encoder_out, num_examples=4):

        #embedded = [1, batch size, emb dim]
        src.unsqueeze(0)
        embedded = self.embedding(src)
                
        #a = [batch size, 1, src len]
        print("In the decoder: ")
        print(hidden[0].shape)
        print(encoder_out.shape)
        if self.bi:
            hidden_attn = torch.cat([hidden[0][-1, :, :], hidden[0][-2, :, :]], -1)
        else:
            hidden_attn = hidden[0][-1, :, :]

        a = self.attn(hidden_attn, encoder_out, src_mask)
        print(a.shape)
        #pay_attn_to = [batch size, src len, enc hid dim]
        encoder_out = encoder_out.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_out)
        #weighted = [1, batch size, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)

        lstm_in = torch.cat((embedded, weighted), dim = 2)
        hidden_in = hidden.unsqueeze(0)
        output, hidden_out = sorted_rnn(
                lstm_in, src_mask, self.lstm, hidden_in
        )
                    
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        hidden_out = self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [batch size, dec hid dim]

        unpooled = (torch.tanh(self.max_pool_linear(hidden_out))
            .view(-1, num_examples, self.hidden_size)
            .permute(0, 2, 1)
        )
        pooled = F.max_pool1d(unpooled, num_examples).squeeze(2)
        program_embedding = self.softmax_linear(pooled)

        return F.softmax(program_embedding, dim=1)


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
