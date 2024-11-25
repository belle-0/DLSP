import torch
from torch import nn

import numpy as np
import settings
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

device = settings.gpuId if torch.cuda.is_available() else 'cpu'

class DLSP(nn.Module):
    def __init__(
            self,
            vocab_size,
            f_embed_size=60,
            num_encoder_layers=1,
            num_lstm_layers=1,
            num_rnn_layers=4,
            num_gru_layers=2,
            num_heads=1,
            forward_expansion=2,
            dropout_p=0.5
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.total_embed_size = f_embed_size * 5

        # Layers
        self.embedding = CheckInEmbedding(
            f_embed_size,
            vocab_size
        )
        self.encoder = TransformerEncoder(
            self.embedding,
            self.total_embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout_p,
        )
        self.lstm = nn.LSTM(
            input_size=self.total_embed_size,
            hidden_size=self.total_embed_size,
            num_layers=num_lstm_layers,
            dropout=0
        )
        self.rnn = nn.RNN(
            input_size=self.total_embed_size,
            hidden_size=self.total_embed_size,
            num_layers=num_rnn_layers
        )
        self.gru = nn.GRU(
            input_size=self.total_embed_size, 
            hidden_size=self.total_embed_size, 
            num_layers=num_gru_layers
        )
        self.tcn = TCN(
            num_inputs=self.total_embed_size, 
            channels=[128,128,64,self.total_embed_size]
        )
        self.final_attention = Attention(
            qdim=f_embed_size,
            kdim=self.total_embed_size
        )
        self.out_linear = nn.Sequential(nn.Linear(self.total_embed_size, self.total_embed_size * forward_expansion),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout_p),
                                        nn.Linear(self.total_embed_size * forward_expansion, vocab_size["POI"]))

        self.loss_func = nn.CrossEntropyLoss()

        self.tryone_line2 = nn.Linear(self.total_embed_size, f_embed_size)
        self.enhance_val = nn.Parameter(torch.tensor(0.5))
        self.fc = nn.Linear(self.total_embed_size, self.total_embed_size)
        self.output_layer = nn.Linear(self.total_embed_size, 1)
        self.CL_builder = Contrastive_BPR()

    def feature_mask(self, sequences, mask_prop):
        masked_sequences = []
        for seq in sequences: 
            feature_seq, day_nums = seq[0], seq[1]
            seq_len = len(feature_seq[0])
            mask_count = torch.ceil(mask_prop * torch.tensor(seq_len)).int()
            masked_index = torch.randperm(seq_len - 1) + torch.tensor(1)
            masked_index = masked_index[:mask_count] 

            feature_seq[0, masked_index] = self.vocab_size["POI"]  # mask POI
            feature_seq[1, masked_index] = self.vocab_size["cat"]  # mask cat
            feature_seq[3, masked_index] = self.vocab_size["hour"]  # mask hour
            feature_seq[4, masked_index] = self.vocab_size["day"]  # mask day

            masked_sequences.append((feature_seq, day_nums))
        return masked_sequences

    def forward(self, sample):
        long_term_sequences = sample[:-1]
        short_term_sequence = sample[-1]
        short_term_features = short_term_sequence[0][:, :- 1]
        target = short_term_sequence[0][0, -1]
        user_id = short_term_sequence[0][2, 0]

        # Long-term
        long_term_sequences = self.feature_mask(long_term_sequences, settings.mask_prop)
        long_term_out = []
        u_long_term_out = []
        for seq in long_term_sequences:
            u_long = self.embedding(seq[0])
            output = self.encoder(u_long)
            long_term_out.append(output)
            u_long_term_out.append(u_long)
        long_term_catted = torch.cat(long_term_out, dim=0)
        u_long_term_catted = torch.cat(u_long_term_out, dim=0)

        # Short-term
        short_term_feature = self.embedding(short_term_features)
        short_term_state0 = torch.unsqueeze(short_term_feature,0).permute(1,0,2)
        short_term_state1, _ = self.gru(short_term_state0)
        short_term_state2 = short_term_state1.permute(1,0,2)
        short_term_state = torch.squeeze(short_term_state2)

        # User enhancement
        user_embed = self.embedding.user_embed(user_id)
        embedding = torch.unsqueeze(self.embedding(short_term_features), 0)
        embedding = embedding.permute(0, 2, 1)
        output1 = self.tcn(embedding)
        output = output1.permute(0, 2, 1)

        short_term_enhance = torch.squeeze(output)
        user_embed = self.enhance_val * user_embed + (1 - self.enhance_val) * self.tryone_line2(
            torch.mean(short_term_enhance, dim=0))

        u_embed_mean_long = torch.mean(u_long_term_catted, dim=0)
        short_embed_mean = torch.mean(short_term_state, dim=0)
        long_embed_mean = torch.mean(long_term_catted, dim=0)
        u_embed_mean_short = torch.mean(short_term_feature, dim=0)
        ssl_loss = self.CL_builder(short_embed_mean, u_embed_mean_short, long_embed_mean) + \
                    self.CL_builder(long_embed_mean, u_embed_mean_long, short_embed_mean)

        dot_product = torch.dot(u_embed_mean_long, u_embed_mean_short)
        norm_vec1 = torch.norm(u_embed_mean_long)
        norm_vec2 = torch.norm(u_embed_mean_short)
        cosine_sim = dot_product / (norm_vec1 * norm_vec2)
        h = (cosine_sim + 1) / 2
        g = 1-h

        n2 = short_term_state.shape[0]
        n1 = long_term_catted.shape[0]
        result_length = max(n1, n2)
        h_all = torch.zeros(result_length, self.total_embed_size)
        for i in range(result_length):
            if i < n2 and i < n1:
                h_all[i] = h * short_term_state[i] + g * long_term_catted[i]
            elif i < n2:
                h_all[i] = h * short_term_state[i]
            elif i < n1:
                h_all[i] = g * long_term_catted[i]
        
        h_all = h_all.to(device=device)

        final_att = self.final_attention(user_embed, h_all, h_all)
        output = self.out_linear(final_att)  
        
        label = torch.unsqueeze(target, 0)   
        pred = torch.unsqueeze(output, 0)   
        pred_loss = self.loss_func(pred, label)
        loss = pred_loss + ssl_loss * settings.neg_weight
        return loss, output, short_term_state, long_term_catted, user_embed
    

    def predict(self, sample):
        _, pred_raw, short_term_state, long_term_catted, user_embed = self.forward(sample)
        ranking = torch.sort(pred_raw, descending=True)[1]
        target = sample[-1][0][0, -1]
        return ranking, target, short_term_state, long_term_catted, user_embed


class CheckInEmbedding(nn.Module):
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)

    def forward(self, x):
        poi_emb = self.poi_embed(x[0])
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])

        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb), 1)

class Contrastive_BPR(nn.Module):
    def __init__(self):
        super(Contrastive_BPR, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
    def forward(self, x, pos, neg):
        pos_score = self.sigmoid((x * pos).sum(-1))
        neg_score = self.sigmoid((x * neg).sum(-1))
        pos_loss = self.bce_loss(pos_score, torch.ones_like(pos_score))
        neg_loss = self.bce_loss(neg_score, torch.zeros_like(neg_score))
        return pos_loss + neg_loss

    
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query):
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)
        values = values.reshape(value_len, self.heads, self.head_dim)
        keys = keys.reshape(key_len, self.heads, self.head_dim)
        queries = queries.reshape(query_len, self.heads, self.head_dim)
        energy = torch.einsum("qhd,khd->hqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)
        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out
    

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embedding_layer,
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.add_module('embedding', self.embedding_layer)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.embed_size = embed_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq):
        out = self.dropout(feature_seq)
        for layer in self.layers:
            out = layer(out, out, out)
        return out


class Attention(nn.Module):
    def __init__(
            self,
            qdim,
            kdim,
    ):
        super().__init__()
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):
        q = self.expansion(query) 
        temp = torch.inner(q, key)
        weight = torch.softmax(temp, dim=0) 
        weight = torch.unsqueeze(weight, 1)
        temp2 = torch.mul(value, weight)
        out = torch.sum(temp2, 0)
        return out

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
      
        self.chomp1 = Chomp1d(padding)  
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding) 
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, num_inputs, channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation_size = 2 ** i 
            in_channels = num_inputs if i == 0 else channels[i - 1] 
            out_channels = channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        return x