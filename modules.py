import numpy as np

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""activation function"""
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

"""Transformer toolkits"""

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias



class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states




class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states



class Layer(nn.Module): # attention block
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output




class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """

        :param hidden_states: bxmax_Sqxd
        :param attention_mask: b*1*max_Sq*max_Sq
        :param output_all_encoded_layers: True or False
        :return:
        """

        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """

        :param hidden_states: bxmax_Sqxd
        :param attention_mask: b*1*max_Sq*max_Sq
        :param output_all_encoded_layers: True or False
        :return:
        """
        
        all_decoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_decoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_decoder_layers.append(hidden_states)
        return all_decoder_layers


class NCELoss(nn.Module):
    """
    """

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two):  # batch_size*

        sim11 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_one.unsqueeze(-3)) / self.temperature
        sim22 = self.cossim(batch_sample_two.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature
        sim12 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature

        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss


"""toolkit for variational dropout"""

def _logit(x):
    return np.log(x/(1. - x))

def _check_p(p):
    if p == 0.5:
        return 0.4999
    elif p > 0.5:
        return 0.4999
    elif p <= 0.0:
        return 0.0001
    else:
        return p

class VariationalDropout(nn.Module):
    def __init__(self, inputshape, p=0.2, adaptive=None):
        super(VariationalDropout, self).__init__()

        self.adaptive = adaptive
        p = _check_p(p)

        if self.adaptive == None:
            self.logitalpha =nn.Parameter(torch.tensor(_logit(np.sqrt(p / (1. - p)))
                ), requires_grad=False)

        elif self.adaptive == 'layerwise':
            self.logitalpha =nn.Parameter(torch.tensor(_logit(np.sqrt(p / (1. - p)))
                ), requires_grad=True)

        elif self.adaptive == "elementwise":
            # initialise param for each activation passed
            self.logitalpha =nn.Parameter(torch.tensor(
                    np.ones(inputshape[1:]).astype(np.float32) * _logit(np.sqrt(p / (1. - p)))
                ), requires_grad=True)

        elif self.adaptive == "weightwise":
            # this will only work in the case of dropout type B
            self.logitalpha =nn.Parameter(torch.tensor(np.ones(inputshape).astype(np.float32) * _logit(np.sqrt(p / (1. - p)))
                ), requires_grad=True)

    def forward(self, x):

        alpha = F.sigmoid(self.logitalpha)
        output =x*alpha*torch.randn_like(x)

        return output, alpha


def priorKL(alpha):

    c1 = 1.161451241083230
    c2 = -1.502041176441722
    c3 = 0.586299206427007
    return -torch.mean(0.5 * torch.log(alpha) + c1 * alpha + c2 * torch.pow(alpha, 2) + c3 * torch.pow(alpha, 3))


