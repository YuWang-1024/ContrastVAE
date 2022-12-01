import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, Decoder, VariationalDropout
import math
import numpy as np
import random

class ContrastVAE(nn.Module):

    def __init__(self, args):
        super(ContrastVAE, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder_mu = Encoder(args)
        self.item_encoder_logvar = Encoder(args)
        self.item_decoder = Decoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.latent_dropout = nn.Dropout(args.reparam_dropout_rate)
        self.apply(self.init_weights)
        self.temperature = nn.Parameter(torch.zeros(1), requires_grad=True)

    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence) # shape: b*max_Sq*d
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb # shape: b*max_Sq*d


    def extended_attention_mask(self, input_ids):
        attention_mask = (input_ids > 0).long()# used for mu, var
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64 b*1*1*max_Sq
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8 for causality
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) #1*1*max_Sq*max_Sq
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask #shape: b*1*max_Sq*max_Sq
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


    def eps_anneal_function(self, step):

        return min(1.0, (1.0*step)/self.args.total_annealing_step)

    def reparameterization(self, mu, logvar, step):  # vanila reparam

        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            res = mu + std * eps
        else: res = mu + std
        return res

    def reparameterization1(self, mu, logvar, step): # reparam without noise
        std = torch.exp(0.5*logvar)
        return mu+std


    def reparameterization2(self, mu, logvar, step): # use dropout

        if self.training:
            std = self.latent_dropout(torch.exp(0.5*logvar))
        else: std = torch.exp(0.5*logvar)
        res = mu + std
        return res

    def reparameterization3(self, mu, logvar,step): # apply classical dropout on whole result
        std = torch.exp(0.5*logvar)
        res = self.latent_dropout(mu + std)
        return res


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def encode(self, sequence_emb, extended_attention_mask): # forward

        item_encoded_mu_layers = self.item_encoder_mu(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        item_encoded_logvar_layers = self.item_encoder_logvar(sequence_emb, extended_attention_mask,
                                                              True)

        return item_encoded_mu_layers[-1], item_encoded_logvar_layers[-1]

    def decode(self, z, extended_attention_mask):
        item_decoder_layers = self.item_decoder(z,
                                                extended_attention_mask,
                                                output_all_encoded_layers = True)
        sequence_output = item_decoder_layers[-1]
        return sequence_output



    def forward(self, input_ids, aug_input_ids, step):

        sequence_emb = self.add_position_embedding(input_ids)# shape: b*max_Sq*d
        extended_attention_mask = self.extended_attention_mask(input_ids)

        if self.args.latent_contrastive_learning:
            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(sequence_emb, extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2 = self.reparameterization2(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)
            return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2

        elif self.args.latent_data_augmentation:
            aug_sequence_emb = self.add_position_embedding(aug_input_ids)  # shape: b*max_Sq*d
            aug_extended_attention_mask = self.extended_attention_mask(aug_input_ids)

            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(aug_sequence_emb, aug_extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2 = self.reparameterization2(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)
            return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2

        else: # vanilla attentive VAE
            mu, log_var = self.encode(sequence_emb, extended_attention_mask)
            z = self.reparameterization(mu, log_var, step)
            reconstructed_seq1 = self.decode(z, extended_attention_mask)
            return reconstructed_seq1, mu, log_var





class ContrastVAE_VD(ContrastVAE):

    def __init__(self, args):
        super(ContrastVAE, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.item_encoder_mu = Encoder(args)
        self.item_encoder_logvar = Encoder(args)
        self.item_decoder = Decoder(args)

        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.latent_dropout_VD = VariationalDropout(inputshape=[args.max_seq_length, args.hidden_size], adaptive='layerwise')
        self.latent_dropout = nn.Dropout(0.1)
        self.args = args
        self.apply(self.init_weights)

        self.drop_rate = nn.Parameter(torch.tensor(0.2), requires_grad=True)


    def reparameterization3(self, mu, logvar, step): # use drop out

        std, alpha = self.latent_dropout_VD(torch.exp(0.5*logvar))
        res = mu + std
        return res, alpha

    def forward(self, input_ids, augmented_input_ids, step):
        if self.args.variational_dropout:
            sequence_emb = self.add_position_embedding(input_ids)  # shape: b*max_Sq*d
            extended_attention_mask = self.extended_attention_mask(input_ids)
            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(sequence_emb, extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2, alpha = self.reparameterization3(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)

        elif self.args.VAandDA:
            sequence_emb = self.add_position_embedding(input_ids)  # shape: b*max_Sq*d
            extended_attention_mask = self.extended_attention_mask(input_ids)
            aug_sequence_emb = self.add_position_embedding(augmented_input_ids)  # shape: b*max_Sq*d
            aug_extended_attention_mask = self.extended_attention_mask(augmented_input_ids)

            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(aug_sequence_emb, aug_extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2, alpha = self.reparameterization3(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)


        return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2, alpha
