# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch
import argparse
import pdb

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SeqDataset
from trainers import ContrastVAETrainer
from models import ContrastVAE, ContrastVAE_VD
from utils import EarlyStopping, get_user_seqs, get_user_seqs_replace, get_item2attribute_json, check_path, set_seed


def main():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Toys_and_Games', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")


    # model args
    parser.add_argument("--model_name", default='ContrastVAE', type=str)
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of layers")
    parser.add_argument('--num_attention_heads', default=4, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.0, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=100, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=400, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # model variants
    parser.add_argument("--variational_dropout", action='store_true')
    parser.add_argument("--VAandDA", action='store_true')
    parser.add_argument("--latent_contrastive_learning", action='store_true')
    parser.add_argument('--latent_data_augmentation', action='store_true')

    parser.add_argument("--latent_clr_weight", type=float, default=0.3, help="weight for latent clr loss")
    parser.add_argument("--reparam_dropout_rate", type=float, default=0.2, help="dropout rate for reparameterization dropout")
    parser.add_argument("--store_latent", action='store_true', help="store the latent representation of sequence embedding")

    # KL annealing args
    parser.add_argument('--anneal_cap', type=float, default=0.3)
    parser.add_argument('--total_annealing_step', type=int, default=10000)

    # contrastive args
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--i', type=int)
    parser.add_argument('--eval_model_path', type = str)
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda


    args.data_file = args.data_dir + args.data_name + '.txt'


    """
    load data
    user_seq: original list contains all interacted items
    max_item: number of all items plus 0
    valid_rating_matrix: shape (num_users, num_items), sparse matrix, value:1, row: user_id, col: item_id, record [:-2] items from user_seq
    test_rating_matrix: same as valid_rating_matrix, but record [:-1] items
    """

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
        get_user_seqs(args.data_file)


    args.item_size = max_item + 2
    args.num_users = num_users
    args.mask_id = max_item + 1


    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix
    print(f"valid rating matix shape: {valid_rating_matrix.shape}")





    # save model args
    args_str = f'{args.model_name}' \
               f'-{args.data_name}' \
               f'-{args.hidden_size}' \
               f'-{args.num_hidden_layers}' \
               f'-{args.num_attention_heads}' \
               f'-{args.hidden_act}' \
               f'-{args.attention_probs_dropout_prob}' \
               f'-{args.hidden_dropout_prob}' \
               f'-{args.max_seq_length}' \
               f'-{args.lr}' \
               f'-{args.weight_decay}' \
               f'-{args.anneal_cap}' \
               f'-{args.total_annealing_step}' \
               f'-{args.reparam_dropout_rate}'\
               f'-{args.latent_clr_weight}'

    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)


    train_dataset = SeqDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SeqDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)


    test_dataset = SeqDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    if args.variational_dropout:
        model = ContrastVAE_VD(args)
    else: model = ContrastVAE(args=args)

    trainer = ContrastVAETrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)

    if args.do_eval:
        # load the best model
        print('---------------load best model and do eval-------------------')
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        trainer.args.train_matrix = test_rating_matrix
        trainer.test('best', full_sort=True)
    else:

        early_stopping = EarlyStopping(args.checkpoint_path, patience=200, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array([scores[4], scores[5]]), trainer.model) # here only check best recall@10, ndcg@10
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        valid_scores, _, _ = trainer.valid('best', full_sort=True)
        trainer.args.train_matrix = test_rating_matrix
        scores, result_info, _ = trainer.test('best', full_sort=True)

        print(args_str)
        with open(args.log_file, 'a') as f:
            f.write(args_str + '\n')
            f.write(result_info + '\n')


if __name__ == '__main__':
    main()
