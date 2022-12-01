import numpy as np
import tqdm
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import Adam
from modules import NCELoss, priorKL
from utils import recall_at_k, ndcg_k, get_metric, cal_mrr, get_user_performance_perpopularity, get_item_performance_perpopularity

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]), flush=True)
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def complicated_eval(self, user_seq, args):
        return self.eval_analysis(self.test_dataloader, user_seq, args)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def eval_analysis(self, dataloader, seqs):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix, flush=True)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix), None

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg, mrr = [], [], 0
        recall_dict_list = []
        ndcg_dict_list = []
        for k in [1, 5, 10, 15, 20, 40]:
            recall_result, recall_dict_k = recall_at_k(answers, pred_list, k)
            recall.append(recall_result)
            recall_dict_list.append(recall_dict_k)
            ndcg_result, ndcg_dict_k = ndcg_k(answers, pred_list, k)
            ndcg.append(ndcg_result)
            ndcg_dict_list.append(ndcg_dict_k)
        mrr, mrr_dict = cal_mrr(answers, pred_list)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.8f}'.format(recall[0]), "NDCG@1": '{:.8f}'.format(ndcg[0]),
            "HIT@5": '{:.8f}'.format(recall[1]), "NDCG@5": '{:.8f}'.format(ndcg[1]),
            "HIT@10": '{:.8f}'.format(recall[2]), "NDCG@10": '{:.8f}'.format(ndcg[2]),
            "HIT@15": '{:.8f}'.format(recall[3]), "NDCG@15": '{:.8f}'.format(ndcg[3]),
            "HIT@20": '{:.8f}'.format(recall[4]), "NDCG@20": '{:.8f}'.format(ndcg[4]),
            "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(ndcg[5]),
            "MRR": '{:.8f}'.format(mrr)
        }
        print(post_fix, flush=True)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[3], ndcg[3], recall[4], ndcg[4], recall[5], ndcg[5], mrr], str(post_fix), [recall_dict_list, ndcg_dict_list, mrr_dict]

    def get_pos_items_ranks(self, batch_pred_lists, answers):
        num_users = len(batch_pred_lists)
        batch_pos_ranks = defaultdict(list)
        for i in range(num_users):
            pred_list = batch_pred_lists[i]
            true_set = set(answers[i])
            for ind, pred_item in enumerate(pred_list):
                if pred_item in true_set:
                    batch_pos_ranks[pred_item].append(ind+1)
        return batch_pos_ranks

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location='cuda:0'))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc
    

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class ContrastVAETrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        super(ContrastVAETrainer, self).__init__(model, train_dataloader, eval_dataloader, test_dataloader,args)
        self.step = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cl_criterion = NCELoss(args.temperature, device)
        self.variational_dropout = args.variational_dropout
        self.args = args


    def kl_anneal_function(self, anneal_cap, step, total_annealing_step):
        """

        :param step: increment by 1 for every  forward-backward step
        :param k: temperature for logistic annealing
        :param x0: pre-fixed parameter control the speed of anealing. total annealing steps
        :return:
        """
        # borrows from https://github.com/timbmg/Sentence-VAE/blob/master/train.py
        return min(anneal_cap, (1.0*step)/total_annealing_step)

    def loss_fn_vanila(self, reconstructed_seq1, mu, log_var, target_pos_seq, target_neg_seq, step):
        """
        compute kl divergence, reconstruction
        :param sequence_reconstructed: b*max_Sq*d
        :param mu: b*d
        :param log_var: b*d
        :param target_pos_seq: b*max_Sq*d
        :param target_neg_seq: b*max_Sq*d
        :return:
        """

        """compute KL divergence"""

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1))
        kld_weight = self.kl_anneal_function(self.args.anneal_cap, step, self.args.total_annealing_step)

        """compute reconstruction loss from Trainer"""
        recons_loss1, recons_auc = self.cross_entropy(reconstructed_seq1, target_pos_seq, target_neg_seq)


        loss = recons_loss1  + kld_weight*kld_loss

        return loss, recons_auc


    def loss_fn_latent_clr(self, reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2, target_pos_seq, target_neg_seq, step):
        """
        compute kl divergence, reconstruction loss and contrastive loss
        :param sequence_reconstructed: b*max_Sq*d
        :param mu: b*d
        :param log_var: b*d
        :param target_pos_seq: b*max_Sq*d
        :param target_neg_seq: b*max_Sq*d
        :return:
        """

        """compute KL divergence"""

        kld_loss1 = torch.mean(-0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - log_var1.exp(), dim=-1))
        kld_loss2 = torch.mean(-0.5 * torch.sum(1 + log_var2 - mu2 ** 2 - log_var2.exp(), dim=-1))
        kld_weight = self.kl_anneal_function(self.args.anneal_cap, step, self.args.total_annealing_step)

        """compute reconstruction loss from Trainer"""
        recons_loss1, recons_auc = self.cross_entropy(reconstructed_seq1, target_pos_seq, target_neg_seq)
        recons_loss2, recons_auc = self.cross_entropy(reconstructed_seq2, target_pos_seq, target_neg_seq)

        """compute clr loss"""
        user_representation1 = torch.sum(z1, dim=1)
        user_representation2 = torch.sum(z2, dim=1)

        contrastive_loss = self.cl_criterion(user_representation1, user_representation2)


        loss = recons_loss1 + recons_loss2 + kld_weight*(kld_loss1 + kld_loss2) + self.args.latent_clr_weight * contrastive_loss
        return loss, recons_auc


    def loss_fn_VD_latent_clr(self, reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1,z2, target_pos_seq, target_neg_seq, step, alpha):
        """
        compute kl divergence, reconstruction loss and contrastive loss
        :param sequence_reconstructed: b*max_Sq*d
        :param mu: b*d
        :param log_var: b*d
        :param target_pos_seq: b*max_Sq*d
        :param target_neg_seq: b*max_Sq*d
        :return:
        """

        """compute KL divergence"""

        kld_loss1 = torch.mean(-0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - log_var1.exp(), dim=-1))
        kld_loss2 = torch.mean(-0.5 * torch.sum(1 + log_var2 - mu2 ** 2 - log_var2.exp(), dim=-1))
        kld_weight = self.kl_anneal_function(self.args.anneal_cap, step, self.args.total_annealing_step)

        """compute reconstruction loss from Trainer"""
        recons_loss1, recons_auc = self.cross_entropy(reconstructed_seq1, target_pos_seq, target_neg_seq)
        recons_loss2, recons_auc = self.cross_entropy(reconstructed_seq2, target_pos_seq, target_neg_seq)

        """compute clr loss"""

        user_representation1 = torch.sum(z1, dim=1)
        user_representation2 = torch.sum(z2, dim=1)
        contrastive_loss = self.cl_criterion(user_representation1, user_representation2)
        
        """compute priorKL loss"""
        adaptive_alpha_loss = priorKL(alpha)
        loss = recons_loss1 + recons_loss2 + kld_weight * (kld_loss1 + kld_loss2) + self.args.latent_clr_weight * contrastive_loss+ adaptive_alpha_loss
      
        return loss, recons_auc

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        rec_data_iter = dataloader
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_auc = 0.0


            for batch in rec_data_iter:
                
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _,aug_input_ids = batch # input_ids, target_ids: [b,max_Sq]


                if self.variational_dropout:
                    # reconstructed_seq1, reconstructed_seq2, mu, log_var, alpha = self.model.forward(input_ids, self.step)  # shape:b*max_Sq*d
                    reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2, alpha = self.model.forward(input_ids,0, self.step)
                    loss, recons_auc = self.loss_fn_VD_latent_clr(reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2,z1,z2, target_pos, target_neg, self.step, alpha)

                elif self.args.latent_contrastive_learning:
                    reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2 = self.model.forward(input_ids, 0,self.step)
                    loss, recons_auc  = self.loss_fn_latent_clr(reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2, target_pos, target_neg, self.step)

                elif self.args.latent_data_augmentation:
                    reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2 = self.model.forward(input_ids, aug_input_ids, self.step)
                    loss, recons_auc = self.loss_fn_latent_clr(reconstructed_seq1, reconstructed_seq2, mu1, mu2,log_var1, log_var2, z1, z2, target_pos, target_neg,self.step)
                
                elif self.args.VAandDA:
                    reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2, alpha = self.model.forward(input_ids, aug_input_ids, self.step)
                    loss, recons_auc = self.loss_fn_VD_latent_clr(reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2,z1,z2, target_pos, target_neg, self.step, alpha)

                else:
                    reconstructed_seq1,  mu, log_var = self.model.forward(input_ids, 0, self.step)  # shape:b*max_Sq*d
                    loss, recons_auc = self.loss_fn_vanila(reconstructed_seq1, mu, log_var, target_pos, target_neg, self.step)



                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                self.step += 1
                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += recons_auc.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.4f}'.format(rec_avg_auc / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            with torch.no_grad():
                pred_list = None

                if self.args.store_latent:
                    user_embeddings = torch.zeros((self.args.num_users, self.args.hidden_size))
                    seq_mus = torch.zeros((self.args.num_users, self.args.max_seq_length, self.args.hidden_size))
                    seq_logvar = torch.zeros((self.args.num_users, self.args.max_seq_length, self.args.hidden_size))

                if full_sort:
                    answer_list = None
                    #for i, batch in rec_data_iter:
                    print(f"full sort evaluation")
                    i = 0
                    for batch in rec_data_iter:
                        # 0. batch_data will be sent into the device(GPU or cpu)
                        batch = tuple(t.to(self.device) for t in batch)
                        user_ids, input_ids, target_pos, target_neg, answers,aug_input_ids = batch
                        istarget =torch.unsqueeze((target_pos > 0),-1) # [batch*seq_len]

                        if self.variational_dropout:
                            recommend_reconstruct1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2, alpha= self.model.forward(input_ids,0, self.step)
                            if self.args.store_latent:
                                user_embeddings[user_ids, :] = torch.sum(z1*istarget, 1).cpu()
                                seq_mus[user_ids, :,:] = mu1.cpu()
                                seq_logvar[user_ids, :,:] = log_var1.cpu()


                        elif self.args.latent_contrastive_learning:
                            recommend_reconstruct1, recommend_reconstruct2, mu1, mu2, log_var1, log_var2, z1, z2 = self.model.forward(input_ids, 0, self.step)
                            if self.args.store_latent:
                                user_embeddings[user_ids, :] = torch.sum(z1*istarget, 1).cpu()
                                seq_mus[user_ids, :, :] = mu1.cpu()
                                seq_logvar[user_ids,:, :] = log_var1.cpu()

                        elif self.args.latent_data_augmentation == True:
                            recommend_reconstruct1, recommend_reconstruct2, mu1, mu2, log_var1, log_var2, z1, z2 = self.model.forward(
                                input_ids, aug_input_ids, self.step)
                            if self.args.store_latent:
                                user_embeddings[user_ids, :] = (z1*istarget).sum(1).cpu()
                                seq_mus[user_ids,:, :] = mu1.cpu()
                                seq_logvar[user_ids,:, :] = log_var1.cpu()

                        elif self.args.VAandDA:
                            recommend_reconstruct1, _, mu1, mu2, log_var1, log_var2, z1, z2, alpha = self.model.forward(input_ids, aug_input_ids, self.step)
                            if self.args.store_latent:
                                user_embeddings[user_ids, :] = (z1*istarget).sum(1).cpu()
                                seq_mus[user_ids,:, :] = mu1.cpu()
                                seq_logvar[user_ids,:, :] = log_var1.cpu()
                                
                        else: # vanila beta-vae with transformerr
                            recommend_reconstruct1, mu, log_var,= self.model.forward(input_ids,0, self.step)
                            if self.args.store_latent:
                                res = mu + torch.exp(0.5 * log_var)
                                user_embeddings[user_ids, :] = torch.sum(res*istarget, 1).cpu()
                                seq_mus[user_ids,:, :] = mu.cpu()
                                seq_logvar[user_ids,:, :] = log_var.cpu()

                        recommend_output = recommend_reconstruct1[:, -1, :]
                        rating_pred = self.predict_full(recommend_output)

                        rating_pred = rating_pred.cpu().data.numpy().copy()
                        batch_user_index = user_ids.cpu().numpy()

                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                        ind = np.argpartition(rating_pred, -40)[:, -40:]
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                        if i == 0:
                            pred_list = batch_pred_list
                            answer_list = answers.cpu().data.numpy()
                        else:
                            pred_list = np.append(pred_list, batch_pred_list, axis=0)
                            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                        i += 1

                    return self.get_full_sort_score(epoch, answer_list, pred_list)


                else:
                    assert "We need full_sort evaluation"    
                    return 0



