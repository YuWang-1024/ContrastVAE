# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 11:06
# @Author  : Hui Wang

import numpy as np
import math
import random
import os
import json
import pickle
from scipy.sparse import csr_matrix
from tqdm import tqdm
import multiprocessing

import torch
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def neg_sample(item_set, item_size):  # random sample an item id that is not in the user's interact history
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score





def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)

def avg_pooling(x, dim):
    return x.sum(dim=dim)/x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_user_seqs(data_file):
    """
    load txt data file
    :param data_file: path of dataset, every line: [user_id item1 item2 item3 ...]
    :return:
    """
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users

def get_user_seqs_replace(data_file, num_items):
    """
    load txt data file
    :param data_file: path of dataset, every line: [user_id item1 item2 item3 ...]
    :return:
    """
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)

    num_users = len(lines)
    # num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, valid_rating_matrix, test_rating_matrix, num_users

def get_user_seqs_long(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        long_sequence.extend(items) # 后面的都是采的负例
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence

def get_user_seqs_and_sample(data_file, sample_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, sample_seq

def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set) # 331
    return item2attribute, attribute_size

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)

def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    recall_dict = {}
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            #sum_recall += len(act_set & pred_set) / float(len(act_set))
            one_user_recall = len(act_set & pred_set) / float(len(act_set))
            recall_dict[i] = one_user_recall
            sum_recall += one_user_recall
            true_users += 1
    return sum_recall / true_users, recall_dict

def cal_mrr(actual, predicted):
    sum_mrr = 0.
    true_users = 0
    num_users = len(predicted)
    mrr_dict = {}
    for i in range(num_users):
        r = []
        act_set = set(actual[i])
        pred_list = predicted[i]
        for item in pred_list:
            if item in act_set:
                r.append(1)
            else:
                r.append(0)
        r = np.array(r)
        if np.sum(r) > 0:
            #sum_mrr += np.reciprocal(np.where(r==1)[0]+1, dtype=np.float)[0]
            one_user_mrr = np.reciprocal(np.where(r==1)[0]+1, dtype=np.float)[0]
            sum_mrr += one_user_mrr
            true_users += 1
            mrr_dict[i] = one_user_mrr
        else:
            mrr_dict[i] = 0.
    return sum_mrr / len(predicted), mrr_dict


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def ndcg_k(actual, predicted, topk):
    res = 0
    ndcg_dict = {}
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
        ndcg_dict[user_id] = dcg_k / idcg
    return res / float(len(actual)), ndcg_dict


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def itemperf_recall(ranks, k):
    ranks = np.array(ranks)
    if len(ranks) == 0:
        return 0
    return np.sum(ranks<=k) / len(ranks)

def itemperf_ndcg(ranks, k, size):
    ndcg = 0.0
    if len(ranks) == 0:
        return 0.
    for onerank in ranks:
        r = np.zeros(size)
        r[onerank-1] = 1
        ndcg += ndcg_at_k(r, k)
    return ndcg / len(ranks)


def get_user_performance_perpopularity(train, results_users, Ks):
    [recall_dict_list, ndcg_dict_list, mrr_dict] = results_users
    short_seq_results = {
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "mrr": 0.,
    }
    num_short_seqs = 0

    long_seq_results = {
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "mrr": 0.,
    }
    num_long_seqs = 0

    short7_seq_results = {
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "mrr": 0.,
    }
    num_short7_seqs = 0

    short37_seq_results = {
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "mrr": 0.,
    }
    num_short37_seqs = 0

    medium3_seq_results = {
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "mrr": 0.,
    }

    num_medium3_seqs = 0

    medium7_seq_results = {
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "mrr": 0.,
    }
    num_medium7_seqs = 0

    test_users = list(train.keys())
    for result_user in tqdm(test_users):
        if len(train[result_user]) <= 3:
            num_short_seqs += 1
        if len(train[result_user]) <= 7:
            num_short7_seqs += 1
        if len(train[result_user]) > 3 and len(train[result_user]) <= 7:
            num_short37_seqs += 1
        if len(train[result_user]) > 3 and len(train[result_user]) < 20:
            num_medium3_seqs += 1
        if len(train[result_user]) > 7 and len(train[result_user]) < 20:
            num_medium7_seqs += 1
        if len(train[result_user]) >= 20:
            num_long_seqs += 1
    for k_ind in range(len(recall_dict_list)):
        k = Ks[k_ind]
        recall_dict_k = recall_dict_list[k_ind]
        ndcg_dict_k = ndcg_dict_list[k_ind]

        for result_user in tqdm(test_users):
            if len(train[result_user]) <= 3:
                short_seq_results["recall"][k_ind] += recall_dict_k[result_user]
                short_seq_results["ndcg"][k_ind] += ndcg_dict_k[result_user]

            if len(train[result_user]) <= 7:
                short7_seq_results["recall"][k_ind] += recall_dict_k[result_user]
                short7_seq_results["ndcg"][k_ind] += ndcg_dict_k[result_user]

            if len(train[result_user]) > 3 and len(train[result_user]) <= 7:
                short37_seq_results["recall"][k_ind] += recall_dict_k[result_user]
                short37_seq_results["ndcg"][k_ind] += ndcg_dict_k[result_user]

            if len(train[result_user]) > 3 and len(train[result_user]) < 20:
                medium3_seq_results["recall"][k_ind] += recall_dict_k[result_user]
                medium3_seq_results["ndcg"][k_ind] += ndcg_dict_k[result_user]

            if len(train[result_user]) > 7 and len(train[result_user]) < 20:
                medium7_seq_results["recall"][k_ind] += recall_dict_k[result_user]
                medium7_seq_results["ndcg"][k_ind] += ndcg_dict_k[result_user]

            if len(train[result_user]) >= 20:
                long_seq_results["recall"][k_ind] += recall_dict_k[result_user]
                long_seq_results["ndcg"][k_ind] += ndcg_dict_k[result_user]

    for result_user in tqdm(test_users):
        if len(train[result_user]) <= 3:
            short_seq_results["mrr"] += mrr_dict[result_user]

        if len(train[result_user]) <= 7:
            short7_seq_results["mrr"] += mrr_dict[result_user]

        if len(train[result_user]) > 3 and len(train[result_user]) <= 7:
            short37_seq_results["mrr"] += mrr_dict[result_user]

        if len(train[result_user]) > 3 and len(train[result_user]) < 20:
            medium3_seq_results["mrr"] += mrr_dict[result_user]

        if len(train[result_user]) > 7 and len(train[result_user]) < 20:
            medium7_seq_results["mrr"] += mrr_dict[result_user]

        if len(train[result_user]) >= 20:
            long_seq_results["mrr"] += mrr_dict[result_user]

    if num_short_seqs > 0:
        short_seq_results["recall"] /= num_short_seqs
        short_seq_results["ndcg"] /= num_short_seqs
        short_seq_results["mrr"] /= num_short_seqs
    print(f"testing #of short seq users with less than 3 training points: {num_short_seqs}")

    if num_short7_seqs > 0:
        short7_seq_results["recall"] /= num_short7_seqs
        short7_seq_results["ndcg"] /= num_short7_seqs
        short7_seq_results["mrr"] /= num_short7_seqs
    print(f"testing #of short seq users with less than 7 training points: {num_short7_seqs}")

    if num_short37_seqs > 0:
        short37_seq_results["recall"] /= num_short37_seqs
        short37_seq_results["ndcg"] /= num_short37_seqs
        short37_seq_results["mrr"] /= num_short37_seqs
    print(f"testing #of short seq users with 3 - 7 training points: {num_short37_seqs}")

    if num_medium3_seqs > 0:
        medium3_seq_results["recall"] /= num_medium3_seqs
        medium3_seq_results["ndcg"] /= num_medium3_seqs
        medium3_seq_results["mrr"] /= num_medium3_seqs
    print(f"testing #of short seq users with medium3 training points: {num_medium3_seqs}")

    if num_medium7_seqs > 0:
        medium7_seq_results["recall"] /= num_medium7_seqs
        medium7_seq_results["ndcg"] /= num_medium7_seqs
        medium7_seq_results["mrr"] /= num_medium7_seqs
    print(f"testing #of short seq users with medium7 training points: {num_medium7_seqs}")

    if num_long_seqs > 0:
        long_seq_results["recall"] /= num_long_seqs
        long_seq_results["ndcg"] /= num_long_seqs
        long_seq_results["mrr"] /= num_long_seqs

    print(f"testing #of short seq users with >= 20 training points: {num_long_seqs}")

    print('testshort: ' + str(short_seq_results))
    print('testshort7: ' + str(short7_seq_results))
    print('testshort37: ' + str(short37_seq_results))
    print('testmedium3: ' + str(medium3_seq_results))
    print('testmedium7: ' + str(medium7_seq_results))
    print('testlong: ' + str(long_seq_results))


def eval_one_setitems(x):
    Ks = [1, 5, 10, 15, 20, 40]
    result = {
            "recall": 0,
            "ndcg": 0
    }
    ranks = x[0]
    k_ind = x[1]
    test_num_items = x[2]
    freq_ind = x[3]

    result['recall'] = itemperf_recall(ranks, Ks[k_ind])
    result['ndcg'] = itemperf_ndcg(ranks, Ks[k_ind], test_num_items)

    return result, k_ind, freq_ind


def get_item_performance_perpopularity(items_in_freqintervals, all_pos_items_ranks, Ks, freq_quantiles, num_items):
    cores = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cores)
    test_num_items_in_intervals = []
    interval_results = [{'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))} for _ in range(len(items_in_freqintervals))]

    all_freq_all_ranks = []
    all_ks = []
    all_numtestitems = []
    all_freq_ind = []
    for freq_ind, item_list in enumerate(items_in_freqintervals):
        num_item_pos_interactions = 0
        all_ranks = []
        interval_items = []
        for item in item_list:
            pos_ranks_oneitem = all_pos_items_ranks.get(item, [])
            if len(pos_ranks_oneitem) > 0:
                interval_items.append(item)
            all_ranks.extend(pos_ranks_oneitem)
        for k_ind in range(len(Ks)):
            all_ks.append(k_ind)
            all_freq_all_ranks.append(all_ranks)
            all_numtestitems.append(num_items)
            all_freq_ind.append(freq_ind)
        test_num_items_in_intervals.append(interval_items)

    item_eval_freq_data = zip(all_freq_all_ranks, all_ks, all_numtestitems, all_freq_ind)
    batch_item_result = pool.map(eval_one_setitems, item_eval_freq_data)


    for oneresult in batch_item_result:
        result_dict = oneresult[0]
        k_ind = oneresult[1]
        freq_ind = oneresult[2]
        interval_results[freq_ind]['recall'][k_ind] = result_dict['recall']
        interval_results[freq_ind]['ndcg'][k_ind] = result_dict['ndcg']



    item_freq = freq_quantiles
    for i in range(len(item_freq)+1):
        if i == 0:
            print('For items in freq between 0 - %d with %d items: ' % (item_freq[i], len(test_num_items_in_intervals[i])))
        elif i == len(item_freq):
            print('For items in freq between %d - max with %d items: ' % (item_freq[i-1], len(test_num_items_in_intervals[i])))
        else:
            print('For items in freq between %d - %d with %d items: ' % (item_freq[i-1], item_freq[i], len(test_num_items_in_intervals[i])))
        for k_ind in range(len(Ks)):
            k = Ks[k_ind]
            print('Recall@%d:%.6f, NDCG@%d:%.6f'%(k, interval_results[i]['recall'][k_ind], k, interval_results[i]['ndcg'][k_ind]))
