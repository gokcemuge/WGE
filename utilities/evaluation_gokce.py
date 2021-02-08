import sklearn
from sklearn import tree
import torch
import numpy as np
from time import time
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from torch.autograd import Variable
from utilities.data_utilities import get_minibatches
from utilities.negative_sampling import *
import time
import matplotlib.pyplot as plt
from scipy.stats import rankdata


def plot_mse_pos(test_pos_score, test_pos_weights, batch_no):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 50
    fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size

    scores = test_pos_score.cpu().numpy()
    weights = test_pos_weights.cpu().numpy()

    # sorting
    #merged = tuple(zip(weights, scores))
    #sorted_tuples = sorted(merged, key=lambda x: x[0])
    #unzipped = list(zip(*sorted_tuples))
    #weights = unzipped[0]
    #scores = unzipped[1]

    triple_numbers = range(len(weights))

    plt.plot(triple_numbers[:120], weights[:120], 'go', label='weights')
    plt.plot(triple_numbers[:120], scores[:120], 'r+', label='scores')
    plt.title('Weights vs Scores')
    plt.xlabel('triples')
    plt.ylabel('scores')
    plt.legend()
    plot_dir = 'plots/mse_pos_val_batch_' + str(batch_no) + '.png'
    plt.savefig(plot_dir)
    plt.clf()

    file_name = 'mse_pos_val_batch_' + str(batch_no)
    triple_no = triple_numbers[:120]
    weight_no = weights[:120]
    score_no = scores[:120]

    np.save('plots/triples/' + str(file_name) + '_triples', triple_no )
    np.save('plots/weights/' + str(file_name) + '_weights', weight_no)
    np.save('plots/scores/' + str(file_name) + '_scores', score_no)
    return


def plot_mse_neg(test_neg_score, batch_no):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 50
    fig_size[1] = 4
    plt.rcParams["figure.figsize"] = fig_size

    scores = test_neg_score.cpu()
    triple_numbers = range(scores.shape[1])

    # scores = sorted(scores) # takes too much time because 10 times of a batch size

    plt.plot(triple_numbers, scores.view(-1), 'r+', label='scores')
    plt.title('Weights vs Scores')
    plt.xlabel('triples')
    plt.ylabel('scores')
    plt.legend()
    plot_dir = 'plots/mse_neg_val_batch_' + str(batch_no) + '.png'
    plt.savefig(plot_dir)
    plt.clf()
    return


def get_mse_pos(model, test_pos, test_batch_size, plot):
    with torch.no_grad():
        total_mse_pos = 0
        total_mae_pos = 0
        test_triples = list(get_minibatches(test_pos, test_batch_size, shuffle=True))
        positive_test_triples_size = test_pos.shape[0]
        batch_no = 0

        if model.name == 'UKGE_logi' or model.name == 'WGE_logi' or model.name == 'rescal':
            for iter_triple in test_triples:
                test_pos_weights = iter_triple[:, 3].astype(np.float64)
                test_pos_score = model.forward(iter_triple)

                test_pos_weights = torch.from_numpy(test_pos_weights).view(1, -1).transpose(0, 1).cuda()
                test_pos_score = test_pos_score.view(1, -1).transpose(0, 1)

                if plot:
                    plot_mse_pos(test_pos_score, test_pos_weights, batch_no)
                    batch_no = batch_no + 1

                mse = torch.sum((test_pos_score-test_pos_weights)**2)
                mae = torch.sum(torch.abs(test_pos_score - test_pos_weights))
                total_mse_pos = total_mse_pos + mse
                total_mae_pos = total_mae_pos + mae
                if np.asarray(iter_triple).shape[0] < test_batch_size:
                    break

        elif model.name == 'WGE_rect' or model.name == 'UKGE_rect':
            for iter_triple in test_triples:
                test_pos_weights = iter_triple[:, 3].astype(np.float64)
                test_pos_score = model.forward(iter_triple)

                # bound needed here for the evaluation
                test_pos_score = torch.clamp(torch.clamp(test_pos_score, max=1), min=0)  # checked
                test_pos_weights = torch.from_numpy(test_pos_weights).view(1, -1).transpose(0, 1).cuda()
                test_pos_score = test_pos_score.view(1, -1).transpose(0, 1)

                if plot:
                    plot_mse_pos(test_pos_score, test_pos_weights, batch_no)
                    batch_no = batch_no + 1

                mse = torch.sum((test_pos_score - test_pos_weights) ** 2)
                mae = torch.sum(torch.abs(test_pos_score - test_pos_weights))
                total_mse_pos = total_mse_pos + mse
                total_mae_pos = total_mae_pos + mae
                if np.asarray(iter_triple).shape[0] < test_batch_size:
                    break

        total_mse_pos = total_mse_pos / positive_test_triples_size
        total_mae_pos = total_mae_pos / positive_test_triples_size
        print('mse pos: {:.4f} - mae pos: {:.4f} '.format(total_mse_pos.item(), total_mae_pos.item()))
    return total_mse_pos, total_mae_pos


def get_mse_neg(model, test_pos, test_batch_size, negsample_num, entitiy_list, filter_triples, plot):
    # model.regul = False
    total_mse_neg = 0
    total_mae_neg = 0

    test_triples = list(get_minibatches(test_pos, test_batch_size, shuffle=True))
    '''
    Negative sampling method corrupts negsample_num times head and negsample_num times tail
    At the end the, negative triples number are doubled
    To solve this, negative sample parameter in the arg divided into 2
    However it is needed here to multiply it by 2 again
    '''
    negative_test_triples_size = test_pos.shape[0] * negsample_num * 2
    batch_no = 0
    with torch.no_grad():
        if model.name == 'UKGE_logi' or model.name == 'WGE_logi' or model.name=='rescal':
            for iter_triple in test_triples:

                iter_neg = sample_negatives_only(iter_triple, negsample_num, entitiy_list, filter_triples) # negsample_num is ok
                test_neg_score = model.forward(iter_neg)

                #if plot:
                    #plot_mse_neg(test_neg_score, batch_no)
                    #batch_no = batch_no + 1

                test_neg_score = test_neg_score.view(negsample_num * 2, -1).transpose(0, 1) # negsample_num is ok

                mse = torch.sum((test_neg_score - 0) ** 2) # here torch.sum adds every negative triple into 1
                mae = torch.sum(torch.abs(test_neg_score - 0))

                total_mse_neg = total_mse_neg + mse
                total_mae_neg = total_mae_neg + mae
                if np.asarray(iter_triple).shape[0] < test_batch_size:
                    break

        if model.name == 'WGE_rect' or model.name == 'UKGE_rect':
            for iter_triple in test_triples:

                iter_neg = sample_negatives_only(iter_triple, negsample_num, entitiy_list, filter_triples)  # negsample_num is ok
                test_neg_score = model.forward(iter_neg)

                # bound - not bounded in the calculate_score part(forward)
                test_neg_score = torch.clamp(torch.clamp(test_neg_score, max=1), min=0)  # checked

                if plot:
                    plot_mse_neg(test_neg_score, batch_no)
                    batch_no = batch_no + 1

                test_neg_score = test_neg_score.view(negsample_num*2, -1).transpose(0, 1)  # negsample_num is ok
                mse = torch.sum((test_neg_score - 0) ** 2)
                mae = torch.sum(torch.abs(test_neg_score - 0))

                total_mse_neg = total_mse_neg + mse
                total_mae_neg = total_mae_neg + mae
                if np.asarray(iter_triple).shape[0] < test_batch_size:
                    break

    total_mse_neg = total_mse_neg / negative_test_triples_size
    total_mae_neg = total_mae_neg / negative_test_triples_size

    print('mse neg: {:.4f} - mae neg: {:.4f} '.format(total_mse_neg.item(), total_mae_neg.item()))
    return total_mse_neg, total_mae_neg


def decision_tree_classify(self, confT, train_pos, test_pos,test_neg):
    """
    :param confT: the threshold of ground truth confidence score
    """
    # train
    # train_X = self.get_score_batch(train_h, train_r, train_t)[:, np.newaxis]  # feature(2D, n*1)
    train_X_p = self.forward(train_pos)
    train_X_p = train_X_p.cpu().numpy().reshape(-1, 1)
    train_Y_p = (train_pos[:, 3].astype(np.float) > confT).astype(int).reshape(-1, 1) # label (high confidence/not)

    #train_X_n = self.forward(train_neg)
    #train_X_n = train_X_n.cpu().numpy().reshape(-1, 1)
    #train_Y_n = (train_Y_p.astype(np.float) > 100000).astype(int)
    #train_Y_n = np.full_like(train_X_n, 0)

    # concat strong and weak
    train_X = np.concatenate((train_X_p), axis=None).reshape(-1, 1)
    train_Y = np.concatenate((train_Y_p), axis=None).reshape(-1, 1)
    # train_X = np.concatenate((train_X_p, train_X_n), axis=None).reshape(-1, 1)
    # train_Y = np.concatenate((train_Y_p, train_Y_n), axis=None).reshape(-1, 1)

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_X, train_Y)

    # test data prep
    test_X_p = self.forward(test_pos)
    test_X_p = test_X_p.cpu().numpy().reshape(-1, 1)
    test_Y_truth_p = (test_pos[:, 3].astype(np.float) > confT).astype(int)

    test_X_n = self.forward(test_neg)
    test_X_n = test_X_n.cpu().numpy().reshape(-1, 1)
    test_Y_truth_n = (test_X_n.astype(np.float) > 100000).astype(int)
    test_Y_truth_n = np.full_like(test_X_n, 0)

    # concat strong and weak
    test_X = np.concatenate((test_X_p,test_X_n), axis=None).reshape(-1, 1)
    test_Y_truth = np.concatenate((test_Y_truth_p,test_Y_truth_n), axis=None).reshape(-1, 1)

    test_Y_pred = clf.predict(test_X)
    print('Number of true positive: %d' % np.sum(test_Y_truth))
    print('Number of predicted positive: %d' % np.sum(test_Y_pred))

    precision, recall, F1, _ = sklearn.metrics.precision_recall_fscore_support(test_Y_truth, test_Y_pred, labels=[1, 0])
    accu = sklearn.metrics.accuracy_score(test_Y_truth, test_Y_pred)

    # P-R curve
    P, R, thres = sklearn.metrics.precision_recall_curve(test_Y_truth, test_X)
    print( 'f1:' , F1)
    print('accuracy:', accu)
    return test_X, precision, recall, F1, accu, P, R


class Evaluation(object):
    def __init__(self, model):
        self.vec_e = []  # embedding vectors
        self.vec_r = []

        return

    class IndexScore:
        """
        The score of a tail when h and r is given.
        It's used in the ranking task to facilitate comparison and sorting.
        Print w as 3 digit precision float.
        """

        def __init__(self, index, score):
            self.index = index
            self.score = score

        def __lt__(self, other):
            return self.score < other.score

        def __repr__(self):
            # return "(index: %d, w:%.3f)" % (self.index, self.score)
            return "(%d, %.3f)" % (self.index, self.score)

        def __str__(self):
            return "(index: %d, w:%.3f)" % (self.index, self.score)

    def vecs_from_triples(self, h, r, t):
        """
        :param h,r,t: int index
        :return: h_vec, r_vec, t_vec
        """
        h, r, t = int(h), int(r), int(t)  # just in case of float
        hvec = self.con_index2vec(h)
        rvec = self.rel_index2vec(r)
        tvec = self.con_index2vec(t)
        return hvec, rvec, tvec

    def con_index2vec(self, e):  # embedding
        return self.vec_e[e]

    def rel_index2vec(self, r):
        return self.vec_r[r]

    # TODO: here
    '''
    def con_str2vec(self, str):
        this_index = self.this_data.con_str2index(str)
        if this_index == None:
            return None
        return self.vec_c[this_index]

    def rel_str2vec(self, str):
        this_index = self.this_data.rel_str2index(str)
        if this_index == None:
            return None
        return self.vec_r[this_index]
    '''

    def get_t_ranks(self, model, h, r, ts):  # send model as parameter to get scores
        """
        Given some t index, return the ranks for each t
        :return:
        """
        with torch.no_grad():
            hs0 = np.asarray(np.full_like(ts, h, dtype=np.int64))
            rs0 = np.asarray(np.full_like(ts, r, dtype=np.int64))
            ts0 = np.asarray(ts).astype(np.int64)

            keys0 = list(map(lambda i, j, k: int(str(i) + str(j) + str(k)), hs0, ts0, rs0))
            ranks0 = []

            # all entities for global ranking:
            entities_all = list(model.kg.id2ent_dict.keys())

            heads_global = np.asarray(np.full_like(entities_all, h, dtype=np.int64))
            rels_global = np.asarray(np.full_like(entities_all, r, dtype=np.int64))
            tails_global = np.asarray(entities_all).astype(np.int64)

            scores_global = model.calculate_score(heads_global, tails_global, rels_global).cpu() # head tail relation !
            keys_global = list(map(lambda i, j, k: int(str(i) + str(j) + str(k)), heads_global, tails_global, rels_global))

            if model.name == 'WGE_logi' or model.name == 'UKGE_logi':
                ranks_global = rankdata(-scores_global, method='ordinal')

            if model.name == 'WGE_rect' or model.name == 'UKGE_rect':
                # bound needed here for the evaluations
                scores_global = torch.clamp(torch.clamp(scores_global, max=1), min=0)
                ranks_global = rankdata(-scores_global, method='ordinal')

            dict_global = dict(zip(keys_global, ranks_global))

            for key in keys0:
                try:
                    rank = dict_global[key]
                except:
                    print('key {} not found'.format(key))
                ranks0.append(rank)

            return np.asarray(ranks0)

    def ndcg(self, model, h, r, tw_truth):
        """
        Compute nDCG(normalized discounted cummulative gain)
        sum(score_ground_truth / log2(rank+1)) / max_possible_dcg
        :param tw_truth: [IndexScore1, IndexScore2, ...], soreted by IndexScore.score descending
        :return:
        """
        # max_possible_dcg is ideal dcg here
        # prediction
        # print("is model at training:", model.training)
        with torch.no_grad():
            ts = [tw.index for tw in tw_truth] # tails list
            ranks = self.get_t_ranks(model, h, r, ts)

            # linear gain
            gains = np.array([tw.score for tw in tw_truth]) # gain = score or weight
            discounts = np.log2(ranks + 1)
            discounted_gains = gains / discounts
            dcg = np.sum(discounted_gains)  # discounted cumulative gain

            # normalize
            ranks_best = rankdata(-gains, method='ordinal')
            max_possible_dcg = np.sum(gains / np.log2(ranks_best + 1))  # when ranks = [1, 2, ...len(truth)]
            ndcg = dcg / max_possible_dcg  # normalized discounted cumulative gain

            # exponential gain
            exp_gains = np.array([2 ** tw.score - 1 for tw in tw_truth])
            exp_discounted_gains = exp_gains / discounts
            exp_dcg = np.sum(exp_discounted_gains)

            # normalize
            ranks_exp_best = rankdata(-exp_gains, method='ordinal')

            exp_max_possible_dcg = np.sum(
                exp_gains / np.log2(ranks_exp_best + 1))  # when ranks = [1, 2, ...len(truth)]
            exp_ndcg = exp_dcg / exp_max_possible_dcg  # normalized discounted cumulative gain

            return ndcg, exp_ndcg

    def mean_ndcg(self, model, hr_map):
        """
        :param hr_map: {h:{r:{t:w}}}
        :return:
        """
        # print("is model at training:", model.training)
        with torch.no_grad():
            ndcg_sum = 0  # nDCG with linear gain
            exp_ndcg_sum = 0  # nDCG with exponential gain
            count = 0

            t0 = time.time()

            # debug ndcg
            res = []  # [(h,r,tw_truth, ndcg)]

            for h in hr_map:
                for r in hr_map[h]:
                    tw_dict = hr_map[h][r]  # {t:w}
                    tw_truth = [self.IndexScore(t, w) for t, w in tw_dict.items()]
                    tw_truth.sort(reverse=True)  # descending on w
                    ndcg, exp_ndcg = self.ndcg(model, h, r, tw_truth)  # nDCG with linear gain and exponential gain
                    ndcg_sum += ndcg
                    exp_ndcg_sum += exp_ndcg
                    count += 1

                    #print('Processed %d, time %s' % (count, (time.time() - t0)))
                    #print('mean ndcg (linear gain) now: %f' % (ndcg_sum / count))
                    #print('mean ndcg (exponential gain) now: %f' % (exp_ndcg_sum / count))

                    ranks = self.get_t_ranks(model, h, r, [tw.index for tw in tw_truth])
                    res.append((h, r, tw_truth, ndcg, ranks))

            print('Duration for calculating mean ndcg {:.1f}'.format(time.time() - t0))
            print('mean ndcg (linear): %f' % (ndcg_sum / count))
            print('mean ndcg (expone): %f' % (exp_ndcg_sum / count))
            return ndcg_sum / count, exp_ndcg_sum / count
