import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from torch.autograd import Variable


def log_loss_adv_with_regularization(self, y_pos, y_neg, reg_pos, reg_neg, lam, temp):
    M = y_pos.size(0)
    N = y_neg.size(0)
    C = int(N / M)
    y_neg = y_neg.view(C, -1).transpose(0, 1)
    p = F.softmax(temp * y_neg)
    loss_pos = torch.sum(F.softplus(-1 * y_pos))
    loss_neg = torch.sum(p * F.softplus(y_neg))
    loss = (loss_pos + loss_neg) / 2 / M
    loss = loss + lam * (reg_pos + reg_neg)
    if self.gpu:
        loss = loss.cuda()
    return loss


def log_loss_adv(self, y_pos, y_neg, temp=0):
    M = y_pos.size(0)
    # print(M)

    N = y_neg.size(0)
    # print(N)
    C = int(N / M)
    y_neg = self.gamma - y_neg
    y_pos = self.gamma - y_pos
    # print(y_pos.size())
    # print(y_neg.size())

    y_neg = y_neg.view(C, -1)
    # print(y_neg.size())
    y_neg = y_neg.transpose(0, 1)
    # print(y_neg.size())
    # exit()
    p = F.softmax(temp * y_neg)
    # print(p)
    # exit()
    loss_pos = torch.sum(F.softplus(-1 * y_pos))
    loss_neg = torch.sum(p * F.softplus(y_neg))
    loss = (loss_pos + loss_neg) / 2 / M

    if self.gpu:
        loss = loss.cuda()
    return loss


def adversarial_loss(self, positive_score, negative_score, margin, negative_sample_number, temperature=0):
    '''
    :param positive_score: model score of the positive triples (1d vector)
    :param negative_score: model score of the negative triples (1d vector)
    :param margin: the gamma used in the model (an integer)
    :param negative_sample_number: how many negatives used per positive (integer)
    :param temperature: temperature variable (float or integer)
    :return: final log adversarial negative loss
    '''

    n_positive_score, n_negative_score = positive_score.size(0), negative_score.size(0)
    # if one want to explicitly find out the negative sampling number
    negative_sample_number = int(n_negative_score / n_positive_score)
    positive_loss_part = torch.sum(F.softplus(-1 * (margin - positive_score)))
    # use the probabilistic weights for negative samples
    negative_score = margin - negative_score
    negative_score = negative_score.view(negative_sample_number, -1)
    negative_score = negative_score.transpose(0, 1)
    # get the temperature multiplied by the probability (alpha*p(h',r,t'))
    probability = F.softmax(temperature * negative_score)
    negative_loss_part = torch.sum(probability * F.softplus(negative_score))
    final_loss = (positive_loss_part + negative_loss_part)
    final_loss = final_loss / 2 / n_positive_score

    return final_loss, positive_loss_part/ n_positive_score, negative_loss_part/ n_positive_score


def log_rank_loss(self, y, C, gamma, neg, temp):
    M = y.size(0)
    y = neg * (gamma - y)
    y = y.view(C, -1).transpose(0, 1)
    p = F.softmax(temp * y)
    loss = torch.sum(p * F.softplus(y))
    loss = loss / M * C
    if self.gpu:
        loss = loss.cuda()
    return loss


def margin_rank_loss(self, y, C, gamma, pos, temp):
    M = y.size(0)
    y = y.view(C, -1).transpose(0, 1)
    p = F.softmax(temp * pos * y)
    loss = torch.sum(p * self.relu(pos * (y - gamma))) / M * C
    if self.gpu:
        loss = loss.cuda()
    return loss


def rank_loss(self, y_pos, y_neg, gamma ,temp):
    M = y_pos.size(0)
    N = y_neg.size(0)
    C = int(N / M)
    y_pos = y_pos.repeat(C)
    y_pos = y_pos.view(C, -1).transpose(0, 1)
    y_neg = y_neg.view(C, -1).transpose(0, 1)
    p = F.softmax(temp * y_neg)
    loss = torch.sum(p * self.relu(y_pos + gamma - y_neg)) / M
    return loss


def double_rank_loss(self, X, y_pos, y_neg, gamma1, gamma2, temp, reg):
    M = y_pos.size(0)
    N = y_neg.size(0)
    C = int(N / M)
    e_i = Variable(torch.from_numpy(X[:, 3].astype(np.int64)).cuda())
    error = self.error(e_i).view(-1, 1)
    y_neg = y_neg.view(C, -1).transpose(0, 1)
    p = F.softmax(temp * y_neg)
    loss_pos = torch.sum(self.relu(y_pos - gamma1))
    loss_neg = torch.sum(p * self.relu(gamma2 - y_neg + error))
    loss = (loss_pos + loss_neg) / M
    loss = loss + reg * torch.mean(error ** 2)
    return loss


def double_rank_loss1(self, X, y_pos, y_neg, gamma1, gamma2, temp, reg):
    M = y_pos.size(0)
    N = y_neg.size(0)
    C = int(N / M)
    e_i = Variable(torch.from_numpy(X[:, 3].astype(np.int64)).cuda())
    error = self.error(e_i).view(-1, 1)
    y_neg = y_neg.view(C, -1).transpose(0, 1)
    p = F.softmax(temp * y_neg)
    loss_pos = torch.sum(self.abs(y_pos - gamma1))
    loss_neg = torch.sum(p * self.relu(gamma2 - y_neg + error))
    loss = (loss_pos + loss_neg) / M
    loss = loss + reg * torch.mean(error ** 2)

    return loss


def normalize_embeddings(self):
    self.emb_E_real.weight.data.renorm_(p=2, dim=0, maxnorm=1)
    self.emb_E_img.weight.data.renorm_(p=2, dim=0, maxnorm=1)


def rule_loss_implication(self, groundings, lam=0.0):
    h_i, t_i, p_i, s_i = groundings[:, 0].astype(np.int64), groundings[:, 1].astype(np.int64), groundings[:,
                                                                                               2].astype(
        np.int64), groundings[:, 3].astype(np.int64)

    out_1,_ = self.calculate_score(h_i, t_i, p_i)
    out_2,_ = self.calculate_score(h_i, t_i, s_i)

    loss = self.relu(out_2 - out_1 + lam)

    return loss


def rule_loss_inverse(self, groundings, lam=0.0):
    h_i, t_i, p_i, s_i = groundings[:, 0].astype(np.int64), groundings[:, 1].astype(np.int64), groundings[:,
                                                                                               2].astype(
        np.int64), groundings[:, 3].astype(np.int64)

    out1,_ = self.calculate_score(h_i, t_i, p_i)
    out2,_ = self.calculate_score(t_i, h_i, s_i)
    loss = self.relu(out2 - out1 + lam)

    return loss


def rule_loss_symmetric(self, groundings, lam):
    h_i, t_i, p_i, s_i = groundings[:, 0].astype(np.int64), groundings[:, 1].astype(np.int64), groundings[:,
                                                                                               2].astype(
        np.int64), groundings[:, 3].astype(np.int64)

    out1,_ = self.calculate_score(h_i, t_i, p_i)
    out2,_ = self.calculate_score(t_i, h_i, s_i)
    loss = self.relu(torch.abs(out2 - out1) + lam)

    return loss


def rule_loss_equivalence(self, groundings, lam=0.0):
    h_i, t_i, p_i, s_i = groundings[:, 0].astype(np.int64), groundings[:, 1].astype(np.int64), groundings[:,
                                                                                               2].astype(
        np.int64), groundings[:, 3].astype(np.int64)

    out1,_ = self.calculate_score(h_i, t_i, p_i)
    out2,_ = self.calculate_score(h_i, t_i, s_i)
    loss = self.relu(out2 - out1 + lam)
    return loss


def rule_loss_composition(self, groundings, lam=0.1):
    h_i, t_i, p_i, q_i, s_i = groundings[:, 0].astype(np.int64), groundings[:, 1].astype(np.int64), groundings[:,
                                                                                                    2].astype(
        np.int64), groundings[:, 3].astype(np.int64), groundings[:, 4].astype(np.int64)

    out1 = self.calculate_score(h_i, t_i, p_i)
    out1 = self.sigmoid(0.5 * out1)
    out2 = self.calculate_score(h_i, t_i, q_i)
    out2 = self.sigmoid(0.5 * out2)
    out3 = self.calculate_score(h_i, t_i, s_i)
    out3 = self.sigmoid(0.5 * out3)
    loss = self.relu(((out1 * out2) - out3) + lam)
    return loss


def rule_loss_implication_logicENN(self, groundings, lam):
    h_i, t_i, p_i, s_i = groundings[:, 0].astype(np.int64), groundings[:, 1].astype(np.int64), groundings[:, 2].astype(np.int64), groundings[:, 3].astype(np.int64)

    if self.gpu:
            h_i = Variable(torch.from_numpy(h_i).cuda())
            t_i = Variable(torch.from_numpy(t_i).cuda())
            p_i = Variable(torch.from_numpy(p_i).cuda())
            s_i = Variable(torch.from_numpy(s_i).cuda())
    else:
            h_i = Variable(torch.from_numpy(h_i))
            t_i = Variable(torch.from_numpy(t_i))
            p_i = Variable(torch.from_numpy(p_i))
            s_i = Variable(torch.from_numpy(s_i))

    h_e = self.emb_E(h_i).view(-1, self.embedding_dim)
    t_e = self.emb_E(t_i).view(-1, self.embedding_dim)
    p_e = self.emb_R(p_i).view(-1, self.embedding_dim)
    s_e = self.emb_R(s_i).view(-1, self.embedding_dim)
    loss = self.relu(torch.sum(p_e - s_e, 1) + lam)

    return loss


def rule_loss_inverse_logicENN(self, groundings, lam):

    h_i, t_i, p_i, s_i = groundings[:, 0].astype(np.int64), groundings[:, 1].astype(np.int64), groundings[:,2].astype(np.int64), groundings[:, 3].astype(np.int64)

    if self.gpu:
            h_i = Variable(torch.from_numpy(h_i).cuda())
            t_i = Variable(torch.from_numpy(t_i).cuda())
            p_i = Variable(torch.from_numpy(p_i).cuda())
            s_i = Variable(torch.from_numpy(s_i).cuda())
    else:
            h_i = Variable(torch.from_numpy(h_i))
            t_i = Variable(torch.from_numpy(t_i))
            p_i = Variable(torch.from_numpy(p_i))
            s_i = Variable(torch.from_numpy(s_i))

    h_e = self.emb_E(h_i).view(-1, self.embedding_dim)
    t_e = self.emb_E(t_i).view(-1, self.embedding_dim)
    p_e = self.emb_R(p_i).view(-1, self.embedding_dim, 1)
    s_e = self.emb_R(s_i).view(-1, self.embedding_dim, 1)
    ht_e = torch.cat((h_e, t_e), 1)
    th_e = torch.cat((t_e, h_e), 1)
    out1 = self.relu(self.fc1(ht_e))
    out1 = self.relu(self.fc2(out1))
    out1 = self.relu(self.fc3(out1))
    out1 = torch.bmm(torch.transpose(p_e, 1, 2), out1.view(-1, self.embedding_dim, 1))
    out1 = out1.view(-1, 1)
    out2 = self.relu(self.fc1(th_e))
    out2 = self.relu(self.fc2(out2))
    out2 = self.relu(self.fc3(out2))
    out2 = torch.bmm(torch.transpose(s_e, 1, 2), out2.view(-1, self.embedding_dim, 1))
    out2 = out2.view(-1, 1)
    loss = self.relu(out1 - out2 + lam)

    return loss


def rule_loss_symmetric_logicENN(self, groundings, lam):
    h_i, t_i, p_i, s_i = groundings[:, 0].astype(np.int64), groundings[:, 1].astype(np.int64), groundings[:,2].astype(np.int64), groundings[:, 3].astype(np.int64)

    if self.gpu:
            h_i = Variable(torch.from_numpy(h_i).cuda())
            t_i = Variable(torch.from_numpy(t_i).cuda())
            p_i = Variable(torch.from_numpy(p_i).cuda())
            s_i = Variable(torch.from_numpy(s_i).cuda())
    else:
            h_i = Variable(torch.from_numpy(h_i))
            t_i = Variable(torch.from_numpy(t_i))
            p_i = Variable(torch.from_numpy(p_i))
            s_i = Variable(torch.from_numpy(s_i))

    h_e = self.emb_E(h_i).view(-1, self.embedding_dim)
    t_e = self.emb_E(t_i).view(-1, self.embedding_dim)
    p_e = self.emb_R(p_i).view(-1, self.embedding_dim, 1)
    s_e = self.emb_R(s_i).view(-1, self.embedding_dim, 1)
    ht_e = torch.cat((h_e, t_e), 1)
    th_e = torch.cat((t_e, h_e), 1)
    out1 = self.relu(self.fc1(ht_e))
    out1 = self.relu(self.fc2(out1))
    out1 = self.relu(self.fc3(out1))
    out1 = torch.bmm(torch.transpose(p_e, 1, 2), out1.view(-1, self.embedding_dim, 1))
    out1 = out1.view(-1, 1)
    out2 = self.relu(self.fc1(th_e))
    out2 = self.relu(self.fc2(out2))
    out2 = self.relu(self.fc3(out2))
    out2 = torch.bmm(torch.transpose(s_e, 1, 2), out2.view(-1, self.embedding_dim, 1))
    out2 = out2.view(-1, 1)
    loss = self.relu(torch.abs(out1 - out2) - lam)

    return loss


def rule_loss_equivalence_logicENN(self, groundings, lam):
    h_i, t_i, p_i, s_i = groundings[:, 0].astype(np.int64), groundings[:, 1].astype(np.int64), groundings[:,2].astype(np.int64), groundings[:, 3].astype(np.int64)

    if self.gpu:
            h_i = Variable(torch.from_numpy(h_i).cuda())
            t_i = Variable(torch.from_numpy(t_i).cuda())
            p_i = Variable(torch.from_numpy(p_i).cuda())
            s_i = Variable(torch.from_numpy(s_i).cuda())
    else:
            h_i = Variable(torch.from_numpy(h_i))
            t_i = Variable(torch.from_numpy(t_i))
            p_i = Variable(torch.from_numpy(p_i))
            s_i = Variable(torch.from_numpy(s_i))

    h_e = self.emb_E(h_i).view(-1, self.embedding_dim)
    t_e = self.emb_E(t_i).view(-1, self.embedding_dim)
    p_e = self.emb_R(p_i).view(-1, self.embedding_dim)
    s_e = self.emb_R(s_i).view(-1, self.embedding_dim)
    loss = self.relu(torch.sum(torch.abs(p_e - s_e), 1) - lam)

    return loss


def rule_loss_complex_logicENN(self, groundings, lam, a):
    h1_i, p1_i, t1_i, h2_i, p2_i, t2_i, h3_i, p3_i, t3_i = groundings[:, 0].astype(np.int64), groundings[:,2].astype(
            np.int64), groundings[:, 1].astype(np.int64), groundings[:, 3].astype(np.int64), groundings[:, 5].astype(
            np.int64), groundings[:, 4].astype(np.int64), groundings[:, 6].astype(np.int64), groundings[:, 8].astype(
            np.int64), groundings[:, 7].astype(np.int64)

    if self.gpu:
            h1_i = Variable(torch.from_numpy(h1_i).cuda())
            t1_i = Variable(torch.from_numpy(t1_i).cuda())
            p1_i = Variable(torch.from_numpy(p1_i).cuda())
            h2_i = Variable(torch.from_numpy(h2_i).cuda())
            t2_i = Variable(torch.from_numpy(t2_i).cuda())
            p2_i = Variable(torch.from_numpy(p2_i).cuda())
            h3_i = Variable(torch.from_numpy(h3_i).cuda())
            t3_i = Variable(torch.from_numpy(t3_i).cuda())
            p3_i = Variable(torch.from_numpy(p3_i).cuda())
    else:
            h_i = Variable(torch.from_numpy(h1_i))
            t_i = Variable(torch.from_numpy(t1_i))
            p_i = Variable(torch.from_numpy(p1_i))
            s_i = Variable(torch.from_numpy(s1_i))

    h1_e = self.emb_E(h1_i).view(-1, self.embedding_dim)
    t1_e = self.emb_E(t1_i).view(-1, self.embedding_dim)
    p1_e = self.emb_R(p1_i).view(-1, self.embedding_dim, 1)
    ht_e = torch.cat((h1_e, t1_e), 1)
    #        th_e=torch.cat((t_e,h_e),1)
    out = self.relu(self.fc1(ht_e))
    out = self.relu(self.fc2(out))
    out = self.relu(self.fc3(out))
    out1 = torch.bmm(torch.transpose(p1_e, 1, 2), out.view(-1, self.embedding_dim, 1))
    out1 = out1.view(-1, 1)
    out1 = self.sigmoid(a * out1)

    h2_e = self.emb_E(h2_i).view(-1, self.embedding_dim)
    t2_e = self.emb_E(t2_i).view(-1, self.embedding_dim)
    p2_e = self.emb_R(p2_i).view(-1, self.embedding_dim, 1)
    ht_e = torch.cat((h2_e, t2_e), 1)
    #        th_e=torch.cat((t_e,h_e),1)
    out = self.relu(self.fc1(ht_e))
    out = self.relu(self.fc2(out))
    out = self.relu(self.fc3(out))
    out2 = torch.bmm(torch.transpose(p2_e, 1, 2), out.view(-1, self.embedding_dim, 1))
    out2 = out2.view(-1, 1)
    out2 = self.sigmoid(a * out2)

    h3_e = self.emb_E(h3_i).view(-1, self.embedding_dim)
    t3_e = self.emb_E(t3_i).view(-1, self.embedding_dim)
    p3_e = self.emb_R(p3_i).view(-1, self.embedding_dim, 1)
    ht_e = torch.cat((h3_e, t3_e), 1)
    #        th_e=torch.cat((t_e,h_e),1)
    out = self.relu(self.fc1(ht_e))
    out = self.relu(self.fc2(out))
    out = self.relu(self.fc3(out))
    out3 = torch.bmm(torch.transpose(p3_e, 1, 2), out.view(-1, self.embedding_dim, 1))
    out3 = out3.view(-1, 1)
    out3 = self.sigmoid(a * out3)

    loss = self.relu(out1 * out2 - out3 + lam)

    return loss


def UKGE_main_loss(self, pos_score, pos_weights, neg_score): # f_prob_h ve f_prob_hn&tn gelir
    if self.name == 'rescal':
        n_pos_score = pos_score.shape[0]
        n_neg_score = neg_score.shape[0]
    else:
        n_pos_score = pos_score.shape[1]
        n_neg_score = neg_score.shape[1]
    n_neg_sample = int(n_neg_score / n_pos_score)

    positive_weight = torch.from_numpy(pos_weights).view(1,-1).transpose(0,1).cuda()
    #MSE
    negative_score = neg_score.view(n_neg_sample, -1).transpose(0, 1)
    positive_score = pos_score.view(1, -1).transpose(0, 1)

    pos_loss = torch.mean((positive_score-positive_weight)**2) # f_score_h
    neg_loss = torch.mean((negative_score-0)**2) # f_score_hn&tn

    #total_loss = (pos_loss + neg_loss)
    #return total_loss
    return pos_loss, neg_loss


def UKGE_psl_loss(self, psl_weights, psl_scores):  # psl_prob gelir
    '''
    Based on original implementation
    p_psl is hand-crafted weight, stated in the paper
    torch.clamp((psl_weight-psl_score),min=0) part is distance to satisfaction
    '''

    p_psl = 0.2
    psl_weight = torch.from_numpy(psl_weights).view(1, -1).transpose(0, 1).cuda()
    psl_score = psl_scores.view(1, -1).transpose(0, 1)
    psl_error_each = torch.clamp((psl_weight-psl_score), min=0)**2
    psl_mse = torch.mean(psl_error_each)
    psl_loss = psl_mse * p_psl
    return psl_loss


def WGE_loss(self, pos_scores, pos_weights, neg_scores, epsilons_left, epsilons_right, lambda_1, lambda_2):

    n_pos_scores = pos_scores.shape[1]
    n_neg_scores = neg_scores.shape[1]
    n_neg_samples = int(n_neg_scores / n_pos_scores)

    positive_weight = torch.from_numpy(pos_weights).view(1, -1).transpose(0, 1).cuda()
    positive_score = pos_scores.view(1, -1).transpose(0, 1)  # dimensions checked

    pos_indices = torch.arange(0,1).cuda()
    neg_indices = torch.arange(1,n_neg_samples+1).cuda()

    epsilons_left_p = torch.index_select(epsilons_left, 1,  pos_indices)
    epsilons_left_n = torch.index_select(epsilons_left, 1, neg_indices)

    epsilons_right_p = torch.index_select(epsilons_right, 1, torch.tensor([0]).cuda())
    epsilons_right_n = torch.index_select(epsilons_right, 1, neg_indices)

    epsilons_left_sqr_p = epsilons_left_p ** 2
    epsilons_right_sqr_p = epsilons_right_p ** 2
    epsilons_all_p = torch.mean(lambda_1*epsilons_left_sqr_p + lambda_2*epsilons_right_sqr_p)  # dimensions checked

    epsilons_left_sqr_n = epsilons_left_n**2
    epsilons_right_sqr_n = epsilons_right_n**2
    epsilons_all_n = torch.mean(lambda_1*epsilons_left_sqr_n + lambda_2*epsilons_right_sqr_n)  # dimensions checked

    penalty_l_positives = torch.mean(self.relu(positive_weight - epsilons_left_sqr_p - positive_score))
    penalty_r_positives = torch.mean(self.relu(positive_score - positive_weight - epsilons_right_sqr_p))

    loss_positives = epsilons_all_p + penalty_l_positives + penalty_r_positives

    # For negatives
    # negative_score = neg_scores.view(n_neg_samples, -1).transpose(0, 1)  # dimensions checked

    # TODO: important change here
    # For negatives
    #negative_score = neg_scores.view(n_neg_samples, -1).transpose(0, 1)  # dimensions checked

    # For negatives
    negative_score = neg_scores.view(-1, n_neg_samples)  # dimensions checked

    penalty_l_negatives = torch.mean(self.relu(-epsilons_left_sqr_n - negative_score))  # dimensions checked
    # (0 - epsilons_left_sqr - negative_score)
    penalty_r_negatives = torch.mean(self.relu(negative_score - epsilons_right_sqr_n))
    # (negative_score - 0 - epsilons_right_sqr)

    loss_negatives = epsilons_all_n + penalty_l_negatives + penalty_r_negatives

    return loss_positives, loss_negatives


def WGE_rule_loss(self, grounding_weights, grounding_scores):

    grounding_weights = torch.from_numpy(grounding_weights).view(1, -1).transpose(0, 1).cuda()
    grounding_scores = grounding_scores.view(1, -1).transpose(0, 1)
    rule_loss = torch.mean(self.relu(grounding_weights - grounding_scores))

    return rule_loss
