import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, MLP
from torch.nn import Module, Parameter
import torch.nn.functional as F
from scipy.sparse import csr_matrix
import scipy


class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt
        self.rate = opt.in_rate
        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.hop = opt.n_iter
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        # Aggregator
        self.local_agg = []
        for i in range(self.hop):
            agg_local = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
            self.add_module('agg_local_{}'.format(i), agg_local)
            self.local_agg.append(agg_local)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.mlp = MLP(opt)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.gru_piece = nn.GRUCell(self.dim, self.dim).cuda()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(select, b.transpose(1, 0))
        return scores, select

    def gene_sess(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        return select

    def neg_sample(self, hidden):
        bs = self.embedding.weight[1:]
        scores = torch.matmul(hidden, bs.transpose(1, 0))
        scores = torch.softmax(scores, 0)
        values, postion = scores.topk(200, dim=0, largest=True, sorted=True)
        # print(postion.shape) #k*num_node
        negs = torch.cuda.FloatTensor(10, self.batch_size, self.dim).fill_(0)
        random_slices = torch.randint(10, 200, (10,))

        for i in torch.arange(10):
            negs[i] = bs[postion[random_slices[i]]]
        return negs

    def forward(self, inputs, adj, mask_item, item, data, hg_adj):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        h_local_all = []
        for i in range(self.hop):
            local_agg = self.local_agg[i]
            if i == 0:
                h_local = local_agg(h, adj, mask_item)
                h_local_all.append(h_local)
            else:
                h_local_next = local_agg(h_local_all[i - 1], adj, mask_item)
                h_local_all.append(h_local_next)

        item_neighbors = [inputs]
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num  # max_len * sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))


        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)

        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)

        anchor = self.mlp(sum_item_emb)


        h_local_res = h_local_all[0]

        h_local = F.dropout(h_local_res, self.dropout_local, training=self.training)

        output = h_local

        return output, anchor


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def SSL(sess_emb_hgnn, sess_emb_lgcn):
    def row_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        return corrupted_embedding

    def row_column_shuffle(embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
        return corrupted_embedding

    def score(x1, x2):
        return torch.sum(torch.mul(x1, x2), 1)

    pos = score(sess_emb_hgnn, sess_emb_lgcn)
    neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
    one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
    # one = zeros = torch.ones(neg1.shape[0])
    con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
    return con_loss


def TripletLoss(anchor, pos, neg):
    p = F.pairwise_distance(anchor, pos)
    n = []
    for neg_one in neg:
        if neg_one.shape != anchor.shape:
            return 0
        one_n = F.pairwise_distance(anchor, neg_one)
        n.append(one_n)
    trip_loss = 0
    margin_loss = torch.nn.MarginRankingLoss(margin=0.5, reduce=False)
    margin_label = -1 * torch.ones_like(p)
    for n_one in n:
        trip_loss += (margin_loss(p, n_one, margin_label)).mean()
    return trip_loss


def addnoise(seq_hidden, epoch):
    x = torch.rand(seq_hidden.shape).cuda()
    k = math.log(epoch + 2)
    seq_hidden += torch.mul(torch.sign(seq_hidden), torch.nn.functional.normalize(x, p=2, dim=1)) * 0.1 * k
    return seq_hidden


def forward(model, data, epoch):
    alias_inputs, adj, items, mask, targets, inputs, hg_adj = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    hg_adj = trans_to_cuda(hg_adj).long()


    hidden, anchor = model(items, adj, mask, inputs, data, hg_adj)
    get = lambda index: hidden[index][alias_inputs[index]]

    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])

    ses = model.gene_sess(seq_hidden, mask)
    noise = addnoise(ses, epoch)

    scores, select = model.compute_scores(seq_hidden, mask)
    negs = model.neg_sample(select)
    triploss = TripletLoss(anchor, select, negs)
    ssl_loss = SSL(ses, noise)

    return targets, scores, -ssl_loss, triploss


def train_test(model, train_data, test_data, sslrate):
    print('start training: ', datetime.datetime.now())
    # torch.autograd.set_detect_anomaly(True)
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=False, pin_memory=False)
    # slices = train_data.generate_batch(model.batch_size)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores, ssl_loss, triploss = forward(model, data, model.Eiters)
        model.Eiters += 1
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss = loss + triploss * 0.001 + ssl_loss * 0.0001

        loss.backward()
        model.optimizer.step()
        total_loss += loss

    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()
    print('start predicting: ', datetime.datetime.now())
    model.eval()


    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=True, pin_memory=False)

    result_20 = []

    hit_20, mrr_20 = [], []
    for data in test_loader:
        targets, scores, con_loss, triploss = forward(model, data, model.Eiters)
        sub_scores_20 = scores.topk(20)[1]
        sub_scores_20 = trans_to_cpu(sub_scores_20).detach().numpy()

        targets = targets.numpy()
        for score, target, mask in zip(sub_scores_20, targets, test_data.mask):
            hit_20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_20.append(0)
            else:
                mrr_20.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result_20.append(np.mean(hit_20) * 100)
    result_20.append(np.mean(mrr_20) * 100)

    return result_20
