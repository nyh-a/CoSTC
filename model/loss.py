import re
from itertools import product

import more_itertools as mit
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-9

def ce_loss(output, target):
    return F.cross_entropy(output, target)

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)

def bceWlogit_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

# define "soft" cross-entropy with pytorch tensor operations
def soft_ce_loss(input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]

# bsz : batch size (number of positive pairs)
# d   : latent dim
# x   : Tensor, shape=[bsz, d]
#       latents for one side of positive pairs
# y   : Tensor, shape=[bsz, d]
#       latents for the other side of positive pairs

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    

COSINE_DISTANCE = lambda x,y: 1 - F.cosine_similarity(x,y)

def OnlineContrastiveLoss(emb_a, emb_b, labels, margin=0.5, distance_metric=COSINE_DISTANCE):
    """
    Online Contrastive loss. Similar to ConstrativeLoss, but it selects hard positive (positives that are far apart)
    and hard negative pairs (negatives that are close) and computes the loss only for these pairs. Often yields
    better performances than ConstrativeLoss.
    """
    distance_matrix = distance_metric(emb_a, emb_b)
    negs = distance_matrix[labels == 0]
    poss = distance_matrix[labels == 1]

    # select hard positive and hard negative pairs
    negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
    positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

    positive_loss = positive_pairs.pow(2).sum()
    negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
    loss = positive_loss + negative_loss
    return loss

def ContrastiveLoss(emb_a, emb_b, labels, margin=0.5, distance_metric=COSINE_DISTANCE):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.
    """
    size_average = False
    distances = distance_metric(emb_a, emb_b)
    losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(margin - distances).pow(2))
    return losses.mean() if size_average else losses.sum()

def ContrastiveLoss2(emb_a, emb_b, labels, margin=0.5, distance_metric=COSINE_DISTANCE):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.
    """
    distances = distance_metric(emb_a, emb_b)
    losses = labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(margin - distances).pow(2)
    return losses.sum()

def dis_loss(output, target):
    pos = output[target == 1]
    neg = output[target == 0]
    neg = neg + 1e-10
    pos = -pos - 1e-10
    pos = F.logsigmoid(pos)
    neg = F.logsigmoid(neg)
    loss = pos.mean() + neg.mean()
    loss = - loss
    return loss

def nll_loss(output, target):
    return F.nll_loss(output, target)


def square_exp_loss(output, target, beta=1.0):
    """
    output: a (batch_size, 1) tensor, value should be positive
    target: a (batch_size, ) tensor of dtype int
    beta: a float weight of negative samples
    """
    loss = (output[target == 1] ** 2).sum() + beta * torch.exp(-1.0 * output[target == 0]).sum()
    return loss


def weighted_bce_loss(output, target, weight):
    loss = F.binary_cross_entropy_with_logits(output.squeeze(), target.float(), reduction="none") * weight
    return loss.sum() / weight.sum()


def cross_entropy_loss(output, target, beta=1.0):
    loss = F.cross_entropy(output, target.long(), reduction="mean")
    return loss


def kl_div_loss(output, target):
    loss = F.kl_div(output.log_softmax(1), target, reduction="batchmean")
    return loss


def margin_rank_loss(output, target, sample_size=16, margin=1.0):
    label = target.cpu().numpy()
    pos_indices = []
    neg_indices = []
    for cnt, sublabel in enumerate(mit.sliced(label, sample_size)):
        pos, neg = [], []
        for i, l in enumerate(sublabel):
            i += cnt * sample_size
            if l:
                pos.append(i)
            else:
                neg.append(i)
        len_p = len(pos)
        len_n = len(neg)
        pos_indices.extend([i for i in pos for _ in range(len_n)])
        neg_indices.extend(neg * len_p)

    y = -1 * torch.ones(output[pos_indices, :].shape[0]).to(target.device)
    loss = F.margin_ranking_loss(output[pos_indices, :].squeeze(1), output[neg_indices, :].squeeze(1), y, margin=margin, reduction="mean")
    return loss


def info_nce_loss(output, target):
    """
    output: a (batch_size, 1+negative_size) tensor
    target: a (batch_size, ) tensor of dtype long, all zeros
    """
    target = torch.zeros(output.shape[0]).long().to("cuda:0")
    return F.cross_entropy(output, target, reduction="mean")


class DistMarginLoss:
    def __init__(self, spdist):
        self.spdist = torch.FloatTensor(spdist)  # vocab_size x vocab_size
        self.spdist /= self.spdist.max()

    def loss(self, output, target, nodes):
        label = target.cpu().numpy()
        sep_01 = np.array([0, 1], dtype=label.dtype)
        sep_10 = np.array([1, 0], dtype=label.dtype)

        # fast way to find subarray indices in a large array, c.f. https://stackoverflow.com/questions/14890216/return-the-indexes-of-a-sub-array-in-an-array
        sep10_indices = [(m.start() // label.itemsize) + 1 for m in re.finditer(sep_10.tostring(), label.tostring())]
        end_indices = [(m.start() // label.itemsize) + 1 for m in re.finditer(sep_01.tostring(), label.tostring())]
        end_indices.append(len(label))
        start_indices = [0] + end_indices[:-1]

        pair_indices = []
        for start, middle, end in zip(start_indices, sep10_indices, end_indices):
            pair_indices.extend(list(product(range(start, middle), range(middle, end))))
        positive_indices, negative_indices = zip(*pair_indices)
        positive_indices = list(positive_indices)
        negative_indices = list(negative_indices)
        positive_node_ids = [nodes[i] for i in positive_indices]
        negative_node_ids = [nodes[i] for i in negative_indices]
        margins = self.spdist[positive_node_ids, negative_node_ids].to(target.device)
        output = output.view(-1)

        # y = -1 * torch.ones(output[positive_indices,:].shape[0]).to(target.device)
        # loss = F.margin_ranking_loss(output[positive_indices,:], output[negative_indices,:], y, margin=margin, reduction="sum")
        loss = (-output[positive_indices].sigmoid().clamp(min=EPS) + output[negative_indices].sigmoid().clamp(
            min=EPS) + margins.clamp(min=EPS)).clamp(min=0)
        return loss.mean()
