import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def euclidean_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.pow(2).sum(2).t()
    return score    


class TripletLoss(nn.Module):
    """
    triplet ranking loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False, cost_style='sum', direction='all'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cost_style = cost_style
        self.direction = direction
        if measure == 'order':
            self.sim = order_sim
        elif measure == 'euclidean':
            self.sim = euclidean_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation
        self.l2_loss = torch.nn.MSELoss()


    def cal_text2videotext_triplet_loss(self,loss_matrix):

        diagonal = loss_matrix.diag().view(loss_matrix.size(0), 1)
        d1 = diagonal.t().expand_as(loss_matrix)

        mask = torch.eye(loss_matrix.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        cost_im = (self.margin - 0.1 + loss_matrix - d1).clamp(min=0)
        cost_im = cost_im.masked_fill_(I, 0)
        return cost_im.sum()

    def cal_video2videotext_triplet_loss(self,loss_matrix):

        diagonal = loss_matrix.diag().view(loss_matrix.size(0), 1)
        d1 = diagonal.t().expand_as(loss_matrix)

        mask = torch.eye(loss_matrix.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        cost_im = (self.margin - 0.1 + loss_matrix - d1).clamp(min=0)
        cost_im = cost_im.masked_fill_(I, 0)
        return cost_im.sum()

    def forward(self, s, im, atten_score, atten_score_gt, loss_matrix, loss_matrix2):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s = None
        cost_im = None
        # compare every diagonal score to scores in its column
        if self.direction in  ['i2t', 'all']:
            # caption retrieval
            cost_s = (self.margin + scores - d1).clamp(min=0)
            cost_s = cost_s.masked_fill_(I, 0)
        # compare every diagonal score to scores in its row
        if self.direction in ['t2i', 'all']:
            # image retrieval
            cost_im = (self.margin + scores - d2).clamp(min=0)
            cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            if cost_s is not None:
                cost_s = cost_s.max(1)[0]
            if cost_im is not None:
                cost_im = cost_im.max(0)[0]

        if cost_s is None:
            cost_s = Variable(torch.zeros(1)).cuda()
        if cost_im is None:
            cost_im = Variable(torch.zeros(1)).cuda()

        if torch.cuda.is_available():
            atten_score_gt = atten_score_gt.cuda()

        attention_loss = self.l2_loss(atten_score, atten_score_gt)

        text2videotext_triplet_loss = self.cal_text2videotext_triplet_loss(loss_matrix)
        video2videotext_triplet_loss = self.cal_video2videotext_triplet_loss(loss_matrix2)


        if self.cost_style == 'sum':
            return cost_s.sum() + cost_im.sum() + attention_loss * 100 + text2videotext_triplet_loss * 0.3 + video2videotext_triplet_loss * 0.3

        else:
            return cost_s.mean() + cost_im.mean()