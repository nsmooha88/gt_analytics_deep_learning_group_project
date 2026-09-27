import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None
    #############################################################################
    # TODO: reweight each class by effective numbers                            #
    #############################################################################
    C = len(cls_num_list)
    
    cls_num_list = torch.Tensor(cls_num_list)

    per_cls_weights = ((1 - beta) / (1 - torch.pow(beta, cls_num_list)))

    per_cls_weights = (per_cls_weights / torch.sum(per_cls_weights)) * C
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        #############################################################################
        # TODO: Implement forward pass of the focal loss                            #
        #############################################################################
        # The implementation of calculating focal loss was adapted from the following resource:
        # Class-Balanced Loss Based on Effective Number of Samples (https://arxiv.org/abs/1901.05555)
        
        N = target.size(dim=0)
        C = input.size(dim=1)

        y_encode = F.one_hot(target, num_classes=C)
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        y_encode = y_encode.to(device)

        weights = y_encode * self.weight
        weights = weights.sum(dim=-1)

        softmax = nn.Softmax(dim=1)
        pt = y_encode * softmax(input)
        pt = pt.sum(dim=-1)

        focal_loss = torch.pow((1 - pt), self.gamma)

        log_softmax = nn.LogSoftmax(dim=1)
        log_pt = y_encode * log_softmax(input)
        log_pt = log_pt.sum(dim=-1)

        loss_raw = -1 * weights * focal_loss * log_pt
        loss = loss_raw.sum() / weights.sum()
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss
