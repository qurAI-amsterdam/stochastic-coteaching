import torch
from torch.autograd import Function
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        e = 1e-6
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + e

        t = (2 * self.inter.float() + e) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, (p, t) in enumerate(zip(input, target)):
        one_hot = torch.zeros(p.shape).scatter_(0, t.unsqueeze(0), 1)
        s = s + DiceCoeff().forward(p, one_hot)

    return (s / (i + 1)).squeeze()

def soft_dice_score(prob_c, one_hot):
    """

    Computing the soft-dice-loss for a SPECIFIC class according to:

    DICE_c = \frac{\sum_{i=1}^{N} (R_c(i) * A_c(i) ) }{ \sum_{i=1}^{N} (R_c(i) +   \sum_{i=1}^{N} A_c(i)  }

    Input: (1) probs: 4-dim tensor [batch_size, num_of_classes, width, height]
               contains the probabilities for each class
           (2) true_binary_labels, 4-dim tensor with the same dimensionalities as probs, but contains binary
           labels for a specific class

           Remember that classes 0-3 belongs to ES images (phase) and 4-7 to ED images

    """
    eps = 1.0e-6
    nominator = torch.dot(one_hot.view(-1), prob_c.view(-1)) # the other way around
    denominator = torch.sum(one_hot) + torch.sum(prob_c) + eps
    return - 2 * nominator/denominator



class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes


    def forward(self, input, target):
        # if self.n_classes > 2:
        one_hot = torch.zeros(input.shape).scatter_(1, target.unsqueeze(1), 1)
        # else:
        #     one_hot = target.unsqueeze(1).float()
        # print(one_hot)
        # print(input.shape)
        return soft_dice_score(input, one_hot)


class StochasticCoTeachingSegmentationLoss:
    def __init__(self, max_iters, alpha=32, beta=2, delay=10, tp_gradual=10, exponent=1, Loss=nn.CrossEntropyLoss):
        maxval = 1
        rate_schedule = np.ones(max_iters) * maxval
        rate_schedule[:delay] = 0
        rate_schedule[delay: delay + tp_gradual] = np.linspace(0, 1, tp_gradual, endpoint=False)
        self.tp_gradual = tp_gradual
        self.rate_schedule = rate_schedule
        self._it = 0
        self.beta = beta
        self.alpha = alpha
        self._axis = 1
        self.Loss = Loss(reduction='none')
        self.rng = np.random.Generator(np.random.PCG64(808))
        self.mask_1 = torch.ones((1,1,1))
        self.mask_2 = torch.ones((1, 1, 1))

    def step(self):
        self._it += 1

    def get_probas(self, logits, y):
        '''
        Get probabilities from logits of classes indicated by y.
        '''
        all_probabilities = F.softmax(logits, self._axis)
        class_probabilities = torch.gather(all_probabilities, self._axis, y.unsqueeze(self._axis))
        return class_probabilities

    def mask_probas(self, probabilities):
        '''
        generating thresholds from a beta pdf takes time. Tiling mitigates time at the cost of less randomness.
        '''

        maxval = self.rate_schedule[self._it]
        shape = probabilities.shape
        samples, channels = shape[:2]
        dist_shape = (samples, channels, 16, 16)


        mask = torch.ones((1,), dtype=torch.float32, device=probabilities.device).expand(tuple(shape))
        if maxval != 0:
            stochastic_thresholds = torch.from_numpy(
                maxval * self.rng.beta(a=self.alpha, b=self.beta, size=dist_shape).astype(np.float32)).cuda()
            stochastic_thresholds = stochastic_thresholds.repeat((1,1,8,8))  # expand does not copy data
            mask = torch.gt(probabilities, stochastic_thresholds).type(torch.cuda.FloatTensor)
        return mask

    @torch.no_grad()
    def current_fraction_rejected(self):
        return self.fraction_reject_1, self.fraction_reject_2

    @torch.no_grad()
    def current_probas(self):
        return self.probas1, self.probas2

    def __call__(self, logits_1, logits_2, y):
        with torch.no_grad():
            self.probas1 = self.get_probas(logits_1, y)
            self.probas2 = self.get_probas(logits_2, y)

            mask_1 = self.mask_probas(self.probas1)
            mask_2 = self.mask_probas(self.probas2)

            self.fraction_reject_1 = (1. - (mask_1.sum() / torch.prod(torch.tensor(mask_1.shape)))).item()
            self.fraction_reject_2 = (1. - (mask_2.sum() / torch.prod(torch.tensor(mask_2.shape)))).item()


        loss_1_update = torch.sum(self.Loss(logits_1, y) * mask_2) / mask_2.sum()
        loss_2_update = torch.sum(self.Loss(logits_2, y) * mask_1) / mask_1.sum()

        return loss_1_update, loss_2_update
