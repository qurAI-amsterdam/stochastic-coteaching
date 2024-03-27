
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CoTeachingLoss:
    def __init__(self, forget_rate, max_iters, tp_gradual, exponent=1, Loss=nn.CrossEntropyLoss):
        rate_schedule = np.ones(max_iters) * forget_rate
        rate_schedule[:tp_gradual] = np.linspace(0, forget_rate ** exponent, tp_gradual)
        self.tp_gradual = tp_gradual
        self.rate_schedule = rate_schedule
        self._it = 0
        self.Loss = Loss(reduction='none')

    def step(self):
        self._it += 1

    def get_probas(self, logits, y):
        probabilities = F.softmax(logits, 1)
        samples, classes = logits.shape
        raveled_indices = y + torch.arange(0, (samples) * classes, classes).cuda()
        return probabilities.take(raveled_indices)

    def mask_losses(self, losses):
        forget_rate = self.rate_schedule[self._it]

        if forget_rate == 0:
            return torch.ones_like(losses)
        elif forget_rate >= 1:
            raise RuntimeError('Forget-rate is equal to or larger than 1')
        mask = torch.zeros_like(losses)
        indices = torch.argsort(losses.view(-1))
        num_to_use = int(len(indices) * (1 - forget_rate))
        take_first_n_indices = indices[:num_to_use]

        mask.view(-1)[take_first_n_indices] = 1
        return mask

    def current_fraction_rejected(self):
        return self.fraction_reject_1, self.fraction_reject_2

    def current_probas(self):
        return self.probas1, self.probas2

    def __call__(self, logits_1, logits_2, y, *args, **kwargs):
        with torch.no_grad():
            loss_1 = self.Loss(logits_1, y)
            loss_2 = self.Loss(logits_2, y)

            self.current_losses = (loss_1, loss_2) # temp\

            mask_1 = self.mask_losses(loss_1)
            mask_2 = self.mask_losses(loss_2)

            self.probas1 = self.get_probas(logits_1, y)
            self.probas2 = self.get_probas(logits_2, y)

            self.fraction_reject_1 = 1. - (mask_1.sum() / len(mask_1))
            self.fraction_reject_2 = 1. - (mask_2.sum() / len(mask_2))

        loss_1_update = torch.sum(self.Loss(logits_1, y) * mask_2) / mask_2.sum()
        loss_2_update = torch.sum(self.Loss(logits_2, y) * mask_1) / mask_1.sum()

        return loss_1_update, loss_2_update


class StochasticCoTeachingLoss_old:
    def __init__(self, mean, std, max_iters, tp_gradual, exponent=1, Loss=nn.CrossEntropyLoss):
        rate_schedule = np.ones(max_iters) * mean
        rate_schedule[:tp_gradual] = np.linspace(0, mean ** exponent, tp_gradual)
        self.tp_gradual = tp_gradual
        self.rate_schedule = rate_schedule
        self._it = 0
        self.std = std
        self.Loss = Loss(reduction='none')

    def step(self):
        self._it += 1

    def get_probas(self, logits, y):
        probabilities = F.softmax(logits, 1)
        samples, classes = logits.shape
        raveled_indices = y + torch.arange(0, (samples) * classes, classes).cuda()
        return probabilities.take(raveled_indices)

    def mask_probas(self, probas):
        mean = self.rate_schedule[self._it]
        rand = torch.from_numpy(np.random.normal(mean, self.std, len(probas)).astype(np.float32)).cuda()
        # rand = torch.where(rand < mean, rand, torch.ones_like(probas) * mean)
        rand = rand.clamp(0, mean)
        mask = torch.gt(probas, rand).type(torch.cuda.FloatTensor)
        return mask

    @torch.no_grad()
    def current_fraction_rejected(self):
        return self.fraction_reject_1, self.fraction_reject_2

    @torch.no_grad()
    def current_probas(self):
        return self.probas1, self.probas2

    def __call__(self, logits_1, logits_2, y, *args, **kwargs):
        with torch.no_grad():
            self.probas1 = self.get_probas(logits_1, y)
            self.probas2 = self.get_probas(logits_2, y)
            mask_1 = self.mask_probas(self.probas1)
            mask_2 = self.mask_probas(self.probas2)

            self.fraction_reject_1 = 1. - (mask_1.sum() / len(mask_1))
            self.fraction_reject_2 = 1. - (mask_2.sum() / len(mask_2))

        loss_1_update = torch.sum(self.Loss(logits_1, y) * mask_2) / mask_2.sum()
        loss_2_update = torch.sum(self.Loss(logits_2, y) * mask_1) / mask_1.sum()


        return loss_1_update, loss_2_update

class StochasticCoTeachingLoss:
    def __init__(self, alpha, beta, max_iters, tp_gradual, delay=0, exponent=1, clip=(0.01, 0.99), Loss=nn.CrossEntropyLoss):
        maxval = 1
        rate_schedule = np.ones(max_iters) * maxval
        # rate_schedule[:tp_gradual] = np.linspace(0, maxval ** exponent, tp_gradual)
        rate_schedule[:delay] = 0
        rate_schedule[delay:delay+tp_gradual] = np.linspace(0, maxval ** exponent, tp_gradual)
        self.tp_gradual = tp_gradual
        self.rate_schedule = rate_schedule
        self._it = 0
        self.beta = beta
        self.alpha = alpha
        self.Loss = Loss(reduction='none')
        self.rng = np.random.Generator(np.random.PCG64(808))
        self._axis = 1
        self.clip = clip
        # self.rng = np.random.RandomState(808)


    def step(self):
        self._it += 1

    def get_probas(self, logits, y):
        '''
        Get probabilities from logits of classes indicated by y.
        '''
        all_probabilities = F.softmax(logits, self._axis)
        class_probabilities = torch.gather(all_probabilities, self._axis, y.unsqueeze(self._axis))
        return class_probabilities.squeeze()

    def mask_probas(self, probas):
        maxval = self.rate_schedule[self._it]

        mask = torch.ones_like(probas)
        if maxval == 0:
            return mask

        rand = torch.from_numpy(maxval * self.rng.beta(a=self.alpha, b=self.beta, size=probas.shape).astype(np.float32)).cuda()
        rand = torch.clamp(rand, *self.clip)
        mask = torch.gt(probas, rand).type(torch.cuda.FloatTensor)
        # mask.view(-1)[torch.argmax(probas)] = 1. # always have at least one true
        # if not mask.any():
        #     mask[0] = True
        return mask

    @torch.no_grad()
    def current_fraction_rejected(self):
        return self.fraction_reject_1, self.fraction_reject_2

    @torch.no_grad()
    def current_probas(self):
        return self.probas1, self.probas2

    def __call__(self, logits_1, logits_2, y, *args, **kwargs):
        with torch.no_grad():
            self.probas1 = self.get_probas(logits_1, y)
            self.probas2 = self.get_probas(logits_2, y)

            loop = 0
            while True:
                mask_1 = self.mask_probas(self.probas1)
                mask_2 = self.mask_probas(self.probas2)
                if (mask_1.sum() / np.product(mask_1.shape)) >= 0.1 and\
                        (mask_2.sum() / np.product(mask_2.shape)) >= 0.1:
                    break
                if loop == 5:
                    raise RuntimeError('More than 90 percent consitently rejected')
                # print(f'masks in loop {loop} rejected')
                loop += 1

            self.fraction_reject_1 = 1. - (mask_1.sum() / len(mask_1))
            self.fraction_reject_2 = 1. - (mask_2.sum() / len(mask_2))

        e = 1e-6
        loss_1_update = torch.sum(self.Loss(logits_1, y) * mask_2) / mask_2.sum()
        loss_2_update = torch.sum(self.Loss(logits_2, y) * mask_1) / mask_1.sum()

        # loss_1_update = torch.sum(self.Loss(logits_1, y) * mask_2) / (mask_2.sum() + e)
        # loss_2_update = torch.sum(self.Loss(logits_2, y) * mask_1) / (mask_1.sum() + e)

        return loss_1_update, loss_2_update

class StochasticCoTeachingSegmentationLoss:
    def __init__(self, alpha, beta, max_iters, tp_gradual, exponent=1, Loss=nn.CrossEntropyLoss):
        maxval = 1
        rate_schedule = np.ones(max_iters) * maxval
        rate_schedule[:tp_gradual] = np.linspace(0, maxval ** exponent, tp_gradual)
        self.tp_gradual = tp_gradual
        self.rate_schedule = rate_schedule
        self._it = 0
        self.beta = beta
        self.alpha = alpha
        self._axis = 1
        self.Loss = Loss(reduction='none')
        self.rng = np.random.Generator(np.random.PCG64(808))

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

        if maxval == 0:
            mask = 1  # torch.ones(shape, dtype=torch.float32, device=probabilities.device)
            return mask

        stochastic_thresholds = torch.from_numpy(
            maxval * self.rng.beta(a=self.alpha, b=self.beta, size=(samples, channels)).astype(np.float32)).cuda()
        stochastic_thresholds = stochastic_thresholds[:, :, None, None]
        stochastic_thresholds = stochastic_thresholds.expand(tuple(shape))  # expand does not copy data
        mask = torch.gt(probabilities, stochastic_thresholds).type(torch.cuda.FloatTensor)
        if not mask.any():
            mask[0] = 1
        return mask

    @torch.no_grad()
    def current_fraction_rejected(self):
        return self.fraction_reject_1, self.fraction_reject_2

    @torch.no_grad()
    def current_probas(self):
        return self.probas1, self.probas2

    def __call__(self, logits_1, logits_2, y, *args, **kwargs):
        with torch.no_grad():
            self.probas1 = self.get_probas(logits_1, y)
            self.probas2 = self.get_probas(logits_2, y)

            mask_1 = self.mask_probas(self.probas1)
            mask_2 = self.mask_probas(self.probas2)

            self.fraction_reject_1 = 1. - (mask_1.sum() / len(mask_1))
            self.fraction_reject_2 = 1. - (mask_2.sum() / len(mask_2))

        loss_1_update = torch.sum(self.Loss(logits_1, y) * mask_2) / mask_2.sum()
        loss_2_update = torch.sum(self.Loss(logits_2, y) * mask_1) / mask_1.sum()

        return loss_1_update, loss_2_update


from scipy import special
from collections import deque

def mu(alpha, beta):
    return alpha / (alpha + beta)


def var(alpha, beta, mu_):
    return mu_ * (1. - mu_) / (alpha + beta + 1)


def phi(mu_, var_):
    return (mu_ * (1. - mu_) / var_) - 1.


def beta_llh(X, alpha, beta, e=1e-6):
    X = X.clip(e, 1. - e)
    num_samples = len(X)
    term1 = num_samples * (special.gammaln(alpha + beta) - special.gammaln(alpha) - special.gammaln(beta))
    print(term1)
    term2 = (alpha - 1) * sum(np.log(X))
    print(term2)
    term3 = (beta - 1) * sum(np.log(1 - X))
    return term1 + term2 + term3


class BetaDistribution:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, X):
        return self._pdf(X, self.alpha, self.beta)

    def _pdf(self, x, a, b):
        return np.exp(self._logpdf(x, a, b))

    def _logpdf(self, x, a, b):
        lPx = special.xlog1py(b - 1.0, -x) + special.xlogy(a - 1.0, x)
        lPx -= special.betaln(a, b)
        return lPx

    def set_parameters(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta


class EMBeta:
    def __init__(self, numofcomponents, rs=np.random.RandomState(808)):
        params = rs.randint(1, 64, (numofcomponents, 2))
        self.components = [BetaDistribution(*p) for p in params]

    @classmethod
    def calc_phi(cls, mu_, var_):
        return (mu_ * (1. - mu_) / var_) - 1.

    @classmethod
    def calc_alpha_beta(cls, mu, var):
        phi = cls.calc_phi(mu, var)
        alpha = mu * phi
        beta = (1 - mu) * phi

        return alpha, beta

    def sort_components(self):
        alphas = np.array([component.alpha for component in self.components])
        betas = np.array([component.beta for component in self.components])
        sorted_idcs = np.lexsort((-betas, alphas), 0)

        self.components = [self.components[idx] for idx in sorted_idcs]
        self.components[0].alpha = np.clip(self.components[0].alpha, 1, 1000)
        self.components[0].beta = np.clip(self.components[0].beta, 1, 1000)
        self.components[1].alpha = np.clip(self.components[1].alpha, 1, 1000)
        self.components[1].beta = np.clip(self.components[1].beta, 1, 1000)

    def expectation(self, X, e=1e-6):
        X[X <= 0] = e
        X[X >= 1] = 1.0 - e
        W = np.empty((len(self.components), len(X)))
        for idx in range(len(self.components)):
            W[idx] = self.components[idx](X)
        W /= (W.sum(0) + e)
        return W

    def maximization(self, W, X):
        #         W = W * self.component_weights[:, None]
        mu = (W * X).sum(1) / W.sum(1)
        var = (W * (X - mu[:, None]) ** 2).sum(1) / W.sum(1)
        alphas, betas = self.calc_alpha_beta(mu, var)
        for idx, (alpha, beta) in enumerate(zip(alphas, betas)):
            self.components[idx].set_parameters(alpha, beta)

    def emstep(self, X):
        W = self.expectation(X)
        self.maximization(W, X)

    def optimize(self, X, t=1e-6, maxiters=100):
        params = list()
        params.append(self.params())
        old_params = self.params()
        for idx in range(maxiters):
            self.emstep(X)
            params.append(self.params())
            new_params = self.params()
            k_q = np.abs(old_params - new_params) / np.max((old_params, new_params), 0)
            if (k_q < t).all():
                break
            old_params = new_params
        self.sort_components()

    def params(self):
        return np.array([(c.alpha, c.beta) for c in self.components])

    def __str__(self):
        params = list()
        for c in self.components:
            params.append(f'{c.alpha:0.2f}, {c.beta:0.2f}')
        return ' - '.join(params)

class EMBetaCoTeachingLoss:
    def __init__(self, store_n_arrays=128, delay=10, Loss=nn.CrossEntropyLoss):
        self.n_arrays = store_n_arrays
        #         self.update_every = update_every_n_batches

        self.que = deque(maxlen=store_n_arrays)
        self.emb = EMBeta(2)
        self.th = 0.0
        self._it = 0
        self.Loss = Loss(reduction='none')
        self.delay = delay
        self.rng = np.random.Generator(np.random.PCG64(808))

    @torch.no_grad()
    def add(self, probas):
        self.que.append(probas.cpu().numpy())

    def fit(self):
        if self._it >= self.delay:
            probas = np.concatenate(self.que)
            self.emb.optimize(probas)

            x = np.linspace(0.001, 0.999, 100)
            expectations = self.emb.expectation(x)
            y = expectations[0] - expectations[1]
            th = x[np.diff(np.sign(y)).argmin()]

            print('*'*80)
            print('\n')
            print(th)
            print(self.emb.params())
            self.th = th

    def step(self):
        self._it += 1

    def get_probas(self, logits, y):
        probabilities = F.softmax(logits, 1)
        samples, classes = logits.shape
        raveled_indices = y + torch.arange(0, samples * classes, classes).cuda()
        return probabilities.take(raveled_indices)

    @torch.no_grad()
    def mask_probas(self, probas):
        #         maxval = self.rate_schedule[self._it]
        #         mask = torch.ones_like(losses)
        #         if maxval == 0:
        #             return mask
        #         labels = self.gm.predict(losses)

        mask = torch.gt(probas, self.th).type(torch.cuda.FloatTensor)
        return mask

    @torch.no_grad()
    def current_fraction_rejected(self):
        return self.fraction_reject_1, self.fraction_reject_2

    @torch.no_grad()
    def current_probas(self):
        return self.probas1, self.probas2

    def __call__(self, logits_1, logits_2, y, *args, **kwargs):
        with torch.no_grad():
            self.probas1 = self.get_probas(logits_1, y)
            self.probas2 = self.get_probas(logits_2, y)

            self.add(self.probas1)
            self.add(self.probas2)

            mask_1 = self.mask_probas(self.probas1)
            mask_2 = self.mask_probas(self.probas2)

            self.fraction_reject_1 = 1. - (mask_1.sum() / len(mask_1))
            self.fraction_reject_2 = 1. - (mask_2.sum() / len(mask_2))

        loss_1_update = torch.sum(self.Loss(logits_1, y) * mask_2) / mask_2.sum()
        loss_2_update = torch.sum(self.Loss(logits_2, y) * mask_1) / mask_1.sum()

        return loss_1_update, loss_2_update


from sklearn.mixture import GaussianMixture


class CustomGM(GaussianMixture):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def sort(self):
        sorted_idcs = np.argsort(self.means_, axis=0).ravel()
        self.means_ = self.means_[sorted_idcs]
        self.weights_ = self.weights_[sorted_idcs]
        self.covariances_ = self.covariances_[sorted_idcs]
        self.precisions_ = self.precisions_[sorted_idcs]
        self.precisions_cholesky_ = self.precisions_cholesky_[sorted_idcs]

from scipy import stats
class EMCoTeachingLoss:
    def __init__(self, store_n_arrays, delay=1, Loss=nn.CrossEntropyLoss):
        self.delay = delay
        self.n_arrays = store_n_arrays
        #         self.update_every = update_every_n_batches

        self.que = deque(maxlen=store_n_arrays)
        self.gm = CustomGM(2)

        self._it = 0
        self.Loss = Loss(reduction='none')
        self.rng = np.random.Generator(np.random.PCG64(808))

    @torch.no_grad()
    def add(self, batch):
        self.que.append(batch.cpu().numpy())

    def fit(self):
        # if self._it >= self.delay:
        samples = np.concatenate(self.que)
        self.gm.fit(samples.reshape(-1, 1))
        self.gm.sort()
        print('\n')
        print('*'*80)
        print(self.gm.means_, self.gm.covariances_)

    def step(self):
        self._it += 1

    @torch.no_grad()
    def mask_losses(self, losses):
        if self._it < self.delay:
            self.th = 0
            mask = torch.ones_like(losses)
        else:


        # maxval = self.rate_schedule[self._it]
        #
        # if maxval == 0:
        #     mask = torch.ones_like(losses)
        #     return mask
        #         labels = self.gm.predict(losses)

        # first mean from sorted components. This indicates lowest loss.
            mean = self.gm.means_[0, 0]
            var = self.gm.covariances_[0, 0]
            # y = stats.norm.pdf(x, loc=mean, scale=var)
            th = torch.from_numpy(self.rng.normal(mean, var, size=len(losses)).clip(mean, 100).astype(np.float32)).cuda()
            self.th = th.mean().item()
            mask = torch.le(losses, th).type(torch.cuda.FloatTensor)
        return mask

    @torch.no_grad()
    def current_fraction_rejected(self):
        return self.fraction_reject_1, self.fraction_reject_2

    @torch.no_grad()
    def current_probas(self):
        return self.probas1, self.probas2

    def get_probas(self, logits, y):
        probabilities = F.softmax(logits, 1)
        samples, classes = logits.shape
        raveled_indices = y + torch.arange(0, (samples) * classes, classes).cuda()
        return probabilities.take(raveled_indices)

    def __call__(self, logits_1, logits_2, y, *args, **kwargs):
        with torch.no_grad():
            loss_1 = self.Loss(logits_1, y)
            loss_2 = self.Loss(logits_2, y)

            self.add(loss_1)
            self.add(loss_2)

            mask_1 = self.mask_losses(loss_1)
            mask_2 = self.mask_losses(loss_2)

            self.probas1 = self.get_probas(logits_1, y)
            self.probas2 = self.get_probas(logits_2, y)

            self.fraction_reject_1 = 1. - (mask_1.sum() / len(mask_1))
            self.fraction_reject_2 = 1. - (mask_2.sum() / len(mask_2))

        loss_1_update = torch.sum(self.Loss(logits_1, y) * mask_2) / mask_2.sum()
        loss_2_update = torch.sum(self.Loss(logits_2, y) * mask_1) / mask_1.sum()

        return loss_1_update, loss_2_update