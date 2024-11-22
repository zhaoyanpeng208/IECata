import os
import random
import numpy as np
import torch
import torch.nn as nn
import dgl
import logging
from collections import Counter
import math
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prottrans_graph_collate_func(x):
    d, p, y, w= zip(*x)
    d = dgl.batch(d)
    p = np.array(p)
    w = np.array(w)
    # return d, torch.tensor(np.array(p)), torch.tensor(y)
    return d, torch.tensor(p), torch.tensor(y), torch.tensor(w)

def integer_graph_collate_func(x):
    d, p, y, w = zip(*x)
    d = dgl.batch(d)
    w = np.array(w)
    return d, torch.tensor(np.array(p)), torch.tensor(y),torch.tensor(w)

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """
    Integer encoding for protein string sequence.
    Args:
        sequence (str): Protein string sequence.
        max_length: Maximum encoding length of input protein string.
    """
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding

# evidential regression
def evidential_loss(mu, v, alpha, beta, targets, lamba = 0.1):
    """
    Use Deep Evidential Regression Sum of Squared Error loss

    :mu: Pred mean parameter for NIG
    :v: Pred lambda parameter for NIG
    :alpha: predicted parameter for NIG
    :beta: Predicted parmaeter for NIG
    :targets: Outputs to predict

    :return: Loss
    """

    # Calculate SOS
    # Calculate gamma terms in front
    def Gamma(x):
        return torch.exp(torch.lgamma(x))

    coeff_denom = 4 * Gamma(alpha) * v * torch.sqrt(beta)
    coeff_num = Gamma(alpha - 0.5)
    coeff = coeff_num / coeff_denom

    # Calculate target dependent loss
    second_term = 2 * beta * (1 + v)
    second_term += (2 * alpha - 1) * v * torch.pow((targets - mu), 2)
    L_SOS = coeff * second_term

    # Calculate regularizer
    L_REG = torch.pow((targets - mu), 2) * (2 * alpha + v)

    loss_val = L_SOS + lamba * L_REG

    return loss_val
#
def Smooth_Label_CBW(Label_new,beta=0.9):
    labels = Label_new
    for i in range(len(labels)):
        labels[i] = labels[i] - min(labels)
    bin_index_per_label = [int(label*10) for label in labels]
    # print(bin_index_per_label)
    Nb = max(bin_index_per_label) + 1
    print(Nb)
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    print(num_samples_of_bins)
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    print(emp_label_dist, len(emp_label_dist))
    eff_label_dist = []
    # beta = 0.9
    for i in range(len(emp_label_dist)):
        eff_label_dist.append((1-math.pow(beta, emp_label_dist[i])) / (1-beta))
    print(eff_label_dist)
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]
    weights = np.array(weights)
    print(weights)
    print(len(weights))
    return weights


# def Smooth_Label_CBW(Label_new):
#     labels = Label_new
#     for i in range(len(labels)):
#         labels[i] = labels[i] - min(labels)
#     bin_index_per_label = [int(label*10) for label in labels]
#     # print(bin_index_per_label)
#     Nb = max(bin_index_per_label) + 1
#     print(Nb)
#     num_samples_of_bins = dict(Counter(bin_index_per_label))
#     print(num_samples_of_bins)
#     emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
#     print(emp_label_dist, len(emp_label_dist))
#     eff_label_dist = []
#     # beta = 0.9
#     for i in range(len(emp_label_dist)):
#         eff_label_dist.append((1-math.pow(beta, emp_label_dist[i])) / (1-beta))
#     print(eff_label_dist)
#     eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
#     weights = [np.float32(1 / x) for x in eff_num_per_label]
#     weights = np.array(weights)
#     print(weights)
#     print(len(weights))
#     return weights

def Smooth_Label_CSW(Label_new):
    labels = Label_new
    for i in range(len(labels)):
        labels[i] = labels[i] - min(labels)
    bin_index_per_label = [int(label * 10) for label in labels]
    # print(bin_index_per_label)
    Nb = max(bin_index_per_label) + 1
    print(Nb)
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    print(num_samples_of_bins)
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    print(emp_label_dist, len(emp_label_dist))
    eff_label_dist = []
    eff_label_dist = emp_label_dist
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    #weights = [math.sqrt(np.float32(1 / x)) for x in eff_num_per_label]
    #weights = [math.pow(np.float32(1 / x), 2) for x in eff_num_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]


    weights = np.array(weights)
    print(weights)
    print(len(weights))
    return weights

def Smooth_Label_DMW(Label_new):
    labels = Label_new
    weights = np.ones([len(Label_new)], dtype=float)
    for i in range(len(labels)):
        if Label_new[i] > 5 or Label_new[i] < -5:
            weights[i] = 2
    print(weights)
    print(len(weights))
    return weights


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))
    return kernel_window

def Smooth_Label_LDS(Label_new):
    labels = Label_new
    for i in range(len(labels)):
        labels[i] = labels[i] - min(labels)
    bin_index_per_label = [int(label*4) for label in labels]
    # print(bin_index_per_label)
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
    print(emp_label_dist, len(emp_label_dist))
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=3, sigma=1)
    print(lds_kernel_window)
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
    print(eff_label_dist, emp_label_dist)
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]
    weights = np.array(weights)
    # print(weights)
    return weights



class EvidentialLossSumOfSquares(nn.Module):
    """The evidential loss function on a matrix.

    This class is implemented with slight modifications from the paper. The major
    change is in the regularizer parameter mentioned in the paper. The regularizer
    mentioned in the paper didnot give the required results, so we modified it
    with the KL divergence regularizer from the paper. In orderto overcome the problem
    that KL divergence are missing near zero so we add the minimum values to alpha,
    beta and lambda and compare distance with NIG(alpha=1.0, beta=0.1, lambda=1.0)

    This class only allows for rank-4 inputs for the output `targets`, and expectes
    `inputs` be of the form [mu, alpha, beta, lambda]

    alpha, beta and lambda needs to be positive values.
    """

    def __init__(self, debug=False, return_all=False):
        """Sets up loss function.

        Args:
          debug: When set to 'true' prints all the intermittent values
          return_all: When set to 'true' returns all loss values without taking average

        """
        super(EvidentialLossSumOfSquares, self).__init__()

        self.debug = debug
        self.return_all_values = return_all
        self.MAX_CLAMP_VALUE = 5.0  # Max you can go is 85 because exp(86) is nan  Now exp(5.0) is 143 which is max of a,b and l

    def kl_divergence_nig(self, mu1, mu2, alpha_1, beta_1, lambda_1):
        alpha_2 = torch.ones_like(mu1) * 1.0
        beta_2 = torch.ones_like(mu1) * 0.1
        lambda_2 = torch.ones_like(mu1) * 1.0

        t1 = 0.5 * (alpha_1 / beta_1) * ((mu1 - mu2) ** 2) * lambda_2
        # t1 = 0.5 * (alpha_1/beta_1) * (torch.abs(mu1 - mu2))  * lambda_2
        t2 = 0.5 * lambda_2 / lambda_1
        t3 = alpha_2 * torch.log(beta_1 / beta_2)
        t4 = -torch.lgamma(alpha_1) + torch.lgamma(alpha_2)
        t5 = (alpha_1 - alpha_2) * torch.digamma(alpha_1)
        t6 = -(beta_1 - beta_2) * (alpha_1 / beta_1)
        return (t1 + t2 - 0.5 + t3 + t4 + t5 + t6)

    def forward(self, inputs, targets):
        """ Implements the loss function

        Args:
          inputs: The output of the neural network. inputs has 4 dimension
            in the format [mu, alpha, beta, lambda]. Must be a tensor of
            floats
          targets: The expected output

        Returns:
          Based on the `return_all` it will return mean loss of batch or individual loss

        """
        assert torch.is_tensor(inputs)
        assert torch.is_tensor(targets)
        assert (inputs[:, 1] > 0).all()
        assert (inputs[:, 2] > 0).all()
        assert (inputs[:, 3] > 0).all()

        targets = targets.view(-1)
        y = inputs[:, 0].view(-1)  # first column is mu,delta, predicted value
        a = inputs[:, 1].view(-1) + 1.0  # alpha
        b = inputs[:, 2].view(-1) + 0.1  # beta to avoid zero
        l = inputs[:, 3].view(-1) + 1.0  # lamda

        if self.debug:
            print("a :", a)
            print("b :", b)
            print("l :", l)

        J1 = torch.lgamma(a - 0.5)
        J2 = -torch.log(torch.tensor([4.0]))
        J3 = -torch.lgamma(a)
        J4 = -torch.log(l)
        J5 = -0.5 * torch.log(b)
        J6 = torch.log(2 * b * (1 + l) + (2 * a - 1) * l * (y - targets) ** 2)

        if self.debug:
            print("lgama(a - 0.5) :", J1)
            print("log(4):", J2)
            print("lgama(a) :", J3)
            print("log(l) :", J4)
            print("log( ---- ) :", J6)

        J = J1 + J2 + J3 + J4 + J5 + J6
        # Kl_divergence = torch.abs(y - targets) * (2*a + l)/b ######## ?????
        # Kl_divergence = ((y - targets)**2) * (2*a + l)
        # Kl_divergence = torch.abs(y - targets) * (2*a + l)
        # Kl_divergence = 0.0
        # Kl_divergence = (torch.abs(y - targets) * (a-1) *  l)/b
        Kl_divergence = self.kl_divergence_nig(y, targets, a, b, l)

        if self.debug:
            print("KL ", Kl_divergence.data.numpy())
        loss = torch.exp(J) + Kl_divergence

        if self.debug:
            print("loss :", loss.mean())

        if self.return_all_values:
            ret_loss = loss
        else:
            ret_loss = loss.mean()
        # if torch.isnan(ret_loss):
        #  ret_loss.item() = self.prev_loss + 10
        # else:
        #  self.prev_loss = ret_loss.item()

        return ret_loss


class EvidentialLossNLL(nn.Module):
    """The evidential loss function on a matrix.

    This class is implemented with slight modifications from the paper. The major
    change is in the regularizer parameter mentioned in the paper. The regularizer
    mentioned in the paper didnot give the required results, so we modified it
    with the KL divergence regularizer from the paper. In orderto overcome the problem
    that KL divergence are missing near zero so we add the minimum values to alpha,
    beta and lambda and compare distance with NIG(alpha=1.0, beta=0.1, lambda=1.0)

    This class only allows for rank-4 inputs for the output `targets`, and expectes
    `inputs` be of the form [mu, alpha, beta, lambda]

    alpha, beta and lambda needs to be positive values.
    """

    def __init__(self, debug=False, return_all=False):
        """Sets up loss function.

        Args:
          debug: When set to 'true' prints all the intermittent values
          return_all: When set to 'true' returns all loss values without taking average

        """
        super(EvidentialLossNLL, self).__init__()

        self.debug = debug
        self.return_all_values = return_all
        self.MAX_CLAMP_VALUE = 5.0  # Max you can go is 85 because exp(86) is nan  Now exp(5.0) is 143 which is max of a,b and l

    def kl_divergence_nig(self, mu1, mu2, alpha_1, beta_1, lambda_1):
        alpha_2 = torch.ones_like(mu1) * 1.0
        beta_2 = torch.ones_like(mu1) * 0.1
        lambda_2 = torch.ones_like(mu1) * 1.0

        t1 = 0.5 * (alpha_1 / beta_1) * ((mu1 - mu2) ** 2) * lambda_2
        # t1 = 0.5 * (alpha_1/beta_1) * (torch.abs(mu1 - mu2))  * lambda_2
        t2 = 0.5 * lambda_2 / lambda_1
        t3 = alpha_2 * torch.log(beta_1 / beta_2)
        t4 = -torch.lgamma(alpha_1) + torch.lgamma(alpha_2)
        t5 = (alpha_1 - alpha_2) * torch.digamma(alpha_1)
        t6 = -(beta_1 - beta_2) * (alpha_1 / beta_1)
        return (t1 + t2 - 0.5 + t3 + t4 + t5 + t6)

    def forward(self, mu, v, alpha, beta, targets, lam, epsilon):
        """ Implements the loss function

        Args:
          inputs: The output of the neural network. inputs has 4 dimension
            in the format [mu, alpha, beta, lambda]. Must be a tensor of
            floats
          targets: The expected output

        Returns:
          Based on the `return_all` it will return mean loss of batch or individual loss

        """
        # assert torch.is_tensor(inputs)
        assert torch.is_tensor(targets)
        #     assert (inputs[:,1] > 0).all()
        #     assert (inputs[:,2] > 0).all()
        #     assert (inputs[:,3] > 0).all()

        targets = targets.view(-1)
        # mu = inputs[:, 0].view(-1)  # first column is mu,predicted value
        # loglambdas = inputs[:, 1].view(-1)  # lamda
        # logalphas = inputs[:, 2].view(-1)  # alpha to limit >1
        # logbetas = inputs[:, 3].view(-1)  # beta

        # if self.debug:
        #     print("a :", a)
        #     print("b :", b)
        #     print("l :", l)

        # new_loss_230618

        # min_val = 1e-6
        #
        # v = torch.nn.Softplus()(loglambdas) + min_val
        # alpha = torch.nn.Softplus()(logalphas) + min_val + 1  # add 1 for numerical contraints of Gamma function
        # beta = torch.nn.Softplus()(logbetas) + min_val

        twoBlambda = 2 * beta * (1 + v)
        nll = 0.5 * torch.log(np.pi / v) \
              - alpha * torch.log(twoBlambda) \
              + (alpha + 0.5) * torch.log(v * (targets - mu) ** 2 + twoBlambda) \
              + torch.lgamma(alpha) \
              - torch.lgamma(alpha + 0.5)

        L_NLL = nll  # torch.mean(nll, dim=-1)

        # Calculate regularizer based on absolute error of prediction
        error = torch.abs((targets - mu))
        reg = error * (2 * v + alpha)
        L_REG = reg  # torch.mean(reg, dim=-1)

        #     Loss = L_NLL + L_REG
        # TODO If we want to optimize the dual- of the objective use the line below:
        #     print('lambda=',lam)
        loss = L_NLL + lam * (L_REG - epsilon)
        ret_loss = loss.mean()

        ######initial_loss

        #     J1 = torch.lgamma(a - 0.5)
        #     J2 = -torch.log(torch.tensor([4.0]))

        #     device = ('cuda' if torch.cuda.is_available() else 'cpu')

        #     J2=J2.to(device)
        #     J3 = -torch.lgamma(a)
        #     J4 = -torch.log(l)
        #     J5 = -0.5*torch.log(b)
        #     J6 = torch.log(2*b*(1 + l) + (2*a - 1)*l*(y-targets)**2)

        #     if self.debug:
        #       print("lgama(a - 0.5) :", J1)
        #       print("log(4):", J2)
        #       print("lgama(a) :", J3)
        #       print("log(l) :", J4)
        #       print("log( ---- ) :", J6)

        #     J = J1 + J3 + J4 + J5 + J6 + J2

        #     Kl_divergence = self.kl_divergence_nig(y, targets, a, b, l)

        #     if self.debug:
        #       print ("KL ",Kl_divergence.data.numpy())
        #     loss = torch.exp(J) + Kl_divergence

        #     if self.debug:
        #       print ("loss :", loss.mean())

        #     if self.return_all_values:
        #       ret_loss = loss
        #     else:
        #       ret_loss = loss.mean()
        #######

        return ret_loss
