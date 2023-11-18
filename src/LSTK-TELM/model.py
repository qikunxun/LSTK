import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

device = torch.device('cuda')
class TELMModel(nn.Module):
    def __init__(self, n, T, L, tau_1=10, tau_2=0.2, use_gpu=False):
        super(TELMModel, self).__init__()
        self.T = T
        self.L = L
        self.n = n
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.w = nn.Parameter(torch.Tensor(self.T - 1, self.L, 2 * self.n))
        nn.init.kaiming_uniform_(self.w.view(2 * self.n, -1), a=np.sqrt(5))
        self.w_inv = nn.Parameter(torch.Tensor(self.T - 1, self.L, 2 * self.n))
        nn.init.kaiming_uniform_(self.w_inv.view(2 * self.n, -1), a=np.sqrt(5))
        self.weight = nn.Parameter(torch.Tensor(self.L, 1))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        self.alpha = nn.Parameter(torch.Tensor(self.T - 1, 2, self.L))
        nn.init.zeros_(self.alpha)
        self.beta = nn.Parameter(torch.Tensor(self.T - 1, 2, self.L))
        nn.init.zeros_(self.beta)

        self.use_gpu = use_gpu
        self.dropout = nn.Dropout(0.3)

    def forward(self, input, input_all, all_states, t, entity2id, flag, is_training=False):
        s = 0
        if t != self.T - 1:
            w_probs = self.w[t]
            alpha = self.alpha[t][0]
            beta = self.beta[t][0]
            if flag:
                w_probs = self.w_inv[t]
                alpha = self.alpha[t][1]
                beta = self.beta[t][1]
            alpha = torch.sigmoid(alpha)
            beta = torch.sigmoid(beta)
            w_probs_h = torch.softmax(w_probs[:, :self.n], dim=-1)
            w_probs_s = torch.softmax(w_probs[:, self.n:], dim=-1)
            w_probs = torch.cat([w_probs_h * torch.unsqueeze(alpha, dim=-1),
                                 w_probs_s * torch.unsqueeze(beta, dim=-1),
                                 1 - self.activation(torch.unsqueeze(alpha, dim=-1)
                                                     + torch.unsqueeze(beta, dim=-1))], dim=-1)
            if t == 0:
                w = w_probs
                s_tmp = torch.sparse.mm(input, torch.permute(w, (1, 0))).view(len(entity2id), -1, self.L)
                s = s_tmp
                s = self.activation(s)
                if is_training: s = self.dropout(s)
            if t >= 1:
                w = w_probs
                s_tmp = torch.sparse.mm(input_all, all_states[t - 1].reshape(len(entity2id), -1))
                s_tmp = s_tmp.view(len(entity2id), 2 * self.n + 1, -1, self.L)
                if is_training: s_tmp = self.dropout(s_tmp)
                s_tmp = torch.einsum('mrnl,lr->mnl', s_tmp, w)
                s = s_tmp
                s = self.activation(s)
        else:
            s = all_states[t - 1]
            if is_training: s = self.dropout(s)
            s = torch.squeeze(torch.einsum('nml,lk->nmk', s, torch.tanh(self.weight)), dim=-1)
        return s

    def negative_loss(self, p_score, n_score):
        y = torch.autograd.Variable(torch.Tensor([1]))
        if self.use_gpu: y = y.to(device)
        loss = torch.square(1 - torch.minimum(p_score, torch.ones_like(p_score))) + torch.square(torch.maximum(n_score, torch.zeros_like(n_score)))
        return loss


    def log_loss(self, p_score, label, logit_mask, thr=1e-20):
        one_hot = F.one_hot(torch.LongTensor([label]), p_score.shape[-1])
        if self.use_gpu:
            one_hot = one_hot.to(device)
            logit_mask = logit_mask.to(device)
        p_score = p_score - 1e30 * logit_mask.unsqueeze(dim=0)
        loss = -torch.sum(
            one_hot * torch.log(torch.maximum(F.softmax(p_score / self.tau_2, dim=-1), torch.ones_like(p_score) * thr)),
            dim=-1)
        return loss

    def gt_constrain(self, x):
        zero = torch.autograd.Variable(torch.Tensor([0]).detach())
        if self.use_gpu: zero = zero.to(device)
        return torch.square(torch.minimum(x, zero))

    def lt_constrain(self, x):
        zero = torch.autograd.Variable(torch.Tensor([0]).detach())
        if self.use_gpu: zero = zero.to(device)
        return torch.square(torch.maximum(x, zero))

    def eq_constrain(self, x):
        zero = torch.autograd.Variable(torch.Tensor([0]).detach())
        if self.use_gpu: zero = zero.to(device)
        return torch.square(torch.maximum(x, zero) + torch.minimum(x, zero))

    def activation(self, x):
        one = torch.autograd.Variable(torch.Tensor([1]))
        zero = torch.autograd.Variable(torch.Tensor([0]).detach())
        if self.use_gpu:
            one = one.to(device)
            zero = zero.to(device)
        return torch.minimum(torch.maximum(x, zero), one)

    def activation_leaky(self, x):
        one = torch.autograd.Variable(torch.Tensor([1]))
        if self.use_gpu:
            one = one.to(device)
        return torch.minimum(F.leaky_relu(x), one)
