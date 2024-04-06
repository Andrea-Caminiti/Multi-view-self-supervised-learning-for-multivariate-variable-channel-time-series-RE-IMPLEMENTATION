import torch
import torch.functional as F
import numpy as np 

def NT_Xent_Loss(z1, z2, temperature = 0.5): 
    simi = F.cosine_similarity(z1, z2, dim=-1)
    simi[torch.eye(torch.cat(z1, z2).size(0)).bool()] = float("-inf")

    target = torch.arange(z1.shape(0))
    target[0::2] += 1
    target[1::2] -= 1

    return F.cross_entropy(simi / temperature, target, reduction="mean")

def TS2VecLoss(z1, z2, alpha=0.5, temporal_u=0): #taken from https://github.com/zhihanyue/ts2vec/blob/main/models/losses.py
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * inst_contrastive(z1, z2)
        if d >= temporal_u:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temp_contrastive(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * inst_contrastive(z1, z2)
        d += 1
    return loss / d    

def inst_contrastive(z1, z2):
    B = z1.size(0)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temp_contrastive(z1, z2):
    T = z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

def COCOALoss(z, temperature=0.5, scale_loss=1/32, lambda_ = 3.9e-3):
    z = z.reshape(z.shape[0], z.shape[1], -1)
    z = z.transpose(1, 0)
    batch_size, view_size = z.shape[1], z.shape[0]

    z = F.normalize(z, dim = -1)
    pos_error = []
    for i in range(batch_size):
        sim = torch.matmul(z[:, i, :], z[:, i, :].T)
        sim = torch.ones([view_size, view_size]).to(z.device)-sim
        sim = torch.exp(sim/temperature)
        pos_error.append(sim.mean())
    
    neg_error = 0
    for i in range(view_size):
        sim = torch.matmul(z[i], z[i].T)
        sim = torch.exp(sim / temperature)
        tri_mask = np.ones(batch_size ** 2, dtype=np.bool).reshape(batch_size, batch_size)
        tri_mask[np.diag_indices(batch_size)] = False
        tri_mask = torch.tensor(tri_mask).to(z.device)
        off_diag_sim = torch.reshape(torch.masked_select(sim, tri_mask), [batch_size, batch_size - 1])
        neg_error += off_diag_sim.mean(-1)

    pos_error = torch.stack(pos_error)
    error = torch.sum(pos_error)* scale_loss + lambda_ * torch.sum(neg_error)
    return error