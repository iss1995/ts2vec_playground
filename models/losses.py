import torch
from torch import nn
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
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

    # check if I understand this loss correctly
    if 0:
        with torch.no_grad():
            # first normalize for numerical stability
            z1 = z1 / z1.norm(dim=-1, keepdim=True)
            z2 = z2 / z2.norm(dim=-1, keepdim=True)
            z = torch.cat([z1, z2], dim=0)  # 2B x T x C
            z = z.transpose(0, 1)  # T x 2B x C
            sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
            logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
            logits += torch.triu(sim, diagonal=1)[:, :, 1:]
            logits = -F.log_softmax(logits, dim=-1)
            
            i = torch.arange(B, device=z1.device)
            loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2

            # check if I understand the loss correctly
            # firstly, now that you normalize among batches you need to anchor the batch and use the timestep instead of the batch index.
            z1_transpose = z1.transpose(0, 1)
            z2_transpose = z2.transpose(0, 1)

            z1z2 = torch.bmm(z1_transpose, z2_transpose.transpose(1, 2))
            z1z1 = torch.bmm(z1_transpose, z1_transpose.transpose(1, 2))
            z2z2 = torch.bmm(z2_transpose, z2_transpose.transpose(1, 2))

            # If you don't normalize the vectors, you get inf both here and in the denominator.
            # This results in nan loss. This is why the original implementation uses the log_softmax, which implements the logsumexp trick.
            nom = torch.exp(z1z2[:, torch.arange(B), torch.arange(B)])
            denom = torch.sum(
                torch.exp(z1z2) + torch.exp(z1z1 - torch.diag_embed(z1z1[:, torch.arange(B), torch.arange(B)])), 
                dim=-1,
            )
            ration = nom / denom
            ration = torch.where(torch.isinf(ration), torch.finfo(ration.dtype).max, ration)
            loss_1 = -torch.log(ration).mean() # view 1

            denom_2 = torch.sum(
                torch.exp(z1z2) + torch.exp(z2z2 - torch.diag_embed(z2z2[:, torch.arange(B), torch.arange(B)])), 
                dim=-1,
            )
            ration_2 = nom / denom_2
            ration_2 = torch.where(torch.isinf(ration_2), torch.finfo(ration_2.dtype).max, ration_2)
            loss_2 = -torch.log(ration_2).mean() # view 2

            test_loss = (loss_1 + loss_2) / 2 # total loss

            print(f"view 1 loss: {loss_1:.4f}, view 2 loss: {loss_2:.4f}") # view 1 and view 2
            print(f"test loss: {test_loss:.4f}, loss: {loss:.4f}") # total loss and original loss

            # Check for NaNs in numerator and denominator
            if torch.isnan(nom).any() or torch.isnan(denom).any():
                print("Warning: NaN detected in instance contrastive loss calculation")
                print(f"nom shape: {nom.shape}, has NaN: {torch.isnan(nom).any()}, has inf: {torch.isinf(nom).any()}")
                print(f"denom shape: {denom.shape}, has NaN: {torch.isnan(denom).any()}, has inf: {torch.isinf(denom).any()}")
                print(f"denom_2 shape: {denom_2.shape}, has NaN: {torch.isnan(denom_2).any()}, has inf: {torch.isinf(denom_2).any()}")
                print(f"z1z2 has NaN: {torch.isnan(z1z2).any()}, has inf: {torch.isinf(z1z2).any()}")
                print(f"z1z1 has NaN: {torch.isnan(z1z1).any()}, has inf: {torch.isinf(z1z1).any()}")
                print(f"z2z2 has NaN: {torch.isnan(z2z2).any()}, has inf: {torch.isinf(z2z2).any()}")

            print(f"view 1 loss: {loss_1:.4f}, view 2 loss: {loss_2:.4f}") # view 1 and view 2
            print(f"test loss: {test_loss:.4f}, loss: {loss:.4f}") # total loss and original loss
            print(f"Deviation: {(test_loss - loss).abs().item()/loss.abs().item()*100:.4f}%")

    return loss

def temporal_contrastive_loss(z1, z2):
    """
    The implementation is more numerically stable than taking the exponents of the dot products.
    """

    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2

    # check if I understand this loss correctly
    if 0:
        with torch.no_grad():
            # first normalize for numerical stability
            z1 = z1 / z1.norm(dim=-1, keepdim=True)
            z2 = z2 / z2.norm(dim=-1, keepdim=True)

            # recalculate the loss with the normalized vectors
            z = torch.cat([z1, z2], dim=1)  # B x 2T x C
            sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
            logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
            logits += torch.triu(sim, diagonal=1)[:, :, 1:]
            logits = -F.log_softmax(logits, dim=-1)
            
            t = torch.arange(T, device=z1.device)
            loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2

            # check if I understand the loss correctly
            z1z2 = torch.bmm(z1, z2.transpose(1, 2))
            z1z1 = torch.bmm(z1, z1.transpose(1, 2))
            z2z2 = torch.bmm(z2, z2.transpose(1, 2))

            # If you don't normalize the vectors, you get inf both here and in the denominator.
            # This results in nan loss. This is why the original implementation uses the log_softmax, which implements the logsumexp trick.
            nom = torch.exp(z1z2[:, torch.arange(T), torch.arange(T)]) 
            denom = torch.sum(
                torch.exp(z1z2) + torch.exp(z1z1 - torch.diag_embed(z1z1[:, torch.arange(T), torch.arange(T)])), 
                dim=-1,
            )
            # # Replace inf values with the maximum finite value
            # nom = torch.where(torch.isinf(nom), torch.finfo(nom.dtype).max, nom)
            # denom = torch.where(torch.isinf(denom), torch.finfo(denom.dtype).max, denom)

            # denom = torch.exp(z1z2 - torch.diag_embed(z1z1[:, torch.arange(T), torch.arange(T)])) + torch.exp(z2z2 - torch.diag_embed(z2z2[:, torch.arange(T), torch.arange(T)]))
            ration = nom / denom
            ration = torch.where(torch.isinf(ration), torch.finfo(ration.dtype).max, ration)
            loss_1 = -torch.log(ration).mean() # view 1

            denom_2 = torch.sum(
                torch.exp(z1z2) + torch.exp(z2z2 - torch.diag_embed(z2z2[:, torch.arange(T), torch.arange(T)])), 
                dim=-1,
            )
            # denom_2 = torch.where(torch.isinf(denom_2), torch.finfo(denom_2.dtype).max, denom_2)

            ration_2 = nom / denom_2
            ration_2 = torch.where(torch.isinf(ration_2), torch.finfo(ration_2.dtype).max, ration_2)
            loss_2 = -torch.log(ration_2).mean() # view 2

            test_loss = (loss_1 + loss_2) / 2 # total loss

            # Check for NaNs in numerator and denominator
            if torch.isnan(nom).any() or torch.isnan(denom).any():
                print("Warning: NaN detected in temporal contrastive loss calculation")
                print(f"nom shape: {nom.shape}, has NaN: {torch.isnan(nom).any()}, has inf: {torch.isinf(nom).any()}")
                print(f"denom shape: {denom.shape}, has NaN: {torch.isnan(denom).any()}, has inf: {torch.isinf(denom).any()}")
                print(f"denom_2 shape: {denom_2.shape}, has NaN: {torch.isnan(denom_2).any()}, has inf: {torch.isinf(denom_2).any()}")
                print(f"z1z2 has NaN: {torch.isnan(z1z2).any()}, has inf: {torch.isinf(z1z2).any()}")
                print(f"z1z1 has NaN: {torch.isnan(z1z1).any()}, has inf: {torch.isinf(z1z1).any()}")
                print(f"z2z2 has NaN: {torch.isnan(z2z2).any()}, has inf: {torch.isinf(z2z2).any()}")

            print(f"view 1 loss: {loss_1:.4f}, view 2 loss: {loss_2:.4f}") # view 1 and view 2
            print(f"test loss: {test_loss:.4f}, loss: {loss:.4f}") # total loss and original loss

    return loss
