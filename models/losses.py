# models/losses.py — ReIDLoss (ID + Triplet) and CrossModalContrastiveLoss (L_iA)

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TRIPLET_MARGIN, LABEL_SMOOTHING


class TripletLoss(nn.Module):
    """
    Batch-hard triplet loss with in-batch mining.
    For each anchor, pick hardest positive (max dist) and hardest negative (min dist).
    """
    def __init__(self, margin: float = TRIPLET_MARGIN):
        super().__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Pairwise L2 distance matrix (B, B)
        dist = torch.cdist(
            F.normalize(features, dim=-1),
            F.normalize(features, dim=-1),
        )
        B = len(labels)
        loss = torch.tensor(0.0, device=features.device)
        count = 0
        for i in range(B):
            pos_mask = (labels == labels[i])
            pos_mask[i] = False                     # exclude self
            neg_mask = (labels != labels[i])
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            hardest_pos = dist[i][pos_mask].max()   # farthest positive
            hardest_neg = dist[i][neg_mask].min()   # closest negative
            loss += F.relu(hardest_pos - hardest_neg + self.margin)
            count += 1
        return loss / max(count, 1)


class ReIDLoss(nn.Module):
    """
    Combined ReID loss: identity classification + batch-hard triplet.
    L = λ_id * L_id + λ_tri * L_tri
    """
    def __init__(self, num_classes: int,
                 lambda_id: float = 1.0,
                 lambda_tri: float = 1.0):
        super().__init__()
        self.lambda_id  = lambda_id
        self.lambda_tri = lambda_tri
        self.id_loss     = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        self.triplet     = TripletLoss()

    def forward(self, logits: torch.Tensor,
                features: torch.Tensor,
                labels: torch.Tensor):
        loss_id  = self.id_loss(logits, labels)
        loss_tri = self.triplet(features, labels)
        return self.lambda_id * loss_id + self.lambda_tri * loss_tri, {
            "loss_id":  loss_id.item(),
            "loss_tri": loss_tri.item(),
        }


class CrossModalContrastiveLoss(nn.Module):
    """
    Cross-modal alignment loss L_iA (Equation 4 in the paper).

    Treats each (T'_M_i, I_meta_i) as a matched positive pair.
    All other combinations in the batch are negatives.
    Temperature τ is a learnable parameter initialized to 0.07.
    """
    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(init_temp).log())

    @property
    def temperature(self):
        # Clamp to avoid numerical instability
        return self.log_temp.exp().clamp(0.01, 1.0)

    def forward(self, text_emb: torch.Tensor,
                img_emb: torch.Tensor) -> torch.Tensor:
        """
        text_emb : (B, D) — T'_M  (metadata-aware text embeddings)
        img_emb  : (B, D) — I_meta (fused image embeddings)
        """
        t = F.normalize(text_emb, dim=-1)
        v = F.normalize(img_emb,  dim=-1)

        # Similarity matrix (B, B)
        logits = torch.matmul(t, v.T) / self.temperature

        # Diagonal = positive pairs
        labels = torch.arange(len(t), device=t.device)

        # Symmetric: text→image AND image→text
        loss_t2v = F.cross_entropy(logits,   labels)
        loss_v2t = F.cross_entropy(logits.T, labels)
        return (loss_t2v + loss_v2t) / 2
