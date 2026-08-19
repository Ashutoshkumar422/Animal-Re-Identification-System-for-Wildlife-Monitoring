# models/mfa.py — Meta-Feature Adapter (MFA) as described in MetaWild (MM'25),
# extended with a multi-scale image branch (this work).

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (CLIP_MODEL, EMBED_DIM, NUM_HEADS_ATTN, MLP_HIDDEN_RATIO,
                    UNFREEZE_LAST_N_BLOCKS)


# ── Building block ────────────────────────────────────────────────────────────

class MLPAdapter(nn.Module):
    """
    Lightweight MLP adapter with residual connection.
    Used for both the Visual Feature Expert (VFE) and
    the Textual Metadata Expert (TME) — paper Section 4.1.
    """
    def __init__(self, embed_dim: int = EMBED_DIM,
                 hidden_ratio: int = MLP_HIDDEN_RATIO,  dropout: float = 0.3):
        super().__init__()
        hidden = embed_dim // hidden_ratio
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)   # residual: output stays in same space as input


# ── Gated Cross-Attention ─────────────────────────────────────────────────────

class GatedCrossAttention(nn.Module):
    """
    Gated cross-attention module — paper Section 4.2.

    Visual features are the Query; metadata embeddings are Key & Value.
    A learnable sigmoid gate γ ∈ [0,1] controls metadata contribution per sample.

    Equations:
        Q = I'_x · W_Q,   K = T'_M · W_K,   V = T'_M · W_V
        A = softmax(QK^T / √d)
        γ = sigmoid(MLP([I'_x ; T'_M]))
        I_meta = γ · (A·V) + I'_x          ← Eq. (3)
    """
    def __init__(self, embed_dim: int = EMBED_DIM,
                 num_heads: int = NUM_HEADS_ATTN):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.3
        )
        # Gate MLP: concat([I'_x, T'_M]) → scalar γ
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),   # γ ∈ [0, 1]
        )

    def forward(self, img_feat: torch.Tensor,
                txt_feat: torch.Tensor):
        """
        img_feat : (B, D) — I'_x from VFE
        txt_feat : (B, D) — T'_M from TME
        Returns  : I_meta (B, D), gamma (B, 1)
        """
        # Expand to sequence dimension for MultiheadAttention
        Q = img_feat.unsqueeze(1)   # (B, 1, D) — visual is the query
        K = txt_feat.unsqueeze(1)   # (B, 1, D) — metadata is key
        V = txt_feat.unsqueeze(1)   # (B, 1, D) — metadata is value

        attn_out, _ = self.cross_attn(Q, K, V)     # (B, 1, D)
        attn_out    = attn_out.squeeze(1)           # (B, D)

        # Gate: how much should metadata contribute for this specific image?
        gamma   = self.gate(torch.cat([img_feat, txt_feat], dim=-1))  # (B, 1)
        I_meta  = gamma * attn_out + img_feat       # Eq. (3) — gated residual
        return I_meta, gamma


# ── Full MFA model ────────────────────────────────────────────────────────────

class MetaFeatureAdapter(nn.Module):
    """
    Full Meta-Feature Adapter (MFA), extended with a multi-scale image branch.

    Wraps a frozen CLIP backbone and adds:
      0. Multi-scale image branch — fuses CLIP's global [CLS] token with a
         local descriptor pooled from the patch tokens (this work). The
         original MFA used the [CLS] token only, which cannot capture the
         fine rosette/stripe detail that fine-grained re-ID depends on.
      1. Visual Feature Expert  (VFE)  — adapts image embeddings
      2. Textual Metadata Expert (TME) — adapts metadata text embeddings
      3. Gated Cross-Attention  (GCA)  — fuses both modalities
      4. ReID head (BN bottleneck + classifier)

    Only the adapters, the multi-scale fusion, and the ReID head are trained;
    the CLIP backbone stays frozen.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # ── CLIP backbone (frozen, optionally with a fine-tuned tail) ────────
        self.clip_model, _ = clip.load(CLIP_MODEL, device="cpu")
        for param in self.clip_model.parameters():
            param.requires_grad_(False)
        # Optionally unfreeze the last N visual transformer blocks so the
        # backbone can adapt to camera-trap imagery.
        if UNFREEZE_LAST_N_BLOCKS > 0:
            tail = self.clip_model.visual.transformer.resblocks[-UNFREEZE_LAST_N_BLOCKS:]
            for blk in tail:
                for p in blk.parameters():
                    p.requires_grad_(True)

        # ── Multi-scale image fusion (global [CLS] + local patch pooling) ────
        # Residual branch: the final layer is zero-initialized, so at the start
        # of training the branch outputs 0 and the fused feature I_x equals the
        # global [CLS] feature — i.e. identical to the original MFA. Training
        # can then only *add* fine-grained local detail, never regress.
        self.ms_fuse = nn.Sequential(
            nn.Linear(EMBED_DIM * 3, EMBED_DIM),   # in = [global ; mean ; max]
            nn.LayerNorm(EMBED_DIM),
            nn.GELU(),
            nn.Linear(EMBED_DIM, EMBED_DIM),
        )

        # ── Feature Experts (Section 4.1) ────────────────────────────────────
        self.vfe = MLPAdapter(EMBED_DIM)    # Visual Feature Expert
        self.tme = MLPAdapter(EMBED_DIM)    # Textual Metadata Expert

        # ── Gated Cross-Attention (Section 4.2) ─────────────────────────────
        self.gca = GatedCrossAttention(EMBED_DIM)

        # ── ReID head ────────────────────────────────────────────────────────
        self.bottleneck = nn.BatchNorm1d(EMBED_DIM)
        self.bottleneck.bias.requires_grad_(True)
        self.classifier = nn.Linear(EMBED_DIM, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias,   0.0)
        # Zero-init the multi-scale residual so I_x starts == global feature.
        nn.init.zeros_(self.ms_fuse[-1].weight)
        nn.init.zeros_(self.ms_fuse[-1].bias)

    # ── Encoding helpers ──────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image_raw(self, images: torch.Tensor) -> torch.Tensor:
        """Raw CLIP global [CLS] embedding — used for the visual-only baseline."""
        return self.clip_model.encode_image(images).float()

    @torch.no_grad()
    def encode_text_raw(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.clip_model.encode_text(tokens).float()

    def _clip_visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        Manual CLIP-ViT forward returning EVERY token, projected to EMBED_DIM.
        Output: (B, 1 + num_patches, EMBED_DIM); token 0 is the global [CLS],
        tokens 1.. are the patch tokens (196 for ViT-B/16 at 224px).

        The frozen stem and early transformer blocks run under no_grad; the
        last UNFREEZE_LAST_N_BLOCKS blocks run with gradients enabled so they
        can be fine-tuned. With UNFREEZE_LAST_N_BLOCKS = 0 the entire pass is
        under no_grad (fully-frozen backbone, original MFA behaviour).
        """
        vis       = self.clip_model.visual
        resblocks = vis.transformer.resblocks
        split     = len(resblocks) - max(0, UNFREEZE_LAST_N_BLOCKS)

        # Frozen stem + early transformer blocks — no gradients, no activations.
        with torch.no_grad():
            x = vis.conv1(images)                          # (B, width, gh, gw)
            x = x.reshape(x.shape[0], x.shape[1], -1)      # (B, width, P)
            x = x.permute(0, 2, 1)                         # (B, P, width)
            cls = vis.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls, x], dim=1)                 # (B, 1+P, width)
            x = x + vis.positional_embedding.to(x.dtype)
            x = vis.ln_pre(x)
            x = x.permute(1, 0, 2)                         # NLD -> LND
            for blk in resblocks[:split]:
                x = blk(x)

        # Fine-tuned tail blocks — gradients flow into these.
        for blk in resblocks[split:]:
            x = blk(x)

        x = x.permute(1, 0, 2)                             # LND -> NLD
        x = vis.ln_post(x)
        if vis.proj is not None:
            x = x @ vis.proj                               # (B, 1+P, EMBED_DIM)
        return x.float()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor,
                tokens: torch.Tensor,
                return_all: bool = False):
        """
        Args:
            images     : (B, 3, H, W)
            tokens     : (B, 77)  — CLIP-tokenized metadata prompts
            return_all : if True, also return the pre-BN feature, T'_M and γ

        Returns (standard):
            logits : (B, num_classes)
            feat   : (B, EMBED_DIM)  — post-BN embedding (use this for retrieval)

        Returns (return_all=True):
            logits, feat, I_meta, text_emb (T'_M), gamma
            where I_meta is the PRE-BN fused feature — the triplet loss should
            be computed on it (standard BNNeck: triplet pre-BN, ID post-BN).
        """
        # ── Image branch — multi-scale (global [CLS] + local patch pooling) ──
        vis_tokens = self._clip_visual_tokens(images)    # (B, 1+P, D)
        g          = vis_tokens[:, 0]                    # global [CLS]  (B, D)
        patches    = vis_tokens[:, 1:]                   # patch tokens  (B, P, D)
        l_mean     = patches.mean(dim=1)                 # local mean    (B, D)
        l_max      = patches.max(dim=1).values           # local max     (B, D)
        # Residual multi-scale fusion (zero-init → starts == global feature).
        I_x        = g + self.ms_fuse(torch.cat([g, l_mean, l_max], dim=-1))
        I_x_prime  = self.vfe(I_x)                       # VFE: adapted visual

        # ── Text (metadata) branch ───────────────────────────────────────────
        T_M       = self.encode_text_raw(tokens)         # raw CLIP text (B, D)
        T_M_prime = self.tme(T_M)                        # TME: refined metadata

        # ── Gated fusion ─────────────────────────────────────────────────────
        I_meta, gamma = self.gca(I_x_prime, T_M_prime)   # Eq. (3) — pre-BN feat

        # ── ReID head (BNNeck) ───────────────────────────────────────────────
        feat   = self.bottleneck(I_meta)                 # post-BN feat
        logits = self.classifier(feat)

        if return_all:
            return logits, feat, I_meta, T_M_prime, gamma
        return logits, feat
