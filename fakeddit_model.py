import torch
import torch.nn as nn
class CoAttentionLayer(nn.Module):
    def __init__(self, embed_dim=1536, num_heads=8, ffn_expand=2, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_t2i = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_i2t = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ln_fuse = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_expand),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_expand, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, text_feat, img_feat):
        """
        text_feat: [B,1536]
        img_feat:  [B,1536]
        """
        t = text_feat.unsqueeze(1)  # [B,1,1536]
        i = img_feat.unsqueeze(1)   # [B,1,1536]
        t2i, _ = self.attn_t2i(t, i, i)
        i2t, _ = self.attn_i2t(i, t, t)

        t_res = t + t2i
        i_res = i + i2t

        prior = 0.5 * (t + i)
        fuse_pre = t_res + i_res + prior
        fuse = self.ln_fuse(fuse_pre)  # [B,1,1536]

        ffn_out = self.ffn(fuse)
        out = fuse + ffn_out

        return out.squeeze(1)  # [B,1536]

class CoAttentionBlock(nn.Module):
    def __init__(self, embed_dim=1536, num_heads=8, dropout=0.3):
        super().__init__()
        self.layer1 = CoAttentionLayer(embed_dim, num_heads, dropout=dropout)
        self.layer2 = CoAttentionLayer(embed_dim, num_heads, dropout=dropout)
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, text_feat, img_feat):
        f1 = self.layer1(text_feat, img_feat)
        f2 = self.layer2(text_feat, img_feat)
        comb = torch.cat([f1, f2], dim=-1)  # [B,3072]
        return self.combine(comb)           # [B,1536]


class TriBranchMSCAN(nn.Module):
    def __init__(self, embed_dim=1536, num_heads=8, dropout=0.3, out_dim=2):
        super().__init__()
        self.embed_dim = embed_dim  # 1536维

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.block_txt_biip = CoAttentionBlock(embed_dim, num_heads, dropout)
        self.block_txt_cap  = CoAttentionBlock(embed_dim, num_heads, dropout)
        self.block_s1 = CoAttentionBlock(embed_dim, num_heads, dropout)
        self.block_s2 = CoAttentionBlock(embed_dim, num_heads, dropout)
        self.block_s3 = CoAttentionBlock(embed_dim, num_heads, dropout)

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 3)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.eps = 1e-6

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 4, 512),   # 1536*4
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim)
        )

    def _seq_mean(self, x):  # [B,90,1536] -> [B,1536]
        return self.pool(x.transpose(1, 2)).squeeze(-1)

    def _freq_transform(self, s1):  # s1: [B,1536]
        X = torch.fft.fft(s1, dim=-1)  # [B,1536], complex
        mag = torch.abs(X)  # [B,1536], real
        mag = torch.log1p(mag)  # log1p
        mag_norm = mag / (mag.norm(p=2, dim=-1, keepdim=True) + self.eps)
        return mag_norm

    def forward(self, text_seq, image_3sc, biip_seq, cap_seq):
        txt = self._seq_mean(text_seq)   # [B,1536]
        bi  = self._seq_mean(biip_seq)   # [B,1536]
        cap = self._seq_mean(cap_seq)    # [B,1536]
        s1  = image_3sc[:, 0, :]
        s2  = image_3sc[:, 1, :]
        s3  = image_3sc[:, 2, :]         # [B,1536]

        f_txt_biip = self.block_txt_biip(txt, bi)   # [B,1536]

        f_txt_cap  = self.block_txt_cap(txt, cap)   # [B,1536]

        c1 = self.block_s1(f_txt_cap, s1)           # [B,1536]
        f_freq = self._freq_transform(s1)           # [B,1536]
        c2 = self.block_s2(c1, f_freq)


        fused = torch.cat([f_txt_biip, f_txt_cap, c1, c2], dim=-1)
        logits = self.classifier(fused)                                  # [B,2]
        return logits


