import numpy as np
import torch as T 
import torch.nn.functional as f
from torch import nn

class RecommendationTransformer(nn.Module):
    def __init__(self,
                 heads=16,
                 layers=6,
                 emb_dim=1024,
                 hidden_dim=256,
                 dropout=0.1):
        super().__init__()

        self.token_embed = nn.Embedding(4, hidden_dim)

        self.embeder = nn.Sequential(
            nn.Linear(emb_dim, 2*hidden_dim, dtype=T.float32),
            nn.GELU(),
            nn.Linear(2*hidden_dim, hidden_dim, dtype=T.float32),
            nn.GELU(),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim,
                activation=f.gelu,
                dropout=dropout,
                batch_first=True,
                dtype=T.float32),
            num_layers=layers,
            # norm=nn.LayerNorm(hidden_dim),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=T.float32),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim, dtype=T.float32),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, dtype=T.float32),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, dtype=T.float32),
        )

    def forward(self, src, src_token_mask, src_mask=None, src_key_padding_mask=None):
        src     = self.embeder(src) + self.token_embed(src_token_mask)
        out     = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        decoded = self.decoder1(out)
        done    = self.decoder2(out).squeeze(dim=-1)

        return decoded, done
    
def initEnd(shape):
    x = np.empty(shape, dtype=float)
    x.fill(1)
    return x

def initPad(shape):
    x = np.empty(shape, dtype=float)
    x.fill(0)
    return x