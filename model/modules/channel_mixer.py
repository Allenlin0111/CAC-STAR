import torch
import torch.nn as nn

class ChannelMixerHead(nn.Module):
    """
    在变量维(C)做全连接混合：放在输出投影之后最稳妥。
    输入/输出: [B, L, C] -> [B, L, C]（通常 C=c_out）
    """
    def __init__(self, c_out, hidden_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden = max(4, int(c_out * hidden_ratio))
        self.net = nn.Sequential(
            nn.Linear(c_out, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, c_out)
        )
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x):  # x: [B, L, C]
        y = self.net(x)
        return self.norm(x + y)  # 残差 + LN



