import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerformerLayer(nn.Module):
    """
    Single Performer layer with modular FAVOR+ kernel.
    Supports ReLU, ELU, and Exp random feature mappings.
    """

    def __init__(self, dim, n_heads, nb_features=256, dropout=0.1, kernel="relu"):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.nb_features = nb_features
        self.kernel = kernel.lower()

        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        assert self.kernel in {"relu", "elu", "exp"}, f"Unsupported kernel: {self.kernel}"

        # Linear projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

        # FAVOR+ random projection matrix
        self.register_buffer("projection_matrix", self.create_projection())

    def create_projection(self):
        """Create orthogonal random projection matrix for FAVOR+."""
        proj = torch.randn(self.nb_features, self.head_dim)
        torch.nn.init.orthogonal_(proj)
        return proj

    def kernel_function(self, data):
        """Apply chosen kernel to projected data."""
        projection = self.projection_matrix.to(data.device)
        data = data / math.sqrt(self.head_dim)
        data_dash = torch.einsum("b h s d, f d -> b h s f", data, projection)

        if self.kernel == "relu":
            data_prime = F.relu(data_dash) + 1e-6
        elif self.kernel == "elu":
            data_prime = F.elu(data_dash) + 1.0  # ELU-based mapping
        elif self.kernel == "exp":
            data_dash = data_dash - data_dash.max(dim=-1, keepdim=True).values  # stability
            data_prime = torch.exp(data_dash)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

        return data_prime

    def forward(self, x):
        """Forward pass through Performer layer."""
        residual = x
        x = self.layer_norm1(x)

        b, s, d = x.size()
        q = self.to_q(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

        q_prime = self.kernel_function(q)
        k_prime = self.kernel_function(k)

        kv = torch.einsum("b h s f, b h s d -> b h f d", k_prime, v)
        k_sum = k_prime.sum(dim=2)
        z = 1 / (torch.einsum("b h s f, b h f -> b h s", q_prime, k_sum) + 1e-6)

        out = torch.einsum("b h s f, b h f d -> b h s d", q_prime, kv)
        out = out * z.unsqueeze(-1)

        out = out.transpose(1, 2).contiguous().view(b, s, d)
        out = self.to_out(out)
        out = self.dropout(out)

        out = residual + out
        out = self.layer_norm2(out)
        return out
    

class Performer(nn.Module):
    """
    Full Performer model with modular kernel (ReLU, ELU, Exp).
    """

    def __init__(self, dim, n_heads, depth, dropout, num_classes=10, nb_features=256, kernel="relu"):
        super().__init__()
        self.dim = dim
        self.nb_features = nb_features
        self.kernel = kernel

        # Convolutional backbone
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),
        )

        # Flatten + embedding
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Performer layers with selected kernel
        self.performer_layers = nn.ModuleList([
            PerformerLayer(dim, n_heads, nb_features=nb_features, dropout=dropout, kernel=kernel)
            for _ in range(depth)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        """Forward pass through CNN + Performer + classifier."""
        x = self.conv_layers(x)
        x = self.embedding(x).unsqueeze(1)

        for layer in self.performer_layers:
            x = layer(x)

        x = x.squeeze(1)
        x = self.classifier(x)
        return x