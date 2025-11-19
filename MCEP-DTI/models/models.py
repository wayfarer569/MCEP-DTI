import torch.nn as nn
import torch

class AttentionFusionClassifier(nn.Module):
    def __init__(self, input_dims, projection_dim,mlp_dim,num_heads,dropout_rate=0.2):
        super(AttentionFusionClassifier, self).__init__()
        self.input_dims = input_dims
        self.num_features = len(input_dims)
        self.num_heads = num_heads
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, projection_dim), nn.ReLU(),
                          nn.BatchNorm1d(projection_dim)) for dim in input_dims])
        self.attention_query = nn.Linear(projection_dim,projection_dim)
        self.attention_key = nn.Linear(projection_dim,projection_dim)
        self.attention_value = nn.Linear(projection_dim,projection_dim)
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, mlp_dim), nn.ReLU(),
            nn.BatchNorm1d(mlp_dim), nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, mlp_dim // 2), nn.ReLU(),
            nn.BatchNorm1d(mlp_dim // 2), nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim // 2, 1))
    def forward(self, inputs):
        projected = []
        for feat, proj in zip(inputs, self.projections):
            projected.append(proj(feat))
        features = torch.stack(projected, dim=1)
        batch_size = features.size(0)
        Q = self.attention_query(features)
        K = self.attention_key(features)
        V = self.attention_value(features)
        def split_heads(x, num_heads):
            head_dim = x.size(-1) // num_heads
            x = x.view(batch_size, -1, num_heads, head_dim)
            return x.permute(0, 2, 1, 3)
        Q = split_heads(Q, self.num_heads)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (features.size(-1) ** 0.5)  # Scaled dot-product
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_features = torch.matmul(attention_weights, V)
        attended_features = attended_features.permute(0, 2, 1, 3).contiguous()
        attended_features = attended_features.view(batch_size, -1, self.num_heads * (features.size(-1) // self.num_heads))
        fused = torch.sum(attended_features, dim=1)
        logits = self.classifier(fused)
        return logits.squeeze(),fused
