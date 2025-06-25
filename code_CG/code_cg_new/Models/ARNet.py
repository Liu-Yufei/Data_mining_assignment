import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterContrastiveLoss(nn.Module):
    def __init__(self, num_classes, embed_size):
        super(CenterContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.centers = nn.Parameter(torch.randn(num_classes, embed_size))

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        centers_batch = self.centers.index_select(0, labels)

        positive_distances = (embeddings - centers_batch).pow(2).sum(1)
        negative_distances = (embeddings.unsqueeze(1) - self.centers).pow(2).sum(2)

        labels_one_hot = F.one_hot(labels, self.num_classes).float()
        negative_distances = negative_distances * (1 - labels_one_hot)

        negative_distances = negative_distances.min(1)[0]

        loss = positive_distances.mean() + negative_distances.mean()
        return loss
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, embed_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(embed_size, embed_size)
        self.fc2 = nn.Linear(embed_size, embed_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out += identity
        out = self.activation(out)
        return out

class AttentionResidualNetwork(nn.Module):
    def __init__(self, input_size, embed_size, heads, num_classes, num_residual_blocks):
        super(AttentionResidualNetwork, self).__init__()
        self.embed_size = embed_size
        self.attention = MultiHeadAttention(embed_size, heads)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(embed_size) for _ in range(num_residual_blocks)]
        )
        self.fc1 = nn.Linear(input_size, embed_size)
        self.fc2 = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.attention(x, x, x, mask=None)
        x = self.residual_blocks(x)
        x = x.mean(dim=1)  # Global Average Pooling
        embeddings = x
        out = self.fc2(x)
        return out, embeddings

# # Example usage
# input_size = 180
# embed_size = 64
# heads = 8
# num_classes = 2
# num_residual_blocks = 3

# model = AttentionResidualNetwork(input_size, embed_size, heads, num_classes, num_residual_blocks)
# contrastive_loss = CenterContrastiveLoss(num_classes, embed_size)
# criterion = nn.CrossEntropyLoss()

# # Example training loop
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(10):  # Number of epochs
#     for data, labels in train_loader:  # Assuming train_loader is defined
#         optimizer.zero_grad()
#         outputs, embeddings = model(data)
#         loss_cls = criterion(outputs, labels)
#         loss_ctr = contrastive_loss(embeddings, labels)
#         loss = loss_cls + loss_ctr
#         loss.backward()
#         optimizer.step()