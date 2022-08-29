from torch import nn
import torch


class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.sim = nn.CosineSimilarity()

    def forward(self, queries, documents, eps=1e-8):
        assert (
            len(queries.shape) == 3 and len(documents.shape) == 3
        ), "queries and documents should be in form (batch_size, length, dim)"
        assert queries.shape[-1] == documents.shape[-1], "last dimension must be equal"

        dot = torch.einsum("bqd,bld->bql", (queries, documents))
        Q_norm = torch.linalg.vector_norm(queries, dim=-1, keepdim=True)
        D_norm = torch.linalg.vector_norm(documents, dim=-1, keepdim=True)
        denominator = Q_norm * D_norm.transpose(1, 2) + eps

        return dot / denominator


class MultiRBFKernel(nn.Module):
    def __init__(self, mus, sigmas):
        super().__init__()
        self.register_buffer("mus", mus)
        self.register_buffer("sigmas", sigmas)

    def forward(self, mtx):
        rbf = torch.exp(
            -1
            * torch.pow(mtx.unsqueeze(-1) - self.mus, 2)
            / (2 * torch.pow(self.sigmas, 2))
        )
        return rbf.sum(-2)


class KNRM(nn.Module):
    def __init__(self, num_tokens: int, embedding_dim: int, K: int = 11):
        super().__init__()
        self.register_buffer(
            "mus", torch.arange(-1.0, 1.1, 2 / (K - 1))
        )  # gives mus in range [-1., 1.] spread equally
        self.register_buffer("sigma", torch.tensor(0.1))
        self.emb = nn.Embedding(num_tokens, embedding_dim, padding_idx=0)

        self.sim = CosineSimilarity()
        self.kernels = MultiRBFKernel(self.mus, self.sigma)
        self.linear = nn.Linear(K, 1)
        self.activation = nn.Sigmoid()

    def forward(self, queries, documents):
        queries = self.emb(queries)
        documents = self.emb(documents)
        translation_mtx = self.sim(queries, documents)  # bs, Q, D
        pooled_kernels = self.kernels(translation_mtx)  # bs, Q, K
        soft_tf = torch.log(pooled_kernels).sum(1)  # bs, K
        return self.activation(self.linear(soft_tf))


vocab_size = 10
dim = 8
device = "cuda:0" if torch.cuda.is_available() else "cpu"
net = KNRM(vocab_size, dim, 12).to(device)

bs = 12
queries = torch.Tensor([[1, 2]]).int().to(device)
documents = torch.Tensor([[3, 4, 5]]).int().to(device)
label = torch.Tensor([[1.0]]).to(device)


opt = torch.optim.AdamW(params=net.parameters(), amsgrad=True, lr=1e-2)
criterion = nn.BCELoss()

for i in range(10):

    opt.zero_grad()
    res = net(queries, documents)
    loss = criterion(res, label)
    loss.backward()
    opt.step()
    logit = torch.log(res / (1 - res))

    print(f"\nAfter {i} iter: {res=}, {loss=}, {logit=}")
