from __future__ import annotations

import torch

from model import EndToEndUnifiedClustering
from utils import make_synthetic_graph, sample_pagerank_anchors, set_seed


def main() -> None:
    set_seed(11)
    graph = make_synthetic_graph(num_nodes=60, num_features=12, num_clusters=3)
    edge_index = graph.edge_index
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    x = graph.x.float()
    discriminator_inputs = x
    anchors = sample_pagerank_anchors(torch.ones(x.size(0)), num_anchors=18)

    indices = edge_index
    values = torch.ones(indices.size(1))
    norm_adj = torch.sparse_coo_tensor(indices, values, (x.size(0), x.size(0))).coalesce()

    model = EndToEndUnifiedClustering(
        feature_dim=x.size(1),
        discriminator_input_dim=discriminator_inputs.size(1),
        hidden_dim=16,
        rank=3,
        discriminator_hidden_dim=32,
        gnn_layers=2,
        dropout=0.1,
        high_pass_alpha=0.5,
        gumbel_temperature=0.7,
        hard_gumbel=False,
        nystrom_ridge=1e-4,
        ranking_weight=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        output = model(x, discriminator_inputs, edge_index, edge_weight, anchors, norm_adj)
        if not torch.isfinite(output.loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {output.loss.item()}")
        output.loss.backward()

        grad_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_norm += parameter.grad.detach().norm().item()
        if grad_norm <= 0:
            raise RuntimeError("No gradient flowed back to model parameters")

        optimizer.step()
        print(
            f"step={step:02d} "
            f"loss={output.loss.item():.6f} "
            f"subspace={output.subspace_loss.item():.6f} "
            f"ranking={output.ranking_loss.item():.6f} "
            f"grad_norm={grad_norm:.6f}"
        )

    with torch.no_grad():
        output = model(x, discriminator_inputs, edge_index, edge_weight, anchors, norm_adj)
        gram = output.basis.t().matmul(output.basis)
        eye = torch.eye(gram.size(0), dtype=gram.dtype)
        orth_error = torch.norm(gram - eye).item()
        print(f"basis_shape={tuple(output.basis.shape)} orthogonality_error={orth_error:.6e}")


if __name__ == "__main__":
    main()
