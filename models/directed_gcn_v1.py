import torch
import torch.nn as nn
import torch.nn.functional as F


# Directed GCN Layer
class DirectedGCNLayer(nn.Module):
    """
    Direction-aware graph convolution layer.
    Computes separate incoming and outgoing aggregations
    and combines them.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, out_dim)
        self.lin_out = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, num_nodes):
        src, dst = edge_index

        # Incoming aggregation
        deg_in = torch.zeros(num_nodes, device=x.device)
        deg_in.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg_in[deg_in == 0] = 1

        agg_in = torch.zeros_like(x)
        agg_in.index_add_(0, dst, x[src])
        agg_in = agg_in / deg_in.unsqueeze(1)

        # Outgoing aggregation
        deg_out = torch.zeros(num_nodes, device=x.device)
        deg_out.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        deg_out[deg_out == 0] = 1

        agg_out = torch.zeros_like(x)
        agg_out.index_add_(0, src, x[dst])
        agg_out = agg_out / deg_out.unsqueeze(1)

        return self.lin_in(agg_in) + self.lin_out(agg_out)


# Full Directed GCN
class DirectedGCN(nn.Module):
    """
    Two-layer Directed GCN (Full model).
    Uses both incoming and outgoing message passing.
    """

    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.layer1 = DirectedGCNLayer(input_dim, hidden_dim)
        self.layer2 = DirectedGCNLayer(hidden_dim, num_classes)

    def forward(self, data):
        x = F.relu(self.layer1(data.x, data.edge_index, data.num_nodes))
        return self.layer2(x, data.edge_index, data.num_nodes)


# Incoming-Only Variant
class IncomingOnlyLayer(DirectedGCNLayer):
    """
    Uses only incoming aggregation.
    """

    def forward(self, x, edge_index, num_nodes):
        src, dst = edge_index

        deg_in = torch.zeros(num_nodes, device=x.device)
        deg_in.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        deg_in[deg_in == 0] = 1

        agg_in = torch.zeros_like(x)
        agg_in.index_add_(0, dst, x[src])
        agg_in = agg_in / deg_in.unsqueeze(1)

        return self.lin_in(agg_in)


class IncomingGCN(nn.Module):
    """
    Two-layer Incoming-Only Directed GCN.
    """

    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.layer1 = IncomingOnlyLayer(input_dim, hidden_dim)
        self.layer2 = IncomingOnlyLayer(hidden_dim, num_classes)

    def forward(self, data):
        x = F.relu(self.layer1(data.x, data.edge_index, data.num_nodes))
        return self.layer2(x, data.edge_index, data.num_nodes)


# Outgoing-Only Variant
class OutgoingOnlyLayer(DirectedGCNLayer):
    """
    Uses only outgoing aggregation.
    """

    def forward(self, x, edge_index, num_nodes):
        src, dst = edge_index

        deg_out = torch.zeros(num_nodes, device=x.device)
        deg_out.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        deg_out[deg_out == 0] = 1

        agg_out = torch.zeros_like(x)
        agg_out.index_add_(0, src, x[dst])
        agg_out = agg_out / deg_out.unsqueeze(1)

        return self.lin_out(agg_out)


class OutgoingGCN(nn.Module):
    """
    Two-layer Outgoing-Only Directed GCN.
    """

    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.layer1 = OutgoingOnlyLayer(input_dim, hidden_dim)
        self.layer2 = OutgoingOnlyLayer(hidden_dim, num_classes)

    def forward(self, data):
        x = F.relu(self.layer1(data.x, data.edge_index, data.num_nodes))
        return self.layer2(x, data.edge_index, data.num_nodes)