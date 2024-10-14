import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

def generate_graph_embedding(G, num_layers=2, hidden_channels=16, out_channels=32):
    data = from_networkx(G)
    if data.edge_index is None:
        raise ValueError("The input graph does not contain edge indices.")

    # Initialize node embeddings using random values
    data.x = torch.randn(data.num_nodes, hidden_channels)

    # GraphSAGE inner class
    class SAGE(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
            super(SAGE, self).__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            for conv in self.convs:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
            return x

    model = SAGE(in_channels=hidden_channels,
                 hidden_channels=hidden_channels,
                 out_channels=out_channels,
                 num_layers=num_layers)

    model.eval()
    with torch.no_grad():
        embedding = model(data)

    return embedding
G = nx.erdos_renyi_graph(100,0.01)
print(generate_graph_embedding(G))