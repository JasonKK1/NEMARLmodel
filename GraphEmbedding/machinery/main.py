"""Running the GraphWave machine."""

import pandas as pd
import networkx as nx
from texttable import Texttable
import sys
import igraph as ig
sys.path.append('../..')

from param_parser import parameter_parser
from spectral_machinery import WaveletMachine

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    tab = Texttable()
    tab.add_rows([["Parameter", "Value"]])
    tab.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    # print(tab.draw())

def read_graph(settings):
    """
    Reading the edge list from the path and returning the networkx graph object.
    :param path: Path to the edge list.
    :return graph: Graph from edge list.
    """
    if settings.edgelist_input:
        graph = nx.read_edgelist(settings.input)
    else:
        edge_list = pd.read_csv(settings.input).values.tolist()
        graph = nx.from_edgelist(edge_list)
        graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph

def generate_ring_lattice(n, k):
    G = nx.generators.random_graphs.watts_strogatz_graph(n, k, 0)
    return G
if __name__ == "__main__":
    settings = parameter_parser()
    tab_printer(settings)
    # G = read_graph(settings)
    # n = 300  # Number of nodes
    # k = 2  # Degree of each node
    # G = generate_ring_lattice(n, k)
    g = ig.Graph.Read_GML('/home/admin1/Jason/GraphEmbedding/GraphWaveMachine-master/data/node300l5epi150.gml')
    G = g.to_networkx()
    machine = WaveletMachine(G, settings)

    machine.create_embedding()
    machine.transform_and_save_embedding()
