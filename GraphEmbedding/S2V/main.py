from struct2vec import Struc2Vec
#import matplotlib.pyplot as plt
import networkx as nx

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

if __name__ == '__main__':
    # G = nx.Graph()
    # G.add_nodes_from(['x', 'w', 'y', 'i', 'z'])
    # G.add_edges_from([('y', 'z'), ('y', 'w'), ('w', 'z'), ('w', 'x'), ('w', 'i')])
    # # nx.draw(G, with_labels=True, node_color='red')
    # # plt.show()

    G = nx.read_edgelist('data/flight/brazil-airports.edgelist', create_using=nx.DiGraph(), nodetype=None,
                         data=[('weight', int)])  # read graph
   # nx.draw(G, node_size=10, font_size=10, font_color="blue", font_weight="bold")
    # plt.show()


    s2v_model = Struc2Vec(G, 10, 80, workers=4, verbose=40, )  # init model
    s2v_model.train(window_size=5, iter=3)  # train model
    embeddings = s2v_model.get_embeddings()  # get embedding vectors
    # evaluate_embeddings(embeddings)
    # plot_embeddings(embeddings)
    print(embeddings)
