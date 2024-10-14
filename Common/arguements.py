import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser("environments")
    parser.add_argument("--num_of_node", type=int, default=300, help="number of nodes")
    parser.add_argument("--num_of_egde", type=int, default=90, help="number of egdes")
    parser.add_argument("--degree", type=int, default=20, help="original degree of node")
    parser.add_argument("--add_list", type=int, default=5, help="Scope of the ADD Consultative Mechanism ")
    parser.add_argument("--dlt_list", type=int, default=3, help="Scope of the Delete Consultative Mechanism ")
    parser.add_argument("--add_thresh", type=int, default=0.015, help="Threshold of the Add Consultative Mechanism ")
    parser.add_argument("--dlt_thresh", type=int, default=0.97, help="Threshold of the Delete Consultative Mechanism ")
    parser.add_argument("--step", type=int, default=100, help="Number of steps in episode")
    parser.add_argument("--episode", type=int, default=20, help="Number of episodes")
    parser.add_argument("--ego", type=float, default=2, help="network ego")
    parser.add_argument("--epsilonstart", type=float, default=0.5, help="epsilonstart")
    parser.add_argument("--epsilonend", type=float, default=0.02, help="epsilonend")
    parser.add_argument("--epsilondecay", type=float, default=0.9, help="epsilondecay")
    parser.add_argument("--TARGET_UPDATE_FREQUENCY", type=float, default=10, help="TARGET_UPDATE_FREQUENCY")
    args = parser.parse_args()

    return args