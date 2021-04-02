"""
Shared argument parser for different files
"""
import argparse


def parse_args():
    """
        Parses the arguments.
        NOTE: Shared between "GraphPrep.py", "FeatureEngineering.py", and "NodeClassification.py"
    """
    parser = argparse.ArgumentParser(description="Run program.")

    parser.add_argument('--input', nargs='?', default='', help='Input main path.')
    parser.add_argument('--graph', nargs='?', default='eth_0', help='Input partial file name.')
    parser.add_argument('--output', nargs='?', default='', help='Embedding file name.')

    parser.add_argument('--flag', nargs='?', default='sp', help='Input flag')
    parser.add_argument('--clf-opt', nargs='?', default='fe', help='Input classification option.')
    parser.add_argument('--exp-id', nargs='?', help='Experiment identifier.')

    parser.add_argument('--gprep-opt', nargs='?', default='feature', help='Graph preparation option.')

    # Node2Vec.py
    parser.add_argument('--dimensions', type=int, default=64, help='Number of dimensions. Default is 64.')
    parser.add_argument('--walk-length', type=int, default=5, help='Length of walk per source. Default is 5.')
    parser.add_argument('--num-walks', type=int, default=20, help='Number of walks per source. Default is 20.')
    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')
    parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers. Default is 4.')
    parser.add_argument('--iter', default=5, type=int, help='Number of epochs in SGD. Default is 5.')
    parser.add_argument('--p', default=1, type=float, help='p for biased random walk in Node2Vec.')
    parser.add_argument('--q', default=1, type=float, help='q for biased random walk in Node2Vec.')


    return parser.parse_args()

