"""
Node2Vec graph embedding method
The source code is from the repository of the authors of the paper
"""

# Import Statements

import numpy as np
import networkx as nx
import gensim
from joblib import Parallel, delayed
from collections import defaultdict
import os
import random
import pandas as pd
import time

import NodeClassification
import ArgParser


# -------------------------------------------------------------------------------
# Class Definition


class Node2Vec:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph: nx.Graph, dimensions: int = 128, walk_length: int = 80,
                 num_walks: int = 10,
                 p: float = 1,
                 q: float = 1, weight_key: str = 'weight', workers: int = 10,
                 sampling_strategy: dict = None,
                 quiet: bool = True, temp_folder: str = None):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.

        :param graph: Input graph
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param p: Return hyper parameter (default: 1)
        :param q: Inout parameter (default: 1)
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :param workers: Number of workers for parallel execution (default: 1)
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        """

        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.d_graph = defaultdict(dict)

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError("temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"

        self._precompute_probabilities()
        self.walks = self._generate_walks()

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph

        # nodes_generator = self.graph.nodes() if self.quiet \
        #     else tqdm(self.graph.nodes(), desc='Computing transition probabilities')
        nodes_generator = self.graph.nodes()

        for source in nodes_generator:

            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    elif destination in self.graph[source]:  # If the neighbor is connected to the source
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

                # Save neighbors
                d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors

            # Calculate first_travel weights for source
            first_travel_weights = []

            for destination in self.graph.neighbors(source):
                first_travel_weights.append(self.graph[source][destination].get(self.weight_key, 1))

            first_travel_weights = np.array(first_travel_weights)
            d_graph[source][self.FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()

    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.walk_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks

    def fit(self, **skip_gram_params) -> gensim.models.Word2Vec:
        """
        Creates the embeddings using gensim's Word2Vec.
        :param skip_gram_params: Parameteres for gensim.models.Word2Vec - do not supply 'size' it is taken from the Node2Vec 'dimensions' parameter
        :type skip_gram_params: dict
        :return: A gensim word2vec model
        """

        if 'workers' not in skip_gram_params:
            skip_gram_params['workers'] = self.workers

        if 'size' not in skip_gram_params:
            skip_gram_params['size'] = self.dimensions

        return gensim.models.Word2Vec(self.walks, **skip_gram_params)


def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None,
                            walk_length_key: str = None, neighbors_key: str = None,
                            probabilities_key: str = None, first_travel_key: str = None,
                            quiet: bool = False) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    # if not quiet:
    #     pbar = tqdm(total = num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        # if not quiet:
        #     pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    # if not quiet:
    #     pbar.close()

    return walks


def main():
    """
    instantiate a node2vec object
    """
    print("Node2Vec main method.")
    start_time = time.time()

    args = ArgParser.parse_args()

    main_path = args.main_path
    graph_name = args.graph

    iter_num = args.iter
    num_walks = args.num_walks
    dim = args.dimensions
    walk_length = args.walk_length
    workers = args.workers
    window_size = args.window_size
    p = args.p
    q = args.q

    data_path = main_path + '/inputs/'
    edge_list_file_name = data_path + graph_name + '.edgelist'
    node_list_file_name = data_path + 'nodes_' + graph_name + '.csv'
    output_path = main_path + '/outputs/'
    stats_file = output_path + 'stats.csv'

    nx_g = nx.from_pandas_edgelist(pd.read_csv(edge_list_file_name), source='source', target='target',
                                   create_using=nx.DiGraph())
    print("Graph info:", nx.info(nx_g))
    # for edge in nx_g.edges():
    #     nx_g[edge[0]][edge[1]]['weight'] = 1
    # nx_g = nx_g.to_undirected()

    print("\tInstantiate a node2vec object.")
    node2vec = Node2Vec(nx_g, dimensions=dim, walk_length=walk_length,
                        num_walks=num_walks, workers=workers, p=p, q=q)
    print("\tFit node2vec.")
    model = node2vec.fit(window=window_size, sg=1, hs=0, min_count=1, iter=iter_num)

    # read node list
    print("\tExtract embeddings and labels for the anchor nodes.")
    nodes_df = pd.read_csv(node_list_file_name)

    # binary classification anchor nodes
    anchor_nodes_df = nodes_df.loc[nodes_df['is_anchor'] == 1, ['node', 'isp']]
    node_list = [str(node_id) for node_id in anchor_nodes_df['node'].tolist()]
    embeddings = [model.wv.get_vector(node) for node in node_list]
    labels = anchor_nodes_df['isp'].tolist()


    # classification
    print("\tApply classification.")
    rnd_seed = 42
    binary = True
    X_train, X_test, y_train, y_test = NodeClassification.train_test_split(embeddings, labels, rnd_seed)
    NodeClassification.rf_lr_classification(X_train, X_test, y_train, y_test, stats_file, 'n2v',
                                            binary=binary, print_report=True)
    print("Total elapsed time:", str(time.time() - start_time))


if __name__ == '__main__':
    main()
