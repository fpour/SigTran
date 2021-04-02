# -*- coding: utf-8 -*-
"""
Based on the implementation of the following paper:
"RiWalk: Fast Structural Node Embedding via Role Identification", ICDM, 2019
"""

import argparse
import json
import time
import Graph
from gensim.models import Word2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
import networkx as nx
import os
import glob
import logging
import sys

import pandas as pd

import numpy as np
from collections import defaultdict


seed = 42


def debug(type_, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback
        import pdb
        traceback.print_exception(type_, value, tb)
        print(u"\n")
        pdb.pm()


def parse_args():
    """
        Parses the RiWalk arguments.
    """
    parser = argparse.ArgumentParser(description="Run RiWalk")

    parser.add_argument('--input', nargs='?', default='graphs/blockchain/simp_orig_riwalk_eth_0.edgelist',
                        help='Input graph path')

    parser.add_argument('--graph', nargs='?', default='graphs/blockchain/simp_orig_riwalk_eth_0.nodelist',
                        help='Input node list path')

    parser.add_argument('--output', nargs='?', default='embs/blockchain/simp_orig_riwalk_eth_0.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=80,
                        help='Number of walks per source. Default is 80.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--until-k', type=int, default=4,
                        help='Neighborhood size k. Default is 4.')

    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD. Default is 5.')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers. Default is 4.')

    parser.add_argument('--flag', nargs='?', default='sp',
                        help='Flag indicating using RiWalk-SP(sp) or RiWalk-WL(wl). Default is sp.')

    parser.add_argument('--without-discount', action='store_true', default=False,
                        help='Flag indicating not using discount.')

    parser.add_argument('--p', type=float, default=1, help='p value for Node2Vec biased walk generation.')

    parser.add_argument('--q', type=float, default=1, help='q value for Node2Vec biased walk generation.')

    parser.add_argument('--weight-key', nargs='?', default='weight', help='edge weight key.')

    parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                        help="drop a debugger if an exception is raised.")

    parser.add_argument("-l", "--log", dest="log", default="DEBUG",
                        help="Log verbosity level. Default is DEBUG.")

    return parser.parse_args()


class Sentences(object):
    """
    a wrapper of random walk files to feed to word2vec
    """

    def __init__(self, file_names):
        self.file_names = file_names

    def __iter__(self):
        fs = []
        for file_name in self.file_names:
            fs.append(open(file_name))
        while True:
            flag = 0
            for i, f in enumerate(fs):
                line = f.readline()
                if line != '':
                    flag = 1
                    yield line.split()
            if not flag:
                try:
                    for f in fs:
                        f.close()
                except:
                    pass
                return


class RiWalk:
    def __init__(self, args):
        self.args = args
        os.system('rm -rf walks/__random_walks_*.txt')

    def learn_embeddings(self):
        """
        learn embeddings from random walks.
        hs:  0:negative sampling 1:hierarchical softmax
        sg:  0:CBOW              1:skip-gram
        """
        dim = self.args.dimensions
        window_size = self.args.window_size
        workers = self.args.workers
        iter_num = self.args.iter

        logging.debug('begin learning embeddings')
        learning_begin_time = time.time()

        walk_files = glob.glob('walks/__random_walks_*.txt')
        sentences = Sentences(walk_files)
        model = Word2Vec(sentences, size=dim, window=window_size, min_count=0, sg=1, hs=0, workers=workers,
                         iter=iter_num)

        learning_end_time = time.time()
        logging.debug('done learning embeddings')
        logging.debug('learning_time: {}'.format(learning_end_time - learning_begin_time))
        print('learning_time', learning_end_time - learning_begin_time, flush=True)
        return model.wv

    def read_graph(self):
        logging.debug('begin reading graph')
        read_begin_time = time.time()

        input_file_name = self.args.input + 'edges_' + self.args.graph + '.edgelist'
        # nx_g = nx.read_edgelist(input_file_name, nodetype=int, create_using=nx.DiGraph())
        nx_g = nx.from_pandas_edgelist(pd.read_csv(input_file_name), source='source', target='target',
                                       edge_attr=['weight'], create_using=nx.DiGraph())
        print("*** RiWalk Graph ***")
        print(nx.info(nx_g))

        # TODO: attention!!!
        # for edge in nx_g.edges():
        #     nx_g[edge[0]][edge[1]]['weight'] = 1
        # nx_g = nx_g.to_undirected()

        logging.debug('done reading graph')
        read_end_time = time.time()
        logging.debug('read time: {}'.format(read_end_time - read_begin_time))
        return nx_g

    def read_node_attribute(self):
        """
        read the node list and extract the node attributes along the node id
        :param node_attr_names: the name of the attributes to retrieve
        """
        logging.debug('begin reading node list')
        read_begin_time = time.time()

        node_list_file_name = self.args.input + 'nodes_' + self.args.graph + '.nodelist'
        node_df = pd.read_csv(node_list_file_name)
        # node_df_column_names = node_df.columns.to_list
        # node_df_column_names.remove('node')
        # node_df = node_df[node_df_column_names]
        # node_attr_dict = pd.Series(node_list_df['attribute'].values, index=node_list_df['node']).to_dict()

        logging.debug('done reading node list')
        read_end_time = time.time()
        logging.debug('read time: {}'.format(read_end_time - read_begin_time))
        return node_df

    def preprocess_graph(self, nx_g, node_attr_df):
        """
        1. relabel nodes with 0,1,2,3,...,N.
        2. convert graph to adjacency representation as a list of lists.
        """
        debug = logging.debug('begin preprocessing graph')
        preprocess_begin_time = time.time()

        mapping = {_: i for i, _ in enumerate(nx_g.nodes())}
        nx_g = nx.relabel_nodes(nx_g, mapping)

        # generate a node2vec graph instance
        n2v_g = N2VGraph(nx_g, p=self.args.p, q=self.args.q, weight_key=self.args.weight_key)

        nx_g = [list(nx_g.neighbors(_)) for _ in range(len(nx_g))]

        # processing the nodes attributes if flag == 'na'
        if self.args.flag == 'na':
            # inverse mapping
            inv_mapping = {v: k for k, v in mapping.items()}
            # retrieve nodes attributes
            selected_features_list = node_attr_df.columns.to_list()
            selected_features_list.remove('node')  # remove the node id column
            print("\t>>> NA:", len(selected_features_list), " features selected:\n", selected_features_list)
            node_attr_list_orig = [node_attr_df.loc[node_attr_df['node'] == inv_mapping[i],
                                   selected_features_list].values.tolist() for i in range(len(nx_g))]
            node_attr_list = [item for sublist in node_attr_list_orig for item in sublist]
        else:
            node_attr_list = []  # for cases other than 'na', no need to have attribute list
        # ---

        logging.info('#nodes: {}'.format(len(nx_g)))
        logging.info('#edges: {}'.format(sum([len(_) for _ in nx_g]) // 2))

        logging.debug('done preprocessing')
        logging.debug('preprocess time: {}'.format(time.time() - preprocess_begin_time))
        return nx_g, n2v_g, mapping, node_attr_list

    def learn(self, nx_g, n2v_g, mapping, node_attr):
        g = RiWalkGraph.RiGraph(nx_g, n2v_g, node_attr, self.args)

        walk_time, bfs_time, ri_time, walks_writing_time = g.process_random_walks()

        print('walk_time', walk_time / self.args.workers, flush=True)
        print('bfs_time', bfs_time / self.args.workers, flush=True)
        print('ri_time', ri_time / self.args.workers, flush=True)
        print('walks_writing_time', walks_writing_time / self.args.workers, flush=True)

        wv = self.learn_embeddings()

        original_wv = Word2VecKeyedVectors(self.args.dimensions)
        original_nodes = list(mapping.keys())
        original_vecs = [wv.word_vec(str(mapping[node])) for node in original_nodes]
        original_wv.add(entities=list(map(str, original_nodes)), weights=original_vecs)
        return original_wv

    def riwalk(self):
        nx_g = self.read_graph()
        if self.args.flag == 'na':
            node_attr_df = self.read_node_attribute()  # a dataframe
        else:
            node_attr_df = pd.DataFrame()  # empty dataframe
        read_end_time = time.time()
        nx_g, n2v_g, mapping, node_attr = self.preprocess_graph(nx_g, node_attr_df)
        wv = self.learn(nx_g, n2v_g, mapping, node_attr)
        return wv, time.time() - read_end_time


class N2VGraph:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph: nx.Graph, p: float = 1, q: float = 1, weight_key: str = 'weight',
                 sampling_strategy: dict = None, temp_folder: str = None):
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
        self.p = p
        self.q = q
        self.weight_key = weight_key
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
                if unnormalized_weights.sum() != 0:
                    d_graph[current_node][self.PROBABILITIES_KEY][
                        source] = unnormalized_weights / unnormalized_weights.sum()
                elif len(unnormalized_weights) != 0:
                    d_graph[current_node][self.PROBABILITIES_KEY][source] = [1/len(unnormalized_weights)] * len(unnormalized_weights)
                else:
                    # when current node has no successor in DiGraph
                    d_graph[current_node][self.PROBABILITIES_KEY][source] = [0]

                # Save neighbors
                d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors

            # Calculate first_travel weights for source
            first_travel_weights = []

            for destination in self.graph.neighbors(source):
                first_travel_weights.append(self.graph[source][destination].get(self.weight_key, 1))

            first_travel_weights = np.array(first_travel_weights)
            if first_travel_weights.sum() != 0:
                d_graph[source][self.FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()
            elif len(first_travel_weights) != 0:
                d_graph[source][self.FIRST_TRAVEL_KEY] = [1 / len(first_travel_weights)] * len(first_travel_weights)
            else:
                # when source has no successor in DiGraph
                d_graph[source][self.FIRST_TRAVEL_KEY] = [0]


def main():
    args = parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

    os.system('rm -f RiWalk.log')
    logging.basicConfig(filename='RiWalk.log', level=numeric_level, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    logging.info(str(vars(args)))
    if args.debug:
        sys.excepthook = debug

    wv, total_time = RiWalk(args).riwalk()

    write_begin_time = time.time()
    wv.save_word2vec_format(fname=args.output, binary=False)
    logging.debug('embedding_writing_time: {}'.format(time.time() - write_begin_time))

    json.dump({'time': total_time}, open(args.output.replace('.emb', '_time.json'), 'w'))


if __name__ == '__main__':
    main()
