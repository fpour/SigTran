"""
generate the designated features for the nodes of the graph
"""
import datetime
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import entropy

import FeatureSelection
import ArgParser


class NodeEngFeatures:
    def __init__(self, nodes, edges):
        self.nodes = nodes  # a dataframe
        self.edges = edges  # a dataframe
        self.G = nx.from_pandas_edgelist(self.edges, source='source', target='target',
                                         edge_attr=['timestamp', 'amount'],
                                         create_using=nx.MultiDiGraph())
        print("*** Original MD-Graph ***")
        print(nx.info(self.G))
        self.node_feature_names = self.retrieve_feature_name()

    def retrieve_feature_name(self):
        """
        retrieve the names of the features for the nodes
        """
        feature_stat_df = FeatureSelection.FeatureStatus().feature_stat
        feature_name_list = feature_stat_df['feature'].tolist()
        return feature_name_list

    def get_tx_amount_and_interval_list(self, node, opt):
        """
        returns the list of amount and the list of timestamps for all the (opt-) transactions
        :param node: the node that we focus on
        :param opt: 'in', 'out', or 'all' transactions
        """
        if opt == 'in':  # incoming tx
            node_tx_df = self.edges[self.edges['target'] == node]
        elif opt == 'out':  # outgoing tx
            node_tx_df = self.edges[self.edges['source'] == node]
        elif opt == 'all':  # all tx
            node_tx_df = self.edges[(self.edges['target'] == node) | (self.edges['source'] == node)]
        else:
            raise ValueError("Option unavailable!")

        amount_list = node_tx_df['amount'].tolist()
        linux_timestamp_list = node_tx_df['timestamp'].tolist()
        timestamp_list = [datetime.datetime.fromtimestamp(t) for t in linux_timestamp_list]
        timestamp_list.sort()
        # interval of txs in minutes
        tx_interval = [((timestamp_list[i + 1] - timestamp_list[i]).total_seconds() / 60) for i
                       in range(len(timestamp_list) - 1)]

        return amount_list, tx_interval

    def neighbor_degree_features(self, node):
        """
        get the features related to the degree distributions of the neighbors of the node
        """
        # extract the egonet of the node
        egonet = nx.ego_graph(self.G, node)

        # prerequisite for some neighborhood features
        egonet_node = nx.nodes(egonet)
        no_edge_egonet_in = 0  # number of in-coming edges to egonet
        no_edge_egonet_out = 0  # number of out-going edges from egonet
        for nb_node in egonet_node:
            if node != nb_node:
                no_edge_egonet_in += (self.G.in_degree[nb_node] - egonet.in_degree[nb_node])
                no_edge_egonet_out += (self.G.out_degree[nb_node] - egonet.out_degree[nb_node])

        neighbor_degrees = [d for n, d in egonet.degree() if n != node]
        neighbor_w_degrees = [d for n, d in egonet.degree(weight='amount') if n != node]
        neighbor_in_degrees = [d for n, d in egonet.in_degree() if n != node]
        neighbor_out_degrees = [d for n, d in egonet.out_degree() if n != node]

        no_edge_egonet = egonet.number_of_edges()

        return no_edge_egonet, no_edge_egonet_in, no_edge_egonet_out, \
               neighbor_degrees, neighbor_w_degrees, neighbor_in_degrees, neighbor_out_degrees

    def gen_node_features_single(self, nodnodee):
        """
        generate the features for the node
        :param node: node of interest
        """
        no_edge_egonet, no_edge_egonet_in, no_edge_egonet_out, \
            neighbor_degrees, neighbor_w_degrees, neighbor_in_degrees, neighbor_out_degrees = \
            self.neighbor_degree_features(node)
        amnt_in_list, interval_in_tx = self.get_tx_amount_and_interval_list(node, 'in')
        amnt_out_list, interval_out_tx = self.get_tx_amount_and_interval_list(node, 'out')
        amnt_all_list, interval_all_tx = self.get_tx_amount_and_interval_list(node, 'all')
        node_row = self.nodes.loc[self.nodes['node'] == node]
        node_feature_dict = {
            'node': node_row['node'].values[0],
            'address': node_row['address'].values[0],
            'isp': node_row['isp'].values[0],
            'is_anchor': node_row['is_anchor'].values[0],
            # 'balance': node_row['balance'].values[0],

            # structural
            'degree': len(amnt_all_list),
            # 'w_degree': self.G.degree(node, weight='amount'),
            'in_degree': len(amnt_in_list),
            'out_degree': len(amnt_out_list),

            # transactional
            'avg_amount_in_tx': np.mean(amnt_in_list) if len(amnt_in_list) > 0 else 0,
            'min_amount_in_tx': np.min(amnt_in_list) if len(amnt_in_list) > 0 else 0,
            'max_amount_in_tx': np.max(amnt_in_list) if len(amnt_in_list) > 0 else 0,
            'sum_amount_in_tx': np.sum(amnt_in_list),
            'std_amount_in_tx': np.std(amnt_in_list) if len(amnt_in_list) > 0 else 0,
            'ent_amount_in_tx': entropy(amnt_in_list) if np.sum(amnt_in_list) != 0 else 0,

            'avg_in_tx_interval': np.mean(interval_in_tx) if len(interval_in_tx) > 0 else 0,
            'min_in_tx_interval': np.min(interval_in_tx) if len(interval_in_tx) > 0 else 0,
            'max_in_tx_interval': np.max(interval_in_tx) if len(interval_in_tx) > 0 else 0,
            'sum_in_tx_interval': np.sum(interval_in_tx),
            'std_in_tx_interval': np.std(interval_in_tx) if len(interval_in_tx) > 0 else 0,
            'ent_in_tx_interval': entropy(interval_in_tx) if np.sum(interval_in_tx) != 0 else 0,

            'avg_amount_out_tx': np.mean(amnt_out_list) if len(amnt_out_list) > 0 else 0,
            'min_amount_out_tx': np.min(amnt_out_list) if len(amnt_out_list) > 0 else 0,
            'max_amount_out_tx': np.max(amnt_out_list) if len(amnt_out_list) > 0 else 0,
            'sum_amount_out_tx': np.sum(amnt_out_list),
            'std_amount_out_tx': np.std(amnt_out_list) if len(amnt_out_list) > 0 else 0,
            'ent_amount_out_tx': entropy(amnt_out_list) if np.sum(amnt_out_list) != 0 else 0,

            'avg_out_tx_interval': np.mean(interval_out_tx) if len(interval_out_tx) > 0 else 0,
            'min_out_tx_interval': np.min(interval_out_tx) if len(interval_out_tx) > 0 else 0,
            'max_out_tx_interval': np.max(interval_out_tx) if len(interval_out_tx) > 0 else 0,
            'sum_out_tx_interval': np.sum(interval_out_tx),
            'std_out_tx_interval': np.std(interval_out_tx) if len(interval_out_tx) > 0 else 0,
            'ent_out_tx_interval': entropy(interval_out_tx) if np.sum(interval_out_tx) != 0 else 0,

            'avg_amount_all_tx': np.mean(amnt_all_list) if len(amnt_all_list) > 0 else 0,  # all tx: in & out
            'min_amount_all_tx': np.min(amnt_all_list) if len(amnt_all_list) > 0 else 0,
            'max_amount_all_tx': np.max(amnt_all_list) if len(amnt_all_list) > 0 else 0,
            'sum_amount_all_tx': np.sum(amnt_all_list),  # this should be equal to weighted degree
            'std_amount_all_tx': np.std(amnt_all_list) if len(amnt_all_list) > 0 else 0,
            'ent_amount_all_tx': entropy(amnt_all_list) if np.sum(amnt_all_list) != 0 else 0,

            'avg_all_tx_interval': np.mean(interval_all_tx) if len(interval_all_tx) > 0 else 0,
            'min_all_tx_interval': np.min(interval_all_tx) if len(interval_all_tx) > 0 else 0,
            'max_all_tx_interval': np.max(interval_all_tx) if len(interval_all_tx) > 0 else 0,
            'sum_all_tx_interval': np.sum(interval_all_tx),
            'std_all_tx_interval': np.std(interval_all_tx) if len(interval_all_tx) > 0 else 0,
            'ent_all_tx_interval': entropy(interval_all_tx) if np.sum(interval_all_tx) != 0 else 0,

            # regional features
            'no_edge_within_egonet': no_edge_egonet,  # number of edges within the egonet for all nodes
            'no_edge_in_egonet': no_edge_egonet_in,  # number of in-edges to the egonet
            'no_edge_out_egonet': no_edge_egonet_out,  # number of out-edges from the egonet
            'no_edge_all_egonet': no_edge_egonet_in + no_edge_egonet_out,  # total number of edges to/from the egonet

            # neighborhood features
            'avg_neighbor_degree': np.mean(neighbor_degrees) if len(neighbor_degrees) > 0 else 0,
            'min_neighbor_degree': np.min(neighbor_degrees) if len(neighbor_degrees) > 0 else 0,
            'max_neighbor_degree': np.max(neighbor_degrees) if len(neighbor_degrees) > 0 else 0,
            'sum_neighbor_degree': np.sum(neighbor_degrees),
            'std_neighbor_degree': np.std(neighbor_degrees) if len(neighbor_degrees) > 0 else 0,
            'ent_neighbor_degree': entropy(neighbor_degrees) if np.sum(neighbor_degrees) != 0 else 0,

            'avg_neighbor_w_degree': np.mean(neighbor_w_degrees) if len(neighbor_w_degrees) > 0 else 0,
            'min_neighbor_w_degree': np.min(neighbor_w_degrees) if len(neighbor_w_degrees) > 0 else 0,
            'max_neighbor_w_degree': np.max(neighbor_w_degrees) if len(neighbor_w_degrees) > 0 else 0,
            'sum_neighbor_w_degree': np.sum(neighbor_w_degrees),
            'std_neighbor_w_degree': np.std(neighbor_w_degrees) if len(neighbor_w_degrees) > 0 else 0,
            'ent_neighbor_w_degree': entropy(neighbor_w_degrees) if np.sum(neighbor_w_degrees) != 0 else 0,

            'avg_neighbor_in_degree': np.mean(neighbor_in_degrees) if len(neighbor_in_degrees) > 0 else 0,
            'min_neighbor_in_degree': np.min(neighbor_in_degrees) if len(neighbor_in_degrees) > 0 else 0,
            'max_neighbor_in_degree': np.max(neighbor_in_degrees) if len(neighbor_in_degrees) > 0 else 0,
            'sum_neighbor_in_degree': np.sum(neighbor_in_degrees),
            'std_neighbor_in_degree': np.std(neighbor_in_degrees) if len(neighbor_in_degrees) > 0 else 0,
            'ent_neighbor_in_degree': entropy(neighbor_in_degrees) if np.sum(neighbor_in_degrees) != 0 else 0,

            'avg_neighbor_out_degree': np.mean(neighbor_out_degrees) if len(neighbor_out_degrees) > 0 else 0,
            'min_neighbor_out_degree': np.min(neighbor_out_degrees) if len(neighbor_out_degrees) > 0 else 0,
            'max_neighbor_out_degree': np.max(neighbor_out_degrees) if len(neighbor_out_degrees) > 0 else 0,
            'sum_neighbor_out_degree': np.sum(neighbor_out_degrees),
            'std_neighbor_out_degree': np.std(neighbor_out_degrees) if len(neighbor_out_degrees) > 0 else 0,
            'ent_neighbor_out_degree': entropy(neighbor_out_degrees) if np.sum(neighbor_out_degrees) != 0 else 0,

        }
        return node_feature_dict

    def gen_node_features_list(self, node_list):
        """
        generate features for each node in the node_list
        :param node_list: a list of different nodes
        :return node_feature_df: a dataframe of the nodes and their features
        """
        node_features_dict_list = [self.gen_node_features_single(node) for node in node_list]
        node_feature_df = pd.DataFrame(node_features_dict_list, columns=self.node_feature_names)
        return node_feature_df




def main():

    # test the performance of the engineered features
    # edges list and nodes list file
    args = ArgParser.parse_args()

    main_path = args.main_path + 'inputs/'
    pfilename = args.graph
    node_filename = main_path + 'nodes_' + pfilename + '.csv'
    edge_filename = main_path + 'edges_' + pfilename + '.csv'

    print('Read edge list and node list.')
    edges_df = pd.read_csv(edge_filename)
    nodes_df = pd.read_csv(node_filename)

    # get anchor nodes list
    print('Retrieve anchor nodes list.')
    anchor_nodes = nodes_df.loc[nodes_df['is_anchor'] == 1]['node'].tolist()

    # --- generate features for all anchor nodes of a graph ---
    print('Generate features for the anchor nodes.')
    n_eng_features = NodeEngFeatures(nodes_df, edges_df)
    node_feature_df = n_eng_features.gen_node_features_list(anchor_nodes)

    # save anchor nodes features to file
    print(node_feature_df.head())
    print('Save node features dataframe.')
    node_features_filename = main_path + 'features_eth_0.csv'
    node_feature_df.to_csv(node_features_filename, index=False)


if __name__ == '__main__':
    main()




