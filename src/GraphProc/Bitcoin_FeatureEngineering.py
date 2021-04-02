"""
generate the designated features for the nodes of the Bitcoin graph
"""

import datetime
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import entropy

import ArgParser
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class BitcoinNodeEngFeatures:
    def __init__(self, nodes, edges):
        self.nodes = nodes  # a dataframe
        self.edges = edges  # a dataframe
        self.G = nx.from_pandas_edgelist(self.edges, source='source', target='target',
                                         create_using=nx.DiGraph())
        print("*** Original DiGraph ***")
        print(nx.info(self.G))

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
            if node != nb_node:  # not anchor
                no_edge_egonet_in += (self.G.in_degree[node] - egonet.in_degree[node])
                no_edge_egonet_out += (self.G.out_degree[node] - egonet.out_degree[node])

        neighbor_degrees = [d for n, d in egonet.degree() if n != node]
        neighbor_in_degrees = [d for n, d in egonet.in_degree() if n != node]
        neighbor_out_degrees = [d for n, d in egonet.out_degree() if n != node]

        no_edge_egonet = egonet.number_of_edges()

        return no_edge_egonet, no_edge_egonet_in, no_edge_egonet_out, \
               neighbor_degrees, neighbor_in_degrees, neighbor_out_degrees

    def gen_node_features_single(self, node):
        """
        generate the features for the node
        :param node: node of interest
        """
        no_edge_egonet, no_edge_egonet_in, no_edge_egonet_out, \
        neighbor_degrees, neighbor_in_degrees, neighbor_out_degrees = self.neighbor_degree_features(node)
        node_row = self.nodes.loc[self.nodes['node'] == node]
        node_feature_dict = {
            'node': node_row['node'].values[0],
            'address': node_row['address'].values[0],
            'isp': node_row['isp'].values[0],
            'is_anchor': node_row['is_anchor'].values[0],

            'degree': self.G.degree(node),
            'in_degree': self.G.in_degree(node),
            'out_degree': self.G.out_degree(node),

            # neighborhood features
            'no_edge_within_egonet': no_edge_egonet,  # number of edges within the egonet for all nodes
            'no_edge_in_egonet': no_edge_egonet_in,  # number of in-edges to the egonet
            'no_edge_out_egonet': no_edge_egonet_out,  # number of out-edges from the egonet
            'no_edge_all_egonet': no_edge_egonet_in + no_edge_egonet_out,
            # total number of edges to/from the egonet

            # regional features
            'avg_neighbor_degree': np.mean(neighbor_degrees) if len(neighbor_degrees) > 0 else 0,
            'min_neighbor_degree': np.min(neighbor_degrees) if len(neighbor_degrees) > 0 else 0,
            'max_neighbor_degree': np.max(neighbor_degrees) if len(neighbor_degrees) > 0 else 0,
            'sum_neighbor_degree': np.sum(neighbor_degrees),
            'std_neighbor_degree': np.std(neighbor_degrees) if len(neighbor_degrees) > 0 else 0,
            'ent_neighbor_degree': entropy(neighbor_degrees) if np.sum(neighbor_degrees) != 0 else 0,

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
        node_feature_names = list(node_features_dict_list[0].keys())
        node_feature_df = pd.DataFrame(node_features_dict_list, columns=node_feature_names)
        return node_feature_df

    def gen_merged_ell_eng_features(self, elliptic_features_df, engineered_features_df):
        """
        retrieve the elliptic features for the nodes whose engineered features are constructed
        """
        node_list = engineered_features_df['address'].tolist()  # the main tx ID
        node_elliptic_features = elliptic_features_df.loc[elliptic_features_df['address'].isin(node_list)]

        merged_df = pd.merge(left=engineered_features_df, right=node_elliptic_features,
                             how='left', left_on='address', right_on='address')
        return node_elliptic_features, merged_df

    def gen_all_features_all_nodes(self, elliptic_features_df):
        """
        get a dataframe containing the selected features for all nodes
        """
        node_list = self.nodes['node'].tolist()  # select all the nodes
        eng_feature_all_node_df = self.gen_node_features_list(node_list)

        # scale the features; note that it should be fitted on the train set ONLY
        column_names = eng_feature_all_node_df.columns
        not_scale_col = ['node', 'address', 'isp', 'is_anchor', 'timestamp', 'txId', 'label']
        to_scale_col = [col for col in column_names if col not in not_scale_col]

        print('\tScaling the features.')
        scaler = StandardScaler()
        eng_feature_all_node_df[to_scale_col] = \
            scaler.fit_transform(eng_feature_all_node_df[to_scale_col])

        node_elliptic_features, merged_df = \
            self.gen_merged_ell_eng_features(elliptic_features_df,
                                                           eng_feature_all_node_df)

        return eng_feature_all_node_df, node_elliptic_features, merged_df

    def save_graph_node_features(self, prod_data_path, graph_file_name, eng_feature_all_node_df,
                                 node_elliptic_features, merged_df):
        """
        saves three versions of features for the graph nodes
        1. OUR engineered features
        2. Elliptic features
        3. Merged
        """
        our_filename = prod_data_path + 'indiv_feat/eng_features_' + graph_file_name + '.csv'
        elliptic_filename = prod_data_path + 'indiv_feat/ell_features_' + graph_file_name + '.csv'
        merged_filename = prod_data_path + 'features_' + graph_file_name + '.csv'

        eng_feature_all_node_df.to_csv(our_filename, index=False)
        node_elliptic_features.to_csv(elliptic_filename, index=False)
        merged_df.to_csv(merged_filename, index=False)

    def generate_edge_list_for_RiWalk_unweighted(self):
        """
        generates graphs for the original RiWalk method
        :return unweighted, directed, simple graph
        """
        grouped_edges = self.edges.groupby(['source', 'target'])
        source_list = []
        target_list = []
        for key, value in grouped_edges:
            source_list.append(key[0])  # source node
            target_list.append(key[1])  # target node
        weight_list = [1] * len(source_list)
        simp_dir_edge_list = pd.DataFrame(list(zip(source_list, target_list, weight_list)),
                                          columns=['source', 'target', 'weight'])

        return simp_dir_edge_list


def main():
    # test the performance of the engineered features
    # edges list and nodes list file
    args = ArgParser.parse_args()

    main_path = args.main_path + 'inputs/'
    graph_name = args.graph
    node_filename = main_path + 'nodes_' + graph_name + '.csv'
    edge_filename = main_path + 'edges_' + graph_name + '.csv'

    print('Read edge list and node list.')
    edges_df = pd.read_csv(edge_filename)
    nodes_df = pd.read_csv(node_filename)

    # get anchor nodes list
    print('Retrieve anchor nodes list.')
    anchor_nodes = nodes_df.loc[nodes_df['is_anchor'] == 1]['node'].tolist()

    # --- generate features for all anchor nodes of a graph ---
    print('Generate features for the anchor nodes.')
    n_eng_features = BitcoinNodeEngFeatures(nodes_df, edges_df)
    node_feature_df = n_eng_features.gen_node_features_list(anchor_nodes)

    # save anchor nodes features to file
    print(node_feature_df.head())
    print('Save node features dataframe.')
    node_features_filename = main_path + 'features_' + graph_name + '.csv'
    node_feature_df.to_csv(node_features_filename, index=False)


if __name__ == '__main__':
    main()
