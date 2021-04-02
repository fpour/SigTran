""""
Preparation of the graph that is going to be used by RiWalk --- Ethereum
"""

# --- import statements ---

import pandas as pd
import numpy as np
import networkx as nx
import time
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

import FeatureSelection
import ArgParser
import FeatureEngineering

# --- variables ---
rnd_seed = 42


class NodeFeature:
    """
    a class to define the appropriate features for ALL nodes of the graph
    this facilitates the modifications to the RiWalk
    """

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.n_fe_features = FeatureEngineering.NodeEngFeatures(self.nodes, self.edges)

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
        simp_dir_edge_list = pd.DataFrame(list(zip(source_list, target_list)), columns=['source', 'target'])

        return simp_dir_edge_list

    def generate_edge_list_for_RiWalk_weighted(self):
        """
        generate graph for RiWalk
        convert MDG to a simple directed weighted graph
        """
        mdg = nx.from_pandas_edgelist(self.edges, source='source', target='target',
                                      edge_attr=['amount', 'timestamp'], create_using=nx.MultiDiGraph())

        # generate simple graph
        simG = nx.DiGraph()
        for u, v, data in mdg.edges(data=True):
            a = data['amount'] if 'amount' in data else 0
            t = data['timestamp'] if 'timestamp' in data else 0

            if simG.has_edge(u, v):
                simG[u][v]['amount'] += a  # sum of amounts

                current_timestamp = simG[u][v]['timestamp']
                simG[u][v]['timestamp'] = max(t, current_timestamp)  # more recent

                simG[u][v]['n_tx'] += 1
            else:
                simG.add_edge(u, v, amount=a, timestamp=t, n_tx=1)

        # normalize weight values
        for u, v, data in simG.edges(data=True):
            if simG.degree(u, weight='amount') != 0:
                simG[u][v]['amount'] = simG[u][v]['amount'] / simG.degree(u, weight='amount')
            else:
                simG[u][v]['amount'] = 0
            simG[u][v]['timestamp'] = simG[u][v]['timestamp'] / simG.degree(u, weight='timestamp')
            simG[u][v]['n_tx'] = simG[u][v]['n_tx'] / simG.degree(u)

            # aggregated weight value
            simG[u][v]['weight'] = simG[u][v]['amount'] * simG[u][v]['timestamp'] * simG[u][v]['n_tx']

        edge_list_df = nx.to_pandas_edgelist(simG, source='source', target='target')
        # only preserve the 'weight'
        edge_list_df = edge_list_df.drop(['amount', 'timestamp', 'n_tx'], axis=1)

        return edge_list_df

    def gen_features_all_nodes(self):
        """
        get a dataframe containing the selected features for all nodes
        """
        node_list = self.nodes['node'].tolist()  # select all the nodes
        all_feature_df = self.n_fe_features.gen_node_features_list(node_list)
        return all_feature_df

    def gen_t_SNE_components_2d(self, features_df, shift_coff):
        """
        generate the t-SNE components of a set of features
        * the 2 first most important components
        """
        values = features_df.drop(['node', 'address', 'isp', 'is_anchor'], axis=1).values.tolist()

        min_max_scaler = MinMaxScaler()
        values_scaled = min_max_scaler.fit_transform(values)

        print('\tGraphPrep; t-SNE starts.')
        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(values_scaled)
        print('\tt-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

        # plotting
        min_first = [np.abs(np.min(tsne_results[:, 0])) * shift_coff] * len(tsne_results)
        min_second = [np.abs(np.min(tsne_results[:, 1])) * shift_coff] * len(tsne_results)
        tsne_data = {'tsne-2d-first': tsne_results[:, 0] + min_first,
                     'tsne-2d-second': tsne_results[:, 1] + min_second,
                     }
        return tsne_data


def generate_edgelist_for_RiWalk(n_feature, edgelist_RiWalk_filename):
    """
    generates edge list for RiWalk
    """
    # generate edge-list
    print("\tEdge-list generation starts.")
    start_time = time.time()
    # simp_dir_edge_list = n_feature.generate_edge_list_for_RiWalk_unweighted()
    simp_dir_edge_list = n_feature.generate_edge_list_for_RiWalk_weighted()
    # simp_dir_edge_list.to_csv(edgelist_RiWalk_filename, sep=' ', index=False, header=False)
    simp_dir_edge_list.to_csv(edgelist_RiWalk_filename, index=False)
    print("\tEdge-list generation lasted {} seconds.".format(time.time() - start_time))


def generate_features_for_all_nodes(n_feature, features_filename):
    """
    generates node-list with features for RiWalk-NA
    """
    # generate node-features-df
    print("\tFeatures generation starts.")
    start_time = time.time()
    nodes_all_features_df = n_feature.gen_features_all_nodes()
    nodes_all_features_df.to_csv(features_filename, index=False)
    print("\tFeature generation lasted {} seconds.".format(time.time() - start_time))
    return nodes_all_features_df


def generate_node_list_for_RiWalk(nodes_all_features_df, selected_features,
                                  n_feature, tsne, nodelist_RiWalk_filename):
    print("\tSelecting features for RiWalk node-list.")
    start_time = time.time()

    if not tsne:
        # select some features
        nodes_selected_features_df = nodes_all_features_df[selected_features].copy()
    else:
        # t-SNE components
        shift_coeff = 10
        tsne_data = n_feature.gen_t_SNE_components_2d(nodes_all_features_df, shift_coeff)
        nodes_selected_features_df = pd.DataFrame(list(zip(nodes_all_features_df['node'].tolist(),
                                                           tsne_data['tsne-2d-first'],
                                                           tsne_data['tsne-2d-second'])),
                                                  columns=['node', 'tsne_2d_1', 'tsne_2d_2'])

    nodes_selected_features_df.to_csv(nodelist_RiWalk_filename, index=False)
    print("\tFeature selection lasted {} seconds.".format(time.time() - start_time))


def main():
    """
    Experiments
    """
    args = ArgParser.parse_args()
    input_path = args.input
    graph_name = args.graph
    output_path = args.output
    gprep_opt = args.gprep_opt

    # input file; directly from XBlock
    node_filename = input_path + 'nodes_' + graph_name + '.csv'
    edge_filename = input_path + 'edges_' + graph_name + '.csv'

    # generated file
    nodelist_RiWalk_filename = output_path + 'nodes_' + graph_name + '.nodelist'
    edgelist_RiWalk_filename = output_path + 'edges_' + graph_name + '.edgelist'
    features_filename = output_path + 'features_' + graph_name + '.csv'
    feat_imp_filename = output_path + 'feature_importance_' + graph_name + '.csv'

    nodes = pd.read_csv(node_filename)
    edges = pd.read_csv(edge_filename)

    n_feature = NodeFeature(nodes, edges)

    # graph pre-paration tasks
    if gprep_opt == 'feature':
        generate_features_for_all_nodes(n_feature, features_filename)
    elif gprep_opt == 'ri':
        # generate edge list for RiWalk
        generate_edgelist_for_RiWalk(n_feature, edgelist_RiWalk_filename)
        # generate node list for RiWalk
        nodes_all_features_df = pd.read_csv(features_filename)
        feature_rank = pd.read_csv(feat_imp_filename)['feature'].tolist()
        selected_features = ['node'] + feature_rank[0:10]

        tsne = False  # use TSNE components
        generate_node_list_for_RiWalk(nodes_all_features_df, selected_features, n_feature,
                                      tsne, nodelist_RiWalk_filename)
    else:
        raise ValueError("Incorrect value for graph preparation option!")


if __name__ == '__main__':
    main()
