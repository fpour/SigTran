""""
Preparation of the graph that is going to be used by RiWalk --- Bitcoin
"""

import pandas as pd
import numpy as np
import networkx as nx
import time
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

import Bitcoin_FeatureEngineering as bfe
import ArgParser


def generate_features_for_all_nodes(bc_fe, elliptic_features_df, path, graph_name):
    """
    generates node-list with features for RiWalk-NA
    """
    # generate node-features-df
    print("\tFeatures generation starts.")
    start_time = time.time()
    eng_feature_all_node_df, node_elliptic_features, merged_df = \
        bc_fe.gen_all_features_all_nodes(elliptic_features_df)
    bc_fe.save_graph_node_features(path, graph_name, eng_feature_all_node_df,
                                   node_elliptic_features, merged_df)
    print("\tFeature generation lasted {} seconds.".format(time.time() - start_time))


def generate_edgelist_for_RiWalk(bc_fe, edgelist_RiWalk_filename):
    """
    generates edge list for RiWalk
    """
    # generate edge-list
    print("\tEdge-list generation starts.")
    start_time = time.time()
    simp_dir_edge_list = bc_fe.generate_edge_list_for_RiWalk_unweighted()
    simp_dir_edge_list.to_csv(edgelist_RiWalk_filename, index=False)
    print("\tEdge-list generation lasted {} seconds.".format(time.time() - start_time))


def bc_gen_t_SNE_components_2d(features_df):
    """
    generate the t-SNE components of a set of features
    * the 2 first most important components
    """
    values = features_df.drop(['node', 'address', 'isp', 'is_anchor',
                               'timestamp', 'txId', 'label'], axis=1).values.tolist()

    print('\tGraphPrep; t-SNE starts.')
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(values)
    print('\tt-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    # components
    min_first = [np.abs(np.min(tsne_results[:, 0])) * 10] * len(tsne_results)
    min_second = [np.abs(np.min(tsne_results[:, 1])) * 10] * len(tsne_results)
    tsne_data = {'tsne-2d-first': tsne_results[:, 0] + min_first,
                 'tsne-2d-second': tsne_results[:, 1] + min_second,
                 }
    return tsne_data


def generate_node_list_for_RiWalk(node_feature_df, selected_features, nodelist_RiWalk_filename):
    print("\tSelecting features for RiWalk node-list.")
    start_time = time.time()
    # nodes_selected_features_df = node_feature_df[selected_features].copy()
    #
    # # for invalid value in log2
    # to_shift_cols = [col for col in node_feature_df.columns if col not in ['node', 'address', 'isp',
    #                                                                       'is_anchor',
    #                                                                       'timestamp', 'txId', 'label']]
    # for col in to_shift_cols:
    #     nodes_selected_features_df[col] = nodes_selected_features_df[col].tolist() \
    #                                       + np.abs(np.min(nodes_selected_features_df[col]))

    # # t-SNE components
    tsne_data = bc_gen_t_SNE_components_2d(node_feature_df)
    nodes_selected_features_df = pd.DataFrame(list(zip(node_feature_df['node'].tolist(),
                                                       tsne_data['tsne-2d-first'],
                                                       tsne_data['tsne-2d-second'])),
                                              columns=['node', 'tsne_2d_1', 'tsne_2d_2'])

    nodes_selected_features_df.to_csv(nodelist_RiWalk_filename, index=False)
    print("\tFeature selection lasted {} seconds.".format(time.time() - start_time))


def main():
    """
        Experiments
    """
    # retrieve arguments
    args = ArgParser.parse_args()
    subgraph_path = args.input
    graph_name = args.graph
    prod_data_path = args.output
    gprep_opt = args.gprep_opt

    # input file; directly from XBlock
    node_filename = subgraph_path + 'nodes_' + graph_name + '.csv'
    edge_filename = subgraph_path + 'edges_' + graph_name + '.csv'

    # generated file
    nodelist_RiWalk_filename = prod_data_path + 'nodes_' + graph_name + '.nodelist'
    edgelist_RiWalk_filename = prod_data_path + 'edges_' + graph_name + '.edgelist'
    features_filename = prod_data_path + 'features_' + graph_name + '.csv'
    feat_imp_filename = prod_data_path + 'feature_importance_' + graph_name + '.csv'

    # elliptic files
    elliptic_classes_filename = subgraph_path + '_elliptic_txs_classes.csv'
    elliptic_feature_filename = subgraph_path + '_elliptic_txs_features.csv'

    nodes = pd.read_csv(node_filename)
    edges = pd.read_csv(edge_filename)

    # read elliptic dataset files
    elliptic_classes = pd.read_csv(elliptic_classes_filename)
    elliptic_features = pd.read_csv(elliptic_feature_filename)
    tx_features = ["tx_feat_" + str(i) for i in range(2, 95)]
    agg_features = ["agg_feat_" + str(i) for i in range(1, 73)]
    elliptic_features.columns = ["address", "timestamp"] + tx_features + agg_features
    elliptic_features = pd.merge(elliptic_features, elliptic_classes, left_on="address",
                                 right_on="txId", how='left')
    elliptic_features['label'] = elliptic_features['label'].apply(lambda x: '0' if x == "unknown" else x)

    # instantiate a feature engineering object
    bc_fe = bfe.BitcoinNodeEngFeatures(nodes, edges)

    # graph preparation tasks
    if gprep_opt == 'feature':
        generate_features_for_all_nodes(bc_fe, elliptic_features, prod_data_path, graph_name)
    elif gprep_opt == 'ri':
        # generate edge list for RiWalk
        generate_edgelist_for_RiWalk(bc_fe, edgelist_RiWalk_filename)
        # generate node list for RiWalk
        nodes_all_features_df = pd.read_csv(features_filename)
        feature_rank = pd.read_csv(feat_imp_filename)['feature'].tolist()
        # selected_features = ['node'] + feature_rank[0:10]
        selected_features = ['node'] + feature_rank  # all the features
        generate_node_list_for_RiWalk(nodes_all_features_df, selected_features, nodelist_RiWalk_filename)
    else:
        raise ValueError("Incorrect value for graph preparation option!")


if __name__ == '__main__':
    main()
