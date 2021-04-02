""""
performance evaluation for node classification when using the elliptic features
"""
import pandas as pd
import numpy as np
import random
from random import sample
import time
import NodeClassification as nd

rnd_seed = 42
random.seed(rnd_seed)


def read_elliptic_data(features_filename, classes_filename):
    # read elliptic dataset files
    elliptic_classes = pd.read_csv(classes_filename)
    elliptic_features = pd.read_csv(features_filename)
    tx_features = ["tx_feat_" + str(i) for i in range(2, 95)]
    agg_features = ["agg_feat_" + str(i) for i in range(1, 73)]
    elliptic_features.columns = ["address", "timestamp"] + tx_features + agg_features
    elliptic_features = pd.merge(elliptic_features, elliptic_classes, left_on="address",
                                 right_on="txId", how='left')
    elliptic_features['label'] = elliptic_features['label'].apply(lambda x: '0' if x == "unknown" else x)

    return elliptic_features


def elliptic_features_node_classification(features_df, stats_file, flag, binary, seed, exp_id):
    """
    Bitcoin Feature Engineering
    """
    # make ready for classification
    y = [1 if lbl == '1' else 0 for lbl in features_df['label'].tolist()]  # only anchor nodes where selected
    X = features_df.drop(['address', 'timestamp', 'label', 'txId'], axis=1)
    X = X.values.tolist()

    # split the train and test set
    X_train, X_test, y_train, y_test = nd.train_test_split(X, y, seed)

    # classification
    nd.rf_lr_classification(X_train, X_test, y_train, y_test, stats_file, flag, binary,
                            exp_id, print_report=False)


def main():
    no_repeat = 50
    subgraph_path = 'path_to_sub_graphs'
    features_filename = subgraph_path + '_elliptic_txs_features.csv'
    classes_filename = subgraph_path + '_elliptic_txs_classes.csv'
    stats_filename = subgraph_path + '_elliptic_stats.csv'

    # read features
    features = read_elliptic_data(features_filename, classes_filename)

    # retrieve illicit & licit txs
    illicit_tx_list = features.loc[features['label'] == '1', 'txId'].tolist()
    licit_tx_list = features.loc[features['label'] == '2', 'txId'].tolist()

    no_anch_nodes = len(illicit_tx_list)
    # generate graphs repeatedly
    start_time = time.time()
    print("Starts iteration.")
    for idx in range(no_repeat):
        print("Iteration:", idx)
        # randomly select genuine nodes
        random.seed(rnd_seed + idx)
        anch_genuine_nodes = sample(licit_tx_list, no_anch_nodes)
        anch_phish_nodes = sample(illicit_tx_list, no_anch_nodes)

        # retrieve the features of the anchor nodes
        all_anch_nodes = anch_genuine_nodes + anch_phish_nodes
        anch_node_features = features.loc[features['address'].isin(all_anch_nodes)]
        print("\tNumber of all anchor nodes:", len(all_anch_nodes))

        # node classification
        exp_id = str(idx) + ';elliptic'
        elliptic_features_node_classification(anch_node_features, stats_filename, 'ellip', True,
                                              rnd_seed+idx, exp_id)

    print('Total execution time:', time.time() - start_time)


if __name__ == '__main__':
    main()
