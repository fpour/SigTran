"""
Implement the node classification task
In fact, this file is a collection of utility functions
"""

# --- import statements ---
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.manifold import TSNE
import time
import seaborn as sns

import ArgParser

# --- parameters ---
rnd_seed = 42
random.seed(rnd_seed)
test_size = 0.2


# --- utility functions ---

def perf_report(identifier, y_true, y_pred, binary, print_enable=False):
    """
    calculates and prints the performance results
    """
    if binary:
        # print(">>> Binary Classification.")
        prec, rec, f1, num = precision_recall_fscore_support(y_true, y_pred, average='binary')
        micro_f1 = f1_score(y_true, y_pred, average='binary')
    else:
        print(">>> Multi-class Classification.")
        prec, rec, f1, num = precision_recall_fscore_support(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    if print_enable:
        print("\t*** {} performance reports: ***".format(str(identifier)))
        print("\t\tPrecision: %.3f \n\t\tRecall: %.3f \n\t\tF1-Score: %.3f" % (prec, rec, f1))
        print('\t\tMicro-Average F1-Score: %.3f' % micro_f1)
        print('\t\tAccuracy: %.3f' % acc)

    return prec, rec, f1, acc


def train_test_split(X, y, rnd_seed):
    """
    split the features and the labels according to the indices
    :param X: feature set, should be array or list
    :param y: labels, should be array or list
    :param rnd_seed: random seed
    """
    # generate indices for the train and test set
    indices = [i for i in range(len(y))]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=rnd_seed)
    sss.get_n_splits(indices, y)
    train_indices, test_indices = next(sss.split(indices, y))

    # train/test split
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]

    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


def simple_classification(clf, clf_id, emb_flag, X_train, X_test, y_train, y_test,
                          binary, exp_id, print_enable=False):
    """
    train the model on the train set and test it on the test set.
    to be consistent among different run, the indices are passed.
    important NOTE: it is implicitly inferred that the positive label is 1.
    no cross-validation is applied.
    """

    # train the model
    clf.fit(X_train, y_train)

    # predict the training set labels
    y_train_pred = clf.predict(X_train)

    # predict the test set labels
    y_test_pred = clf.predict(X_test)

    # evaluate the performance for the training set
    tr_prec, tr_rec, tr_f1, tr_acc = perf_report(str(clf_id) + ' - Training Set',
                                                 y_train, y_train_pred, binary, print_enable)
    ts_prec, ts_rec, ts_f1, ts_acc = perf_report(str(clf_id) + ' - Test Set',
                                                 y_test, y_test_pred, binary, print_enable)

    # auc-roc
    if binary:
        tr_roc_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        ts_roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    else:
        tr_roc_auc = roc_auc_score(y_train, clf.predict_proba(X_train), multi_class='ovr')
        ts_roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')

    split_exp_id = exp_id.split(";")
    if len(split_exp_id) == 2:
        index = split_exp_id[0]
        id = split_exp_id[1]
    elif len(split_exp_id) == 1:
        index = 0
        id = split_exp_id[0]
    else:
        raise ValueError("Incorrect Experiment ID!")

    perf_dict = {
        'index': index,
        'exp_id': id,
        'emb_method': str(emb_flag),
        'classifier': str(clf_id),

        'train_prec': tr_prec,
        'train_rec': tr_rec,
        'train_f1': tr_f1,
        'train_acc': tr_acc,
        'train_auc': tr_roc_auc,

        'test_prec': ts_prec,
        'test_rec': ts_rec,
        'test_f1': ts_f1,
        'test_acc': ts_acc,
        'test_auc': ts_roc_auc
    }

    return perf_dict, clf


def rf_lr_classification(X_train, X_test, y_train, y_test, stats_file, flag,
                         binary, exp_id, print_report=False):
    """
    apply classification to input X with label y with "Random Forest" & "Logistic Regression"
    :param X_train: train set
    :param X_test: test set
    :param y_train: train set labels
    :param y_test: test set labels
    :param print_report: whether print the results of classification or not
    :return the classification results
    """
    # define classifier
    rf_clf = RandomForestClassifier(n_estimators=50, max_features=10, max_depth=5, random_state=rnd_seed)
    lr_clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1e5, random_state=rnd_seed)

    # apply classification
    rf_perf, rf_clf = simple_classification(rf_clf, 'RF', flag, X_train, X_test, y_train, y_test,
                                            binary, exp_id, print_report)
    lr_perf, lr_clf = simple_classification(lr_clf, 'LR', flag, X_train, X_test, y_train, y_test,
                                            binary, exp_id, print_report)

    # append the results to file
    stats_df = pd.read_csv(stats_file)
    stats_df = stats_df.append(rf_perf, ignore_index=True)
    stats_df = stats_df.append(lr_perf, ignore_index=True)
    stats_df.to_csv(stats_file, index=False)

    return rf_perf, rf_clf, lr_perf, lr_clf


def RF_sorted_feature_importance(clf, feature_name):
    """
    return the top 10 most important features of the RF clf model
    assumption: clf is a trained RF model
    """
    # feature importance
    importance = clf.feature_importances_
    indices = np.argsort(importance)[::-1]

    # Print the feature ranking
    sorted_feature_name = [feature_name[indices[i]] for i in range(len(feature_name))]
    sorted_feature_importance = [importance[indices[i]] for i in range(len(feature_name))]
    feature_imp_df = pd.DataFrame(list(zip(sorted_feature_name, sorted_feature_importance)),
                                  columns=['feature', 'importance'])
    return feature_imp_df


def RF_feature_imp(X, y, feature_name, png_file):
    """
    calculate feature importance for the Random Forest Classifier
    :param X: features
    :param y: labels
    :param feature_name: the name of the features
    """
    # define and fit classifier
    rf_clf = RandomForestClassifier(n_estimators=100, max_features=16, max_depth=5,
                                    random_state=rnd_seed)
    rf_clf.fit(X, y)

    # feature importance
    importances = rf_clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(len(feature_name)):
        print("%d. feature %d (%s) (%f)" % (f + 1, indices[f], feature_name[indices[f]],
                                            importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    plt.figure()
    plt.title("Feature Importance")
    plt.bar(range(len(feature_name)), importances[indices], color="g", yerr=std[indices], align="center")
    plt.xticks(range(len(feature_name)), indices)
    plt.xlim([-1, len(feature_name)])
    # plt.show()
    plt.savefig(png_file)


def read_emb_and_node_list(emb_file, node_file):
    # read embedding
    emb_df = pd.read_csv(emb_file, sep=' ', skiprows=1, header=None)
    emb_df.columns = ['node'] + [f'emb_{i}' for i in range(emb_df.shape[1] - 1)]

    # read node list
    node_df = pd.read_csv(node_file)
    node_df = node_df[['node', 'isp', 'is_anchor']]

    # merge
    merged_df = emb_df.merge(node_df, on='node', how='left')

    return merged_df


def data_preproc_for_RiWalk_Binary_clf(emb_file, node_file):
    """
    pre-process the RiWalk generated embedding for node classification
    """
    # read and merge the data frames
    merged_df = read_emb_and_node_list(emb_file, node_file)

    # datasets for  BINARY classification
    X = merged_df[merged_df['is_anchor'] == 1]  # only anchor nodes
    y = X['isp'].tolist()
    X = X.drop(['node', 'isp', 'is_anchor'], axis=1)
    feature_names = X.columns
    X = X.values.tolist()

    # split the train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, rnd_seed)

    return X_train, X_test, y_train, y_test, feature_names


def prepare_data_for_concat_fe_emb(emb_file, fe_file):
    """
    pre-process the data for the node classification of a new dataset consisting of the
    engineered features and the embeddings
    """
    # read embedding
    emb_df = pd.read_csv(emb_file, sep=' ', skiprows=1, header=None)
    emb_df.columns = ['node'] + [f'emb_{i}' for i in range(emb_df.shape[1] - 1)]

    # read node list
    node_df = pd.read_csv(fe_file)
    # scale features
    feature_col = [f for f in node_df.columns if f not in ['node', 'address', 'isp', 'is_anchor']]
    scaler = StandardScaler()
    node_df[feature_col] = scaler.fit_transform(node_df[feature_col])

    # merge
    merged_df = emb_df.merge(node_df, on='node', how='left')

    # datasets for  BINARY classification
    X = merged_df[merged_df['is_anchor'] == 1]  # only anchor nodes
    y = X['isp'].tolist()
    X = X.drop(['node', 'address', 'isp', 'is_anchor'], axis=1)
    X = X.values.tolist()

    # split the train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, rnd_seed)

    return X_train, X_test, y_train, y_test


def plot_TSNE(values, labels, png_file):
    """
    plot the embeddings as a TSNE graph
    """
    print('\tt-SNE starts.')
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(values)
    print('\tt-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

    # plotting
    p_data = {'tsne-2d-first': tsne_results[:, 0],
              'tsne-2d-second': tsne_results[:, 1],
              'label': labels,
              }

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-first", y="tsne-2d-second",
        hue="label",
        palette=sns.color_palette("hls", len(set(labels))),
        data=p_data,
        legend="full",
        alpha=0.3
    )
    # plt.show()
    plt.savefig(png_file)


def EF_analysis_selected_nodes(output_path, graph, edges_filename, nodes_filename,
                               features_filename, stats_file, feat_imp_filename,
                               flag, binary, rnd_seed, exp_id, extra_analysis):
    # print("\tRead edge list and node list.")
    # start_time = time.time()
    # edges_df = pd.read_csv(edges_filename)
    nodes_df = pd.read_csv(nodes_filename)
    # print("\t\tTime elapsed {} seconds.".format(time.time() - start_time))

    print("\tRetrieve anchor nodes for classification.")
    start_time = time.time()
    selected_node_list = nodes_df.loc[nodes_df['is_anchor'] == 1]['node'].tolist()
    print("\t\tTime elapsed {} seconds.".format(time.time() - start_time))


    print("\tRead features for anchor nodes.")
    start_time = time.time()
    all_node_features_df = pd.read_csv(features_filename)
    features_df = all_node_features_df.loc[all_node_features_df['node'].isin(selected_node_list)]
    print("\t\tTime elapsed {} seconds.".format(time.time() - start_time))

    # make ready for classification
    # features_df = pd.read_csv(features_filename)
    y = features_df['isp'].tolist()  # only anchor nodes where selected
    X_orig = features_df.drop(['node', 'address', 'isp', 'is_anchor'], axis=1)
    feature_names = X_orig.columns
    X_orig = X_orig.values.tolist()

    # split the train and test set
    print("\tTrain-Test split.")
    X_train, X_test, y_train, y_test = train_test_split(X_orig, y, rnd_seed)

    # scale the features; note that it should be fitted on the train set ONLY
    print('\tScaling the features.')
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)
    X_train_scaled = min_max_scaler.transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)

    # classification
    print('\tApplying classification.')
    start_time = time.time()
    rf_perf, rf_clf, lr_perf, lr_clf = rf_lr_classification(X_train_scaled, X_test_scaled, y_train,
                                                            y_test, stats_file, flag, binary,
                                                            exp_id, print_report=True)
    print("\t\tTime elapsed {} seconds.".format(time.time() - start_time))

    # calculates and saves features importance
    feature_imp_df = RF_sorted_feature_importance(rf_clf, feature_names)
    feature_imp_df.to_csv(feat_imp_filename, index=False)

    if extra_analysis:
        # Feature importance
        print("\tInvestigate feature importance.")
        png_file = output_path + '/' + graph + '_' + flag + '_FE_feature_impo.png'
        RF_feature_imp(X_train_scaled, y_train, feature_names, png_file)

        # plot t-SNE graph
        print("\tt-SNE graph.")
        values = X_orig
        groups = y
        png_file = output_path + '/' + graph + '_' + flag + '_FE_tsne.png'
        plot_TSNE(values, groups, png_file)

    print("FE node classification finished.")


def RiWalk_analysis_selected_nodes(output_path, graph, emb_filename, nodes_filename, stats_filename,
                                   flag, binary, exp_id, extra_analysis):
    # prepare the data
    print("\tPrepare data sets.")
    X_train, X_test, y_train, y_test, feature_names = data_preproc_for_RiWalk_Binary_clf(emb_filename,
                                                                                         nodes_filename)
    # classification
    print('\tApplying classification.')
    start_time = time.time()
    rf_lr_classification(X_train, X_test, y_train, y_test, stats_filename, flag,
                         binary, exp_id, print_report=True)
    print("\tTime elapsed {} seconds.".format(time.time() - start_time))

    if extra_analysis:
        # Feature importance
        print("\tInvestigate feature importance.")
        png_file = output_path + '/' + graph + '_' + flag + '_Ri_feature_impo.png'
        RF_feature_imp(X_train, y_train, feature_names, png_file)

        # plot t-SNE graph
        print("\tPlot t-SNE.")
        values = X_train + X_test
        groups = y_train + y_test
        # nodes_df = pd.read_csv(nodes_filename)
        png_file = output_path + '/' + graph + flag + '_Ri_tsne.png'
        plot_TSNE(values, groups, png_file)

    print("RiWalk node classification finished.")


def nd_clf_fe_emb_combined(emb_file, fe_file, stats_file, flag, binary, exp_id):
    """
    apply the node classification based on a new feature set constructed by combining the
    engineered features and the (structural) embedding generated by an automatic method like node2vec
    """
    print("\tConcatenating embedding with engineered features for node classification.")
    # data preparation
    X_train, X_test, y_train, y_test = prepare_data_for_concat_fe_emb(emb_file, fe_file)

    # classification
    print('\tApplying classification.')
    start_time = time.time()
    rf_lr_classification(X_train, X_test, y_train, y_test, stats_file, flag,
                         binary, exp_id, print_report=True)
    print("\tTime elapsed {} seconds.".format(time.time() - start_time))


def main():
    """
    end-to-end classification
    """
    binary = True  # binary or multi-class classification.

    args = ArgParser.parse_args()
    input_path = args.input
    graph_filename = args.graph
    prod_data_dir = args.output

    flag = args.flag
    clf_opt = args.clf_opt
    exp_id = args.exp_id

    # data_path = main_path + 'inputs/'
    # prod_data_dir = main_path + 'outputs/'
    stats_file = prod_data_dir + 'stats_' + flag + '.csv'

    edges_filename = input_path + 'edges_' + graph_filename + '.csv'
    nodes_filename = input_path + 'nodes_' + graph_filename + '.csv'
    features_filename = prod_data_dir + 'features_' + graph_filename + '.csv'

    feat_imp_filename = prod_data_dir + 'feature_importance_' + graph_filename + '.csv'

    if clf_opt == 'fe':
        # ------------------ Feature Engineering ------------------
        # read the input file and generating the features and the labels set
        print("Node Classification --- Feature Engineering ---")

        EF_analysis_selected_nodes(prod_data_dir, graph_filename, edges_filename, nodes_filename,
                                   features_filename, stats_file, feat_imp_filename, 'FE', binary,
                                   rnd_seed, exp_id, extra_analysis=False)
        print("--- Node Classification Feature Engineering is done ---")
        # ---------------------------------------------------------

    elif clf_opt == 'concat':
        print("Node classification: Concat. FE &" + flag + " embeddings.")

        emb_file = prod_data_dir + 'emb_' + str(flag) + '_' + graph_filename + '.emb'
        fe_file = features_filename
        nd_clf_fe_emb_combined(emb_file, fe_file, stats_file, flag, binary, exp_id)

    else:
        # ------------------ RiWalk -------------------------------
        print("Node classification: --- RiWalk - " + flag + "---")

        # set file names
        emb_filename = prod_data_dir + 'emb_' + str(flag) + '_' + graph_filename + '.emb'

        RiWalk_analysis_selected_nodes(prod_data_dir, graph_filename, emb_filename, nodes_filename, stats_file,
                                       flag, binary, exp_id, extra_analysis=False)
        print("--- Classification RiWalk is done ---")
        # ---------------------------------------------------------


if __name__ == '__main__':
    main()
