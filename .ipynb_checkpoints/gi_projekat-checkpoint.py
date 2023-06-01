import scanpy as sc
import pandas as pd
import numpy as np
import squidpy as sp
import math
import warnings
import matplotlib.pyplot as plt
import random
import argparse

from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from mpl_toolkits.mplot3d import Axes3D


def read_file(filename):
    adata = sc.read_h5ad(filename)
    return adata


def preprocess_file(adata):
    # get rid of cells with fewer than 200 genes
    sc.pp.filter_cells(adata, min_genes=200)
    # get rid of genes that are found in fewer than 3 cells
    sc.pp.filter_genes(adata, min_cells=3)
    # get rid of cells whose annotation is unknown
    adata = adata[~adata.obs.annotation.isin(['Unknown'])]
    # data normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def select_best_features(adata, k_best_value):
    feature_selector = SelectKBest(k=k_best_value)
    y = adata.obs.annotation
    adata.feature_selected = np.zeros((len(adata.obs.annotation), 2))
    adata.feature_selected[:, 0] = adata.obsm['spatial'][:, 0]
    adata.feature_selected[:, 1] = adata.obsm['spatial'][:, 1]

    feature_selected = feature_selector.fit_transform(adata.X, y)
    adata.feature_selected = np.hstack((adata.feature_selected, feature_selected.toarray()))
    scaler = StandardScaler()
    adata.feature_selected = scaler.fit_transform(adata.feature_selected, y)

    return adata


def dim_reduction_truncated_pca(adata, n_components):
    feature_selector = TruncatedSVD(n_components=n_components, random_state=42)
    y = adata.obs.annotation
    adata.feature_selected = np.zeros((len(adata.obs.annotation), 2))
    adata.feature_selected[:, 0] = adata.obsm['spatial'][:, 0]
    adata.feature_selected[:, 1] = adata.obsm['spatial'][:, 1]

    feature_selected = feature_selector.fit_transform(adata.X, y)
    adata.feature_selected = np.hstack((adata.feature_selected, feature_selected))
    scaler = StandardScaler()
    adata.feature_selected = scaler.fit_transform(adata.feature_selected, y)

    return adata


def get_all_predictions(X, y, clf, k):
    strtfdKFold = StratifiedKFold(n_splits=k)
    kfold = strtfdKFold.split(X, y)
    predictions = None
    avg_score = 0

    for fold_index, (train, test) in enumerate(kfold):
        clf.fit(X[train], y[train])
        avg_score += clf.score(X[test], y[test])
        if predictions is None:
            predictions = clf.predict(X)
        else:
            predictions = np.vstack((predictions, clf.predict(X)))
    avg_score /= k
    return predictions, avg_score


def refine_annotations(predictions, y, l):
    new_annotations = y.copy()
    count_refined = 0
    for col_index in range(predictions.shape[1]):
        column = predictions[:, col_index].T
        count_different = len(column[column != y[col_index]])
        if count_different <= l:
            continue
        count_different_by_class = dict()
        for label in column[column != y[col_index]]:
            if label in count_different_by_class:
                count_different_by_class[label] += 1
            else:
                count_different_by_class[label] = 1
        label_with_max = max(count_different_by_class, key=lambda key: count_different_by_class[key])
        max_value = count_different_by_class[label_with_max]
        if max_value > l:
            # print(col_index, y[col_index], label_with_max)
            new_annotations[col_index] = label_with_max
            count_refined += 1
    return new_annotations, count_refined


def plot_values(results_table, output_name):
    fig = plt.figure(figsize=(100, 100), layout='constrained')

    # features
    xdata = results_table[:, 0]
    # folds
    ydata = results_table[:, 1]
    # l
    zdata = results_table[:, 2]

    ann_data_changed = results_table[:, 3].flatten()

    ax3d = plt.figure().add_subplot(111, projection='3d')

    ax3d.scatter(xdata, ydata, zdata)

    for x, y, z, label in zip(xdata, ydata, zdata, ann_data_changed):
        ax3d.text(x, y, z, str(label), fontsize=6)

    ax3d.set_xlabel('Features')
    ax3d.set_ylabel('Folds')
    ax3d.set_zlabel('L')
    ax3d.grid(False)

    plt.title("Number of changed annotations per number of features, k and l")
    plt.savefig(output_name + '_num_changed_annotations.png')
    return plt


def fun_image_representation(filename, results_table, new_annotations_table, output_name):
    for i in range(0, 3):
        row = random.randint(0, len(results_table))
        table_row = results_table[row]
        new_ann_row = new_annotations_table[row]
        adata.obs['new_annotation'] = new_ann_row
        adata.uns['new_annotation_colors'] = adata.uns['annotation_colors']
        text = "For {} with features={}, folds = {} and l={},\n the number of features changed is {}"\
            .format(filename, table_row[0], table_row[1], table_row[2], table_row[3])
        output_name = output_name + '_fun_' + str(i) + '.png'
        print(output_name)
        sp.pl.spatial_scatter(adata, shape=None, color=["new_annotation", "annotation"], title=text, save = output_name)


def train_models(adata, feature_selection, clf, best_features, k_number_of_folds, l_value_percentage):
    results_table = None
    new_annotations_table = None
    avg_scores = np.zeros((len(best_features), len(k_number_of_folds)))
    for feature_index, number_of_features in enumerate(best_features):
        feature_selection(adata, number_of_features)
        print("**started {} feature **".format(number_of_features))
        for fold_index, number_of_folds in enumerate(k_number_of_folds):

            predictions, avg_score = get_all_predictions(adata.feature_selected, adata.obs.annotation, clf,
                                                         number_of_folds)
            avg_scores[feature_index, fold_index] = avg_score
            # print(number_of_features, number_of_folds, avg_score)
            for l in range(math.ceil(l_value_percentage * number_of_folds), number_of_folds):
                new_annotations, count_refined = refine_annotations(predictions, adata.obs.annotation, l)
                result = np.array([number_of_features, number_of_folds, l, count_refined])
                if results_table is None:
                    results_table = result
                else:
                    results_table = np.vstack((results_table, result))
                if new_annotations_table is None:
                    new_annotations_table = new_annotations
                else:
                    new_annotations_table = np.vstack((new_annotations_table, new_annotations))
    print(avg_scores)
    print("**done**")
    return results_table, new_annotations_table


if __name__ == '__main__':
    # Create the parser
    my_parser = argparse.ArgumentParser(description='List of argument')

    # Add the arguments
    my_parser.add_argument('--dataset', type=str, default='embryo',
                           help='embryo or brain', required=False)

    my_parser.add_argument('--l', type=float, default='0.35',
                           help='l value, between 0 and 1, represents required number / percentage of different predictions',
                           required=False)

    my_parser.add_argument('--classifier', type=str, default='sgd',
                           help='sgd or mlp', required=False)

    my_parser.add_argument('--feature_selection', type=str, default='truncated_pca',
                           help='k_best or truncated_pca', required=False)

    # Execute the parse_args() method
    args = my_parser.parse_args()

    if args.dataset == 'brain':
        dataset = '../Mouse_brain_cell_bin.h5ad'
    else:
        dataset = '../E9.5_E1S1.MOSTA.h5ad'

    adata = read_file(dataset)
    adata = preprocess_file(adata)

    k_best_features = [10, 20, 50, 100]
    k_number_of_folds = [3, 5, 7, 10]

    l_value_percentage = args.l

    if args.classifier == 'mlp':
        clf = MLPClassifier()
    else:
        clf = SGDClassifier()

    if args.feature_selection == 'k_best':
        func = select_best_features
    else:
        func = dim_reduction_truncated_pca

    results, annotations = train_models(adata, func, clf, k_best_features, k_number_of_folds, l_value_percentage)

    output_name = args.dataset + '_' + args.classifier + '_' + args.feature_selection
    plt = plot_values(results, output_name)
    fun_image_representation(dataset, results, annotations, output_name)
