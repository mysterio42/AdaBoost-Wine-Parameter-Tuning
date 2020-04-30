import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import export_graphviz
import numpy as np
import seaborn as sns
from utils.data import fet_lab_names

FIGURES_DIR = 'figures/'
GRAPH_DIR = 'graph/'

plt.rcParams['figure.figsize'] = (13.66, 6.79)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100


def plot_pca(features, labels):
    pca = PCA(n_components=2)
    projections = pca.fit_transform(features)
    plt.scatter(projections[:, 0], projections[:, 1],
                c=labels, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('nipy_spectral', 10))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.savefig(FIGURES_DIR + 'Figure_wine' + '.png')
    plt.show()


def plot_graph(model, features, labels, graph_name):
    param = '-Tpng'
    dot_path = GRAPH_DIR + graph_name + '.dot'
    png_path = FIGURES_DIR + graph_name + '.png'

    feature_names, label_names = fet_lab_names(features, labels)

    export_graphviz(model, out_file=dot_path,
                    feature_names=feature_names, class_names=label_names,
                    rounded=True, filled=True,
                    precision=2, proportion=True)

    os.system(f'dot {param} {dot_path} -o {png_path}')



def plot_cm(cm, title):
    sns.heatmap(cm / np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    plt.title(title)
    plt.savefig(FIGURES_DIR + f'Figure_cm_{title}' + '.png')
    plt.show()
