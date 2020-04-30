import argparse

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.data import load_data
from utils.model import train_model, load_model
from utils.plot import plot_pca, plot_graph


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()

    parser.add_argument('--load', type=str2bool, default=False,
                        help='True: Load trained model  False: Train model default: True')
    parser.add_argument('--gs', type=str2bool, default=False,
                        help='Find optimal parameters with 10-Fold GridSearchCV')

    parser.print_help()

    return parser.parse_args()


if __name__ == '__main__':

    np.random.seed(1)

    args = parse_args()

    features_name = (
        'fixed acidity',
        'volatile acidity',
        'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol'
    )
    label_name = 'quality'
    features, labels = load_data('data/wine.csv', label_name, features_name)

    features[list(features_name)] = MinMaxScaler().fit_transform(features[list(features_name)])

    if args.load:
        model = load_model('weights/AdaBoost-gs-est-2-pnd2g.pkl')
        plot_graph(model, features, labels, 'wine-graph-33')

        fixed_acidity, volatile_acidity, citric_acid, \
        residual_sugar, chlorides, free_sulfur_dioxide, \
        total_sulfur_dioxide, density, pH, \
        sulphates, alcohol = 7, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3, 0.45, 8.8

        print(model.predict([list((fixed_acidity, volatile_acidity, citric_acid,
                                   residual_sugar, chlorides, free_sulfur_dioxide,
                                   total_sulfur_dioxide, density, pH,
                                   sulphates, alcohol))])[0])
    else:
        plot_pca(features, labels)
        model = train_model(features, labels,args)


        fixed_acidity, volatile_acidity, citric_acid, \
        residual_sugar, chlorides, free_sulfur_dioxide, \
        total_sulfur_dioxide, density, pH, \
        sulphates, alcohol = 7, 0.27, 0.36, 20.7, 0.045, 45, 170, 1.001, 3, 0.45, 8.8

        print(model.predict([list((fixed_acidity, volatile_acidity, citric_acid,
                                   residual_sugar, chlorides, free_sulfur_dioxide,
                                   total_sulfur_dioxide, density, pH,
                                   sulphates, alcohol))])[0])
