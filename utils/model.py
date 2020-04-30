import glob
import os
import random
import re
import string

import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utils.plot import plot_cm
from utils.plot import plot_graph

WEIGHTS_DIR = 'weights/'


def latest_modified_weight():
    """
    returns latest trained weight
    :return: model weight trained the last time
    """
    weight_files = glob.glob(WEIGHTS_DIR + '*')
    latest = max(weight_files, key=os.path.getctime)
    return latest


def load_model(path):
    """

    :param path: weight path
    :return: load model based on the path
    """
    with open(path, 'rb') as f:
        return joblib.load(filename=f)


def dump_model(model, name):
    model_name = WEIGHTS_DIR + name + generate_model_name(5) + '.pkl'
    with open(model_name, 'wb') as f:
        joblib.dump(value=model, filename=f, compress=3)
        print(f'Model saved at {model_name}')


def generate_model_name(size=5):
    """

    :param size: name length
    :return: random lowercase and digits of length size
    """
    letters = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters) for _ in range(size))


def train_model(features, labels, args):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3)

    if args.gs:

        params_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 1]
        }
        gs = GridSearchCV(estimator=AdaBoostClassifier(random_state=42), param_grid=params_grid, cv=10)
        gs.fit(features_train, labels_train)

        preds = gs.predict(features_test)

        score = accuracy_score(labels_test, preds)

        cm = confusion_matrix(labels_test, preds)

        best_params = re.sub("[{}' ,()]", '', str(gs.best_params_))
        plot_cm(cm, f'cm-accuracy:{score:.2f}AdaBoost-gs-{best_params}')

        plot_graph(gs.best_estimator_.estimators_[0], features, labels, 'adaboost-graph-gs-0')
        plot_graph(gs.best_estimator_.estimators_[1], features, labels, 'adaboost-graph-gs-1')
        plot_graph(gs.best_estimator_.estimators_[2], features, labels, 'adaboost-graph-gs-2')

        ans = input('Do you want to save the model weight? ')
        if ans in ('yes', '1'):
            dump_model(gs.best_estimator_.estimators_[0], 'AdaBoost-gs-est-0-')
            dump_model(gs.best_estimator_.estimators_[1], 'AdaBoost-gs-est-1-')
            dump_model(gs.best_estimator_.estimators_[2], 'AdaBoost-gs-est-2-')

        return gs.best_estimator_.estimators_[0]
    else:
        model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy', max_features='sqrt'),
                                   n_estimators=100, learning_rate=1, random_state=123)

        model.fit(features_train, labels_train)
        preds = model.predict(features_test)

        score = accuracy_score(labels_test, preds)
        cm = confusion_matrix(labels_test, preds)
        plot_cm(cm, f'cm-accuracy:{score:.2f}AdaBoost')

        plot_graph(model.estimators_[0], features, labels, 'adaboost-graph-0')

        ans = input('Do you want to save the model weight? ')
        if ans in ('yes', '1'):
            dump_model(model.estimators_[0], 'AdaBoost-0')

        return model.estimators_[0]
