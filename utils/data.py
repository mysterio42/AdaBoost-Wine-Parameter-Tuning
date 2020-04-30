import pandas as pd


def load_data(path, label, *features):
    def isTasty(quality):
        if quality >= 7:
            return 1
        else:
            return 0

    df = pd.read_csv(path, sep=';')
    return df[list(*features)], df[label].apply(isTasty)


def fet_lab_names(features, labels):
    assert isinstance(features, pd.DataFrame)
    assert isinstance(labels, pd.Series)
    return list(features.columns), list(map(str, list(labels.unique())))
