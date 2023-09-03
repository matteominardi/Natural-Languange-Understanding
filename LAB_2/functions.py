from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np

def experiment(df, model, vec):
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)

    for train_index, test_index in stratified_split.split(df.data, df.target):
        df.data = np.array(df.data)
        df.target = np.array(df.target)

        model.fit(vec.fit_transform(df.data[train_index]), df.target[train_index])
        y_pred = model.predict(vec.transform(df.data[test_index]))
        y_true = df.target[test_index]

        print(classification_report(y_true, y_pred, target_names=df.target_names, digits=2))

    