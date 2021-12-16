import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.utils._testing import ignore_warnings
from sklearn.ensemble import RandomForestClassifier


class RFClassifier:
    # {'n_estimators': 50, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True}
    def __init__(self, n=50, _bootstrap=False, _max_depth=10, _min_samples_leaf=2):
        self.rf = RandomForestClassifier(n_estimators=n, bootstrap=_bootstrap, max_depth=_max_depth,
                                         min_samples_leaf=_min_samples_leaf
                                         , max_features='sqrt')
        self.n = n

    # Train the model
    def train(self, x, y):
        self.rf.fit(x, y)

    # Cross validate the model
    def crossValidate(self, x, y, folds=5):
        scores = cross_val_score(self.rf, x, y, cv=folds)
        return scores

    # Return the score
    @ignore_warnings()
    def score(self, x, y):
        return self.rf.score(x, y)

    def testScore(self):
        return self.rf.score(self.data.xTest, self.data.tTest)

    def trainScore(self):
        return self.rf.score(self.data.xTrain, self.data.tTrain)

    # Random search for best parameters
    @ignore_warnings()
    def random_tuning(self, x, y):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]

        # Other parameters
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [2]
        bootstrap = [True]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf = RandomForestClassifier()

        # Random search of parameters, using 3 fold cross validation,
        # search across 300 different combinations, and use all available cores
        random_grid_search = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=300, cv=3,
                                                verbose=2,
                                                random_state=42, n_jobs=-1)
        random_grid_search.fit(x, y)

        self.rf = random_grid_search.best_estimator_

        return random_grid_search.best_params_

    # Search best parameters between number of trees and min samples split4
    @ignore_warnings()
    def tuning(self):
        param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80],
                      'min_samples_split': [2, 3, 4, 5],
                      'bootstrap': [True],
                      'max_depth': [90],
                      'min_samples_leaf': [1]
                      }
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=3, n_jobs=-1, verbose=2)

        grid_search.fit(self.data.xTrain, self.data.tTrain)

        self.rf = grid_search.best_estimator_

        return grid_search.best_params_

    # Predict the output of x
    @ignore_warnings()
    def predict(self, x):
        return self.rf.predict(x)
