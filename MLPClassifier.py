from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import ignore_warnings


# Multi Layer Perceptron classifier
class MLPClassifier:
    def __init__(self):
        self.model = MLP()
    # Train MLP
    @ignore_warnings()
    def train(self, x, y,solver="lbfgs", activation="tanh", alpha=0.05, learninRate="constant", maxIter=200):
        self.model = MLP(activation=activation, alpha=alpha, learning_rate=learninRate, solver=solver,
                         max_iter=maxIter).fit(x,y)

    # Test the model on test data
    @ignore_warnings()
    def score(self,x,y):
        score = self.model.score(x,y)
        return score

    # Test the model on train data

    # Predict the output of x
    def predict(self, x):
        return self.model.predict(x)

    # Cross validate the model
    def crossValidate(self, x, y, folds=5):
        scores = cross_val_score(self.model, x, y, cv=folds)
        return scores

    # Make the hyper parameters research
    # tuned_parameters list of parameters the parameters you want to try
    # scores list of string the score you want to test your parameters on
    # showBest bool if you want to show the results of the best parameters
    # showAll bool if you want to show the results of all the parameters
    @ignore_warnings()
    def tuning(self, tuned_parameters=[{"solver": ["lbfgs"],
                                        "activation": ["logistic", "tanh", "relu"],
                                        "alpha": [0.0001, 0.05],
                                        "learninRate": ["constant"], },
                                       {"solver": ["sgd"],
                                        "activation": ["logistic", "tanh", "relu"],
                                        "alpha": [0.0001, 0.05],
                                        "learning_rate": ["constant", "adaptive", "invscaling"],
                                        "learning_rate_init": [0.001, 0.2],
                                        "momentum": [0, 0.9],
                                        "nesterovs_momentum": [True, False]},
                                       {"solver": ["adam"],
                                        "activation": ["logistic", "tanh", "relu"],
                                        "alpha": [0.0001, 0.05],
                                        "learninRate": ["constant"],
                                        "learning_rate_init": [0.001, 0.01]}, ]
               , scores=["precision"], showBest=True, showAll=False):
        for score in scores:
            print("Tuning hyper-parameters for %s" % score)

            gridSearch = GridSearchCV(MLP(), tuned_parameters, scoring="%s_macro" % score)
            gridSearch.fit(self.data.xTrain, self.data.tTrain)

            if showBest:
                print("Best parameters set found on development set:")
                print(gridSearch.best_params_)

            if showAll:
                print("Grid scores on development set:")
                means = gridSearch.cv_results_["mean_test_score"]
                stds = gridSearch.cv_results_["std_test_score"]
                for mean, std, params in zip(means, stds, gridSearch.cv_results_["params"]):
                    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
