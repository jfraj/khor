import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint as sp_randint
from time import time
from operator import itemgetter

from sklearn import grid_search
from sklearn.learning_curve import learning_curve

from gb import gbClf
import hyperparams
import fit_features

class clf_learning(gbClf):

    """Class that will contain learning functions."""

    def grid_report(self, grid_scores, n_top=5):
        """Utility function to report best scores."""
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            print("Model with rank: {0}".format(i + 1))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                score.mean_validation_score,
                np.std(score.cv_validation_scores)))
            print("Parameters: {0}".format(score.parameters))
            print("")

    def grid_search(self, **kwargs):
        """Using grid search to find the best parameters."""
        n_jobs = kwargs.get('n_jobs', 1)
        n_iter = kwargs.get('n_iter', 5)
        col2fit = kwargs.get('features')
        bids_path = kwargs.get('bids_path', 'data/bids.csv')
        score = kwargs.get('score')

        # use a full grid over all parameters
        parameters = {"max_depth": sp_randint(1, 30),
                      "max_features": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
                      "min_samples_leaf": sp_randint(1, 25),
                      "min_samples_split": sp_randint(1, 25),
                      "learning_rate": [0.1, 0.05, 0.02, 0.01],
                      "subsample": [0.1, 0.5, 0.8, 0.9, 1.0]}

        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(bids_path, **kwargs)
        else:
            print 'data frame is already cleaned...'
        train_values = self.df_train[col2fit].values
        target_values = self.df_train['outcome'].values

        pre_dispatch = '2*n_jobs'

        # Fit the grid
        print 'fitting the grid with n_jobs = {}...'.format(n_jobs)
        start = time()
        self.set_model(**kwargs)
        rf_grid = grid_search.RandomizedSearchCV(self.learner,
                                                 parameters,
                                                 n_jobs=n_jobs, verbose=2,
                                                 pre_dispatch=pre_dispatch,
                                                 scoring=score,
                                                 error_score=0,
                                                 n_iter=n_iter)
        rf_grid.fit(train_values, target_values)
        print('Grid search finished')

        print("\n\nGridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(rf_grid.grid_scores_)))
        self.grid_report(rf_grid.grid_scores_, 15)

        print('\n\nBest score = {}'.format(rf_grid.best_score_))
        print('Best params = {}\n\n'.format(rf_grid.best_params_))

if __name__ == "__main__":
    a = clf_learning("data/train.csv")
    #a = clf_learning(saved_pkl='saved_df/test4.pkl')
    #feat_list = ['nbids', 'lfit_m', 'lfit_b']
    #feat_list.append("phone62")
    feat_list = fit_features.test8
    a.grid_search(score="roc_auc", features=feat_list, n_iter=150, n_jobs=7, n_estimators=5000)
