import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import randint as sp_randint
from time import time
from operator import itemgetter

from sklearn import grid_search
from sklearn.learning_curve import learning_curve

from rf import rfClf
import hyperparams
import fit_features


class clf_learning(rfClf):

    """Class that will contain learning functions."""

    def learn_curve(self, score='roc_auc', **kwargs):
        """Plot the learning curve."""
        nsizes = kwargs.get('nsizes', 8)
        waitNshow = kwargs.get('waitNshow', True)
        cv = kwargs.get('waitNshow', 5)
        n_jobs = kwargs.get('n_jobs', 1)

        col2fit = kwargs.get('features')
        # cleaning
        bids_path = kwargs.get('bids_path', 'data/bids.csv')
        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(bids_path, **kwargs)
        print('columns for fit=\n{}'.format(self.df_train.columns))

        train_values = self.df_train[col2fit].values
        target_values = self.df_train['outcome'].values

        # Create a list of nsize incresing #-of-sample to train on
        train_sizes = [x / float(nsizes) for x in range(1, nsizes + 1)]
        print 'training will be performed on the following sizes'
        print train_sizes

        #n_jobs = 1
        print '\n\nlearning with njobs = {}\n...\n'.format(n_jobs)

        self.set_model(**kwargs)

        train_sizes, train_scores, test_scores =\
            learning_curve(self.learner,
                           train_values, target_values, cv=cv,
                           n_jobs=n_jobs, train_sizes=train_sizes,
                           scoring=score)

        # Plotting
        fig = plt.figure()
        plt.xlabel("Training examples")
        plt.ylabel(score)
        plt.title("Learning Curves, random forest")
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        plt.legend(loc="best")
        print 'Learning curve finisher'
        if waitNshow:
            fig.show()
            raw_input('press enter when finished...')
        return {'fig_learning': fig, 'train_scores': train_scores,
                'test_scores': test_scores}

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
                      "criterion": ["gini", "entropy"],
                      "max_features": [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
                      "min_samples_leaf": sp_randint(1, 25),
                      "min_samples_split": sp_randint(1, 25),
                      "bootstrap": [True, False],
                      "class_weight": [None, "auto", "subsample"]}

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
    #feat_list.append('url_vasstdc27m7nks3')
    #feat_list.append('ipspl1_165')
    #feat_list.append('auc_jqx39')
    feat_list = fit_features.test8

    #a.learn_curve('precision', nbids_rows=1000, features=feat_list)
    #a.learn_curve('roc_auc', nbids_rows=100000, features=feat_list)
    #a.learn_curve(nbids_rows=1000000, features=feat_list)
    #a.learn_curve(None, n_jobs=5, features=feat_list, **hyperparams.rf_params['test2'])
    #a.grid_search(nbids_rows=10000, features=feat_list)
    a.grid_search(score="precision", features=feat_list, n_iter=150, n_jobs=7, n_estimators=5000)
