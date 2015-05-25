import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import learning_curve

from rf import rfClf

class clf_learning(rfClf):
    """
    Class that will contain learning functions
    """
    def learn_curve(self, score='roc_auc', **kwargs):
        """
        Plots the learning curve
        """
        verbose = kwargs.get('verbose', 0)
        nsizes = kwargs.get('nsizes', 8)
        waitNshow = kwargs.get('waitNshow', True)
        cv = kwargs.get('waitNshow', 5)

        col2fit = kwargs.get('features')
        # cleaning
        bids_path = kwargs.get('bids_path', 'data/bids.csv')
        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(bids_path, **kwargs)
        print('columns for fit=\n{}'.format(self.df_train.columns))

        train_values = self.df_train[col2fit].values
        target_values = self.df_train['outcome'].values

        ##Create a list of nsize incresing #-of-sample to train on
        train_sizes = [x / float(nsizes) for x in range(1, nsizes + 1)]
        print 'training will be performed on the following sizes'
        print train_sizes

        n_jobs = 1
        print '\n\nlearning with njobs = {}\n...\n'.format(n_jobs)

        self.set_model(**kwargs)

        train_sizes, train_scores, test_scores =\
            learning_curve(self.learner,
                           train_values, target_values, cv=cv,
                           n_jobs=n_jobs, train_sizes=train_sizes,
                           scoring=score)

        ## Plotting
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
        return {'fig_learning': fig, 'train_scores': train_scores, 'test_scores':test_scores}


if __name__ == "__main__":
    a = clf_learning("data/train.csv")
    feat_list = ['nbids', 'lfit_m', 'lfit_b']
    #a.learn_curve('precision', nbids_rows=1000, features=feat_list)
    #a.learn_curve('roc_auc', nbids_rows=100000, features=feat_list)
    #a.learn_curve(nbids_rows=1000000, features=feat_list)
    a.learn_curve("precision",features=feat_list)
