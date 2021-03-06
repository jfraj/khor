import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

# this project
from basemodel import BaseModel
import submission
import fit_features
import hyperparams

class rfClf(BaseModel):

    """Model using random forest classifier."""

    def __init__(self, train_data_fname=None, nrows=None, **kwargs):
        """Initialize the data frame."""
        super(rfClf, self).__init__(train_data_fname, nrows, **kwargs)

    def set_model(self, **kwargs):
        """Set the classifier.

        No criterion parameters since only one choice: mean sqared error
        """
        verbose = kwargs.get('verbose', 0)
        n_estimators = kwargs.get('n_estimators', 200)
        max_depth = kwargs.get('max_depth', None)
        bootstrap = kwargs.get('bootstrap', True)
        min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        min_samples_split = kwargs.get('min_samples_split', 2)
        max_features = kwargs.get('max_features', "auto")
        class_weight = kwargs.get('class_weight', "auto")
        n_jobs = kwargs.get('n_jobs', 1)
        criterion = kwargs.get('criterion', 'entropy')
        random_state = kwargs.get('random_state', 24)

        self.learner = RandomForestClassifier(n_estimators=n_estimators,
                                              max_depth=max_depth,
                                              bootstrap=bootstrap,
                                              min_samples_leaf=min_samples_leaf,
                                              min_samples_split=min_samples_split,
                                              max_features=max_features,
                                              n_jobs=n_jobs,
                                              verbose=verbose,
                                              criterion=criterion,
                                              class_weight=class_weight,
                                              random_state=random_state)
        print('\n\nRandom forest set with parameters:')
        par_dict = self.learner.get_params()
        for ipar in par_dict.keys():
            print('{}: {}'.format(ipar, par_dict[ipar]))
        print('\n\n')

    def fitNscore(self, **kwargs):
        """Fit classifier and produce score and related plots."""
        col2fit = kwargs.get('features')
        # cleaning
        bids_path = kwargs.get('bids_path', 'data/bids.csv')
        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(bids_path, **kwargs)
        print('columns for fit=\n{}'.format(self.df_train.columns))

        test_size = 0.2  # fraction kept for testing
        rnd_seed = 24  # for reproducibility

        #features_train, features_test, target_train, target_test =\
        #    train_test_split(self.df_train[col2fit].values,
        #                     self.df_train['outcome'].values,
        #                     test_size=test_size,
        #                     random_state=rnd_seed)

        sss = StratifiedShuffleSplit(self.df_train['outcome'].values,
                                     n_iter=1,
                                     test_size=test_size,
                                     random_state=rnd_seed)
        for train_index, test_index in sss:
            features_train = self.df_train[col2fit].values[train_index]
            features_test = self.df_train[col2fit].values[test_index]
            target_train = self.df_train['outcome'].values[train_index]
            target_test = self.df_train['outcome'].values[test_index]


        # Fit Classifier
        self.fitModel(features_train, target_train, **kwargs)

        # Predict on the rest of the sample
        print('\nPredicting...')
        predictions = self.learner.predict(features_test)
        probas = self.learner.predict_proba(features_test)

        # Feature index ordered by importance
        ord_idx = np.argsort(self.learner.feature_importances_)
        print("Feature ranking:")
        for ifeaturindex in ord_idx[::-1]:
            print('{0} \t: {1}'.format(col2fit[ifeaturindex],
                                       round(self.learner.feature_importances_[ifeaturindex], 2)))


        # Score
        print('Score={}'.format(self.learner.score(features_test, target_test)))


        # Plots

        # Feature importances
        maxfeat2show = 30 # number of features to show in plots
        importances = self.learner.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.learner.estimators_],
                    axis=0)
        indices = np.argsort(importances)[::-1]
        indices = indices[:min(maxfeat2show, len(indices))]  # truncate if > maxfeat2show
        ordered_names = [col2fit[i] for i in indices]

        fig_import = plt.figure(figsize=(10, 10))
        plt.title("Feature importances, RF")
        plt.barh(range(len(indices)), importances[indices],
                color="b", xerr=std[indices], align="center",ecolor='r')
        plt.yticks(range(len(indices)), ordered_names)
        plt.ylim([-1, len(indices)])
        plt.ylim(plt.ylim()[::-1])
        plt.subplots_adjust(left=0.22)
        fig_import.show()

        # confusion matrix
        cm = confusion_matrix(target_test.astype(int), predictions.astype(int))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.clip(cm_normalized, 0.0, 0.5)

        fig_cm = plt.figure()
        ax_cm = fig_cm.add_subplot(1,1,1)
        im_cm = ax_cm.imshow(cm_normalized, interpolation='nearest')
        plt.title('Normalized confusion mtx, RF')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        fig_cm.colorbar(im_cm)
        fig_cm.show()

        # ROC curve
        # This ones seems to reflect better the LB score
        #false_pos, true_pos, thr = roc_curve(target_test, predictions)
        false_pos, true_pos, thr = roc_curve(target_test, probas[:, 1])
        fig_roc = plt.figure()
        plt.plot(false_pos, true_pos,
                 label='ROC curve (area = %0.2f)' % auc(false_pos, true_pos))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        fig_roc.show()
        raw_input('press enter when finished...')


if __name__ == "__main__":
    #a = rfClf("data/train.csv", nrows=100)
    a = rfClf(saved_pkl='saved_df/test8.pkl')
    #a = rfClf("data/train.csv")
    #feat_list = ['nbids', 'lfit_m', 'lfit_b', 'lfit_r',
    #             'fft_cent', 'fft_freq_std', 'fft_sflat', 'fft_ptp', 'phone62',
    #             'fft_linfit_m', 'fft_linfit_b', 'fft_linfit_r']
    #a.fitNscore(features = feat_list, n_estimators=10000)
    #a.fitNscore(features = feat_list)
    a.fitNscore(features = fit_features.test8, **hyperparams.gb_params['test8'])
    #a.submit(features = feat_list, nbids_rows=1000)
    #a.submit(features = feat_list)
    #a.submit(features = fit_features.test8, **hyperparams.rf_params['test8'])
