import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

# this project
from basemodel import BaseModel
import submission
import fit_features
import hyperparams

class gbClf(BaseModel):

    """Model using random forest classifier."""

    def __init__(self, train_data_fname=None, nrows=None, **kwargs):
        """Initialize the data frame."""
        super(gbClf, self).__init__(train_data_fname, nrows, **kwargs)

    def set_model(self, **kwargs):
        """Set the classifier."""
        verbose = kwargs.get('verbose', 0)
        n_estimators = kwargs.get('n_estimators', 3000)
        max_depth = kwargs.get('max_depth', 3)
        min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        min_samples_split = kwargs.get('min_samples_split', 2)
        max_features = kwargs.get('max_features', None)
        learning_rate = kwargs.get('learning_rate', 0.1)
        random_state = kwargs.get('random_state', 24)

        self.learner = GradientBoostingClassifier(n_estimators=n_estimators,
                                                  max_depth=max_depth,
                                                  learning_rate=learning_rate,
                                                  min_samples_leaf=min_samples_leaf,
                                                  min_samples_split=min_samples_split,
                                                  max_features=max_features,
                                                  verbose=verbose,
                                                  random_state=random_state)
        print('\n\nGradient Boosting set with parameters:')
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

        features_train, features_test, target_train, target_test =\
            train_test_split(self.df_train[col2fit].values,
                             self.df_train['outcome'].values,
                             test_size=test_size,
                             random_state=rnd_seed)

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
        print('(Self) Score={}'.format(self.learner.score(features_test, target_test)))

        # Plots

        # Feature importances
        maxfeat2show = 30 # number of features to show in plots
        importances = self.learner.feature_importances_
        #std = np.std([tree.feature_importances_ for tree in self.learner.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]
        indices = indices[:min(maxfeat2show, len(indices))]  # truncate if > maxfeat2show
        ordered_names = [col2fit[i] for i in indices]

        fig_import = plt.figure(figsize=(10, 10))
        plt.title("Feature importances, GB")
        #plt.barh(range(len(indices)), importances[indices],
        #        color="b", xerr=std[indices], align="center",ecolor='r')
        plt.barh(range(len(indices)), importances[indices],
                 color="b", align="center")
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
        plt.title('Normalized confusion mtx, GB')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        fig_cm.colorbar(im_cm)
        fig_cm.show()

        # ROC curve
        # This ones seems to reflect better the LB score
        false_pos, true_pos, thr = roc_curve(target_test, predictions)
        #false_pos, true_pos, thr = roc_curve(target_test, probas[:, 1])
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
    #a = gbClf("data/train.csv", nrows=100)
    #a = gbClf("data/train.csv")
    #a = gbClf(saved_pkl='saved_df/test2.pkl')
    a = gbClf(saved_pkl='saved_df/test4.pkl')
    #a.set_model()
    #a.fitNscore(features = fit_features.test2, **hyperparams.rf_params['test2'])
    feat_list = ['nbids', 'lfit_m', 'lfit_b']
    #sub_country_list = ['ctry_us', ]
    #sub_phone_list= ["phone{}".format(x) for x in [119,17,46,62,13,115,122,237,389,528]]
    sub_phone_list= ["phone62",]
    #sub_merch_list = ["mer_mobi"]
    #feat_list.extend(sub_country_list)
    feat_list.extend(sub_phone_list)
    #feat_list.extend(sub_merch_list)
    #feat_list.append('url_vasstdc27m7nks3')
    #feat_list.append('ipspl1_165')
    #feat_list.append('auc_jqx39')
    #a.fitNscore(features = feat_list, **hyperparams.gb_params['test3'])
    a.fitNscore(features = fit_features.test4, **hyperparams.gb_params['test4'])
    #a.submit(features = feat_list)
