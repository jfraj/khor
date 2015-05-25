import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib

# this project
from basemodel import BaseModel
import submission
import fit_features

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
        n_estimators = kwargs.get('n_estimators', 80)
        max_depth = kwargs.get('maxdepth', None)
        bootstrap = kwargs.get('bootstrap', True)
        min_samples_leaf = kwargs.get('min_samples_leaf', 1)
        min_samples_split = kwargs.get('min_samples_split', 2)
        max_features = kwargs.get('max_features', "auto")
        class_weight = kwargs.get('class_weight', "auto")
        n_jobs = kwargs.get('n_jobs', 1)
        random_state = kwargs.get('random_state', 24)

        self.learner = RandomForestClassifier(n_estimators=n_estimators,
                                              max_depth=max_depth,
                                              bootstrap=bootstrap,
                                              min_samples_leaf=min_samples_leaf,
                                              min_samples_split=min_samples_split,
                                              max_features=max_features,
                                              n_jobs=n_jobs,
                                              verbose=verbose,
                                              class_weight=class_weight,
                                              random_state=random_state)
        print('\n\nRandom forest set with parameters:')
        par_dict = self.learner.get_params()
        for ipar in par_dict.keys():
            print('{}: {}'.format(ipar, par_dict[ipar]))
        print('\n\n')

    def fitModel(self, values2fit, targets, **kwargs):
        """Fit the Classifier."""
        if self.fitted:
            print('Already fitted...')
            return
        # Classifier
        self.set_model(**kwargs)

        print('Fitting on values with shape:')
        print(values2fit.shape)
        print('\nFitting...')
        self.learner.fit(values2fit, targets)
        self.fitted = True
        print('Done fitting!')

    def fitNscore(self, **kwargs):
        """Fit classifier and produce score and related plots"""

        col2fit = kwargs.get('features')
        # cleaning
        bids_path = kwargs.get('bids_path', 'data/bids.csv')
        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(bids_path, **kwargs)
        print('columns for fit=\n{}'.format(self.df_train.columns))

        test_size = 0.3  # fraction kept for testing
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
        plt.title("Feature importances, reg")
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
        plt.title('Normalized confusion mtx, reg')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        fig_cm.colorbar(im_cm)
        fig_cm.show()

        # ROC curve
        false_pos, true_pos, thr = roc_curve(target_test, predictions)
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

    def submit(self, **kwargs):
        """Prepare submission file"""

        col2fit = kwargs.get('features')
        bids_path = kwargs.get('bids_path', 'data/bids.csv')
        test_path = kwargs.get('test_path', 'data/test.csv')

        # cleaning
        if not self.iscleaned:
            print 'Preparing the data...'
            self.prepare_data(bids_path, **kwargs)
        print('columns for fit=\n{}'.format(self.df_train.columns))

        # Fit Classifier
        self.fitModel(self.df_train[col2fit].values,
                      self.df_train['outcome'].values, **kwargs)

        # Prepare test sample
        df_test = pd.read_csv(test_path)
        all_bidders = df_test['bidder_id']
        print(df_test.columns)
        df_test = self.prepare_dataframe(df_test, bids_path,
                                         ignore_clean=True,
                                         features=col2fit)

        # Predict on the test sample
        print('\nPredicting...')
        predictions = self.learner.predict(df_test[col2fit].values)

        # Write submission
        fsubname = 'submission.csv'
        fsub = open(fsubname, 'w')
        fsub.write('bidder_id,prediction\n')
        predicted = []
        for ibidder, ipred in zip(df_test.index, predictions):
            fsub.write('{},{}\n'.format(ibidder, ipred))
            predicted.append(ibidder)
        # Now fill the unpredicted as non-robot
        for iunpredicted_bidder in (set(all_bidders) - set(predicted)):
            fsub.write('{},0.0\n'.format(iunpredicted_bidder))
        fsub.close()

        # check validation file
        if not submission.is_submission_ok(fsubname):
            print('\n\n!!!!ERROR with the submission file!')
            return
        print('Ready to submit the file {}'.format(fsubname))


if __name__ == "__main__":
    #a = rfClf("data/train.csv", nrows=100)
    a = rfClf("data/train.csv")
    #a.prepare_data(a.df_train, 'data/bids.csv', nbids_rows=100000)
    #a.prepare_data(a.df_train, 'data/bids.csv')
    #print(a.df_train.head())
    #a.set_model()
    #a.fitNscore(features = ['nbids', 'lfit_m', 'lfit_b'], nbids_rows=100000)
    feat_list = ['nbids', 'lfit_m', 'lfit_b']
    #sub_country_list= feat_list.extend(fit_features.get_ctry_full_feature_list())
    #sub_country_list = ["ctry_{}".format(x) for x in ['id','au','uk','my','us','th','sg','za','in','fr']]
    sub_country_list = ['ctry_in', ]
    feat_list.extend(sub_country_list)
    # This list has the most frequency disparity between robot-nonrobot
    sub_phone_list= ["phone{}".format(x) for x in [119,17,46,62,13,115,122,237,389,528]]
    #feat_list.extend(sub_phone_list)
    feat_list.append('phone46')
    sub_merch_list = fit_features.get_merch_full_feature_list()
    feat_list.extend(sub_merch_list)
    sub_url_list= ["url_{}".format(x) for x in ['vasstdc27m7nks3',
                                                'lacduz3i6mjlfkd',
                                                '4dd8ei0o5oqsua3',
                                                'hzsvpefhf94rnlb',
                                                'ds6j090wqr4tmmf',
                                                'vwjvx8n5d6yjwlj',
                                                'xosquuqcro853d7',
                                                '96ky12gxeqflpwz',
                                                '1bltvi87id7pau1',
                                                'g2sohb92odayedy']]
    #feat_list.extend(sub_url_list)
    feat_list.append('url_vasstdc27m7nks3')
    feat_list.append('ipspl1_165')
    feat_list.append('auc_jqx39')
    #a.fitNscore(features = feat_list, nbids_rows=100000)
    a.fitNscore(features = feat_list)
    #a.submit(features = feat_list, nbids_rows=1000)
    #a.submit(features = feat_list)
