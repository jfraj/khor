import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# scikit-learn
from sklearn.externals import joblib

# This project
import clean
import fit_features


class BaseModel(object):
    """Contain members and function for all trainers."""
    def __init__(self, train_data_fname=None, nrows=None, **kwargs):
        """Turn data in pandas dataframe."""
        verbose = kwargs.get('verbose', True)
        # Define the classifier and regressor variables
        self.learner = False
        self.fitted = False
        self.iscleaned = False
        self.df_train = None

        learner_pkl = kwargs.get('learner_pkl', False)
        if learner_pkl:
            print('\nUsing pickled learner from {}'.format(learner_pkl))
            self.learner = joblib.load(learner_pkl)
            self.fitted = True
            self.iscleaned = False
            return

        if 'saved_df' in kwargs.keys():
            print('Get saved train data from {}'.format(kwargs['saved_df']))
            self.df_train = pd.read_hdf(kwargs['saved_df'], 'dftest')
            print('Train data frame has shape')
            print(self.df_full.shape)
            self.iscleaned = True
            return

        if 'saved_pkl' in kwargs.keys():
            print('Get pickled train data from {}'.format(kwargs['saved_pkl']))
            self.df_train = pd.io.pickle.read_pickle(kwargs['saved_pkl'])
            print('Train data frame has shape')
            print(self.df_train.shape)
            self.iscleaned = True
            return

        if train_data_fname is None:
            print 'Data will not be created by reading csv file'
            self.df_train is None
            return

        skiprows = kwargs.get('skiprows', None)
        if skiprows is not None:
            if verbose:
                print('Skipping {} rows'.format(skiprows))
            skiprows = range(1, skiprows)

        self.df_train = pd.read_csv(train_data_fname, nrows=nrows,
                                    skiprows=skiprows)

        if verbose:
            print('Training data frame has shape')
            print(self.df_train.shape)

    def fill_country_categories(self, df, df_bids, country, **kwargs):
        """Fill the dataframe with country category"""
        verbose = kwargs.get('verbose', False)
        if verbose:
            print('adding country {}'.format(country))
        cat_name = 'ctry_{}'.format(country)
        ctry_counts = df_bids[df_bids['country']==country].groupby(["bidder_id"]).size()
        # join_axes to keep only the index of df
        df = pd.concat([df, ctry_counts.to_frame(name=cat_name)],
                       axis=1, join_axes=[df.index])
        df[cat_name].fillna(0, inplace=True)
        return df

    def fill_merch_categories(self, df, df_bids, merch, **kwargs):
        """Fill the merchandise type category"""
        verbose = kwargs.get('verbose', False)
        merch_dict = fit_features.get_merchandise_rename_dict(inverted=True)
        if verbose:
            print('adding {}'.format(merch))
        cat_name = merch
        merch_counts = df_bids[df_bids['merchandise']==merch_dict[merch]].groupby(["bidder_id"]).size()
        df = pd.concat([df, merch_counts.to_frame(name=cat_name)],
                       axis=1, join_axes=[df.index])
        df[cat_name].fillna(0, inplace=True)
        return df

    def fill_url_categories(self, df, df_bids, url, **kwargs):
        """Fill the url type category"""
        verbose = kwargs.get('verbose', False)
        if verbose:
            print('adding {}'.format(url))
        cat_name = url
        url_counts = df_bids[df_bids['url']==url[4:]].groupby(["bidder_id"]).size()
        df = pd.concat([df, url_counts.to_frame(name=cat_name)],
                       axis=1, join_axes=[df.index])
        df[cat_name].fillna(0, inplace=True)
        return df

    def fill_auction_categories(self, df, df_bids, auction, **kwargs):
        """Fill the url type category"""
        verbose = kwargs.get('verbose', False)
        if verbose:
            print('adding {}'.format(auction))
        cat_name = auction
        auc_counts = df_bids[df_bids['auction']==auction[4:]].groupby(["bidder_id"]).size()
        df = pd.concat([df, auc_counts.to_frame(name=cat_name)],
                       axis=1, join_axes=[df.index])
        df[cat_name].fillna(0, inplace=True)
        return df

    def fill_phone_categories(self, df, df_bids, phone, **kwargs):
        """Fill the phone type category"""
        verbose = kwargs.get('verbose', False)
        if verbose:
            print('adding {}'.format(phone))
        cat_name = phone
        phone_counts = df_bids[df_bids['device']==phone].groupby(["bidder_id"]).size()
        df = pd.concat([df, phone_counts.to_frame(name=cat_name)],
                       axis=1, join_axes=[df.index])
        df[cat_name].fillna(0, inplace=True)
        return df

    def fill_ipsplit_categories(self, df, df_bids, ipspl, **kwargs):
        """Fill the sub ip address category"""
        verbose = kwargs.get('verbose', False)
        if verbose:
            print('adding {}'.format(ipspl))
        cat_name = ipspl
        ipspl_counts = df_bids[df_bids[ipspl[:6]]==ipspl[7:]].groupby(["bidder_id"]).size()
        df = pd.concat([df, ipspl_counts.to_frame(name=cat_name)],
                       axis=1, join_axes=[df.index])
        df[cat_name].fillna(0, inplace=True)
        return df


    def fill_features_from_bids(self, df, df_bids, **kwargs):
        """Fill self.df_train with features for fitting."""
        verbose = kwargs.get('verbose', True)
        features = kwargs.get('features', 'all')
        ignore_clean = kwargs.get('ignore_clean', False)

        if self.iscleaned and not ignore_clean:
                print('Data is already cleaned prepared... no filling done')
                return
        if verbose:
            print('Setting bidders as index of self.df_train')
        df = df.set_index('bidder_id')

        if verbose:
            print('Creating list of bids for each bidders')
        by_bidders = df_bids.groupby('bidder_id')
        df['bidtimes'] =\
            by_bidders['time'].apply(lambda x: x.tolist())
        df['bidtimes'] =\
            df['bidtimes'].apply(lambda x: [x, ] if type(x) is float else x)

        if features == 'all' or any("nbids" in s for s in features):
            if verbose:
                print('Adding number of bids')
            df['nbids'] =\
                df['bidtimes'].apply(lambda x: np.count_nonzero(~np.isnan(x)))
        if features == 'all' or any("lfit" in s for s in features):
            if verbose:
                print('Adding linear fit related values')
            df['lfit_m'], df['lfit_b'] =\
                zip(*df['bidtimes'].apply(clean.get_linearfit_features))
            # replace nan with -1
            df.loc[:, 'lfit_b'].fillna(-1, inplace=True)
            df.loc[:, 'lfit_b'].fillna(-1, inplace=True)

        if features == 'all' or any("ctry" in s for s in features):
            if verbose:
                print('Adding country features')
            for ictry in fit_features.FULL_COUNTRY_LIST:
                if "ctry_{}".format(ictry) not in features:
                    continue
                df = self.fill_country_categories(df, df_bids, ictry)

        if features == 'all' or any("phone" in s for s in features):
            if verbose:
                print('Adding phone features')
            for iphone in features:
                if iphone.find('phone') != 0:
                    continue
                print('filling {}'.format(iphone))
                df = self.fill_phone_categories(df, df_bids, iphone)

        if features == 'all' or any("mer_" in s for s in features):
            if verbose:
                print('Adding merchandise features')
            for imerch in features:
                if imerch.find('mer_') != 0:
                    continue
                print('filling {}'.format(imerch))
                df = self.fill_merch_categories(df, df_bids, imerch)

        if features == 'all' or any("url_" in s for s in features):
            if verbose:
                print('Adding url features')
            for iurl in features:
                if iurl.find('url_') != 0:
                    continue
                print('filling {}'.format(iurl))
                df = self.fill_url_categories(df, df_bids,
                                                iurl, verbose=True)

        if features == 'all' or any("auc_" in s for s in features):
            if verbose:
                print('Adding auction features')
            for iauc in features:
                if iauc.find('auc_') != 0:
                    continue
                print('filling {}'.format(iurl))
                df = self.fill_auction_categories(df, df_bids,
                                                iauc, verbose=True)

        if features == 'all' or any("ipspl" in s for s in features):
            if verbose:
                print('Splitting ip addresses to create new columns')
            df_bids['ipspl1'],df_bids['ipspl2'],df_bids['ipspl3'],df_bids['ipspl4']=\
                zip(*df_bids["ip"].apply(lambda x: x.split('.')))

        if features == 'all' or any("ipspl1_" in s for s in features):
            if verbose:
                print('Adding ip split 1 features')
            for ispl1 in features:
                if ispl1.find('ipspl1_') != 0:
                    continue
                print('filling {}'.format(ispl1))
                df = self.fill_ipsplit_categories(df, df_bids,
                                                ispl1, verbose=True)


        return df

    def prepare_dataframe(self, df, bids_fname, **kwargs):
        """Make the dataframe 'df' reading for fitting.

        df is filled with features determined from csv file 'bids_fname'
        """
        nbids_rows = kwargs.get('nbids_rows', None)  # All by default
        features = kwargs.get('features', 'all')
        verbose = kwargs.get('verbose', True)
        if verbose and nbids_rows is not None:
            print 'preparing datafram with {} rows'.format(nbids_rows)
        df_bids = pd.read_csv(bids_fname, nrows=nbids_rows)

        # Exclude rows that have no bids
        # though something may eventually be done with same adresses of url...
        df = df[df['bidder_id'].isin(df_bids['bidder_id'])]

        df = self.fill_features_from_bids(df, df_bids, **kwargs)

        # Removing useless columns
        if features is not 'all':
            if verbose:
                print('Removing useless columns...')
            to_keep = features + ['outcome', 'bidder_id']
            for icol in df.columns:
                if icol not in to_keep:
                    df.drop(icol, axis=1, inplace=True)
        return df

    def prepare_data(self, bids_fname, **kwargs):
        """Prepare this class' object for fitting."""
        self.df_train = self.prepare_dataframe(self.df_train,
                                               bids_fname, **kwargs)
        self.iscleaned = True

    def prepareNsave(self, col2save, **kwargs):
        """Clean and prepare data and save pickle it."""
        save_name = kwargs.get('save_name', 'saved_df/default.pkl')
        bids_fname = kwargs.get('bids_fname', 'data/bids.csv')
        print('\nWill prepare and save the following column')
        print(col2save)
        print('\nFrom the bids file: {}'.format(bids_fname))
        print('\nPreparing the data...')
        self.prepare_data(bids_fname, features = col2save, **kwargs)
        print('Pickling dataframe...')
        self.df_train.to_pickle(save_name)
        print('columns saved: {}'.format(self.df_train.columns))
        print('Done saving dataframe in {}'.format(save_name))

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

    def show_feature(self, feature):
        """Plot the given feature."""
        fig = plt.figure()
        self.df_train[feature].hist()
        fig.show()
        raw_input("press enter when finished...")

if __name__ == "__main__":
    import os
    data_dir = 'data'
    #data_dir = 'data4testing'
    train_path = os.path.join(data_dir, 'train.csv')
    bids_path = os.path.join(data_dir, 'bids.csv')
    a = BaseModel(train_path)
    #a.df_train = a.prepare_data(a.df_train, bids_path)
    feat_list = ['nbids', 'lfit_m', 'lfit_b', 'ctry_us', 'phone4',
                 'mer_offi', 'url_vasstdc27m7nks3', 'ipspl1_105', 'auc_lx0hm']
    #a.prepare_data(bids_path, features=feat_list, nbids_rows=10000)
    #a.prepare_data(bids_path, features=feat_list)
    #a.df_train = a.prepare_data(a.df_train, bids_path, nbids_rows=10000)
    #print(a.df_train.head())
    a.prepareNsave(fit_features.test2, save_name='saved_df/test2.pkl')
