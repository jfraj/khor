from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_freq_diff(param, **kwargs):
    """Return freq (robot/non-robot)."""
    nrows = kwargs.get('nrows', None)
    nbid_rows = kwargs.get('nbid_rows', None)

    df = pd.read_csv('data/train.csv', nrows)
    df.set_index('bidder_id', inplace=True)

    df_bids = pd.read_csv('data/bids.csv', nrows=nbid_rows)

    if param in ('ipspl1', 'ipspl2', 'ipspl3', 'ipspl4'):
        df_bids['ipspl1'],df_bids['ipspl2'],df_bids['ipspl3'],df_bids['ipspl4'] =\
            zip(*df_bids["ip"].apply(lambda x: x.split('.')))

    id_robot = df[df['outcome']==1].index.values
    df_bids['outcome'] = df_bids['bidder_id'].apply(lambda x: 1 if x in id_robot else 0)

    by_param = df_bids.groupby(["outcome", param], as_index=False)
    df_byparam = by_param.aggregate(np.size)[['outcome', param, 'bid_id']]
    df_byparam.columns = ['outcome', param, 'counts']

    df_byparam['rob_freq'] = df_byparam['counts']/df_byparam[df_byparam['outcome']==1]['counts'].sum()
    df_byparam['nonrob_freq'] = df_byparam['counts']/df_byparam[df_byparam['outcome']==0]['counts'].sum()

    df_byparam_rob = df_byparam[df_byparam['outcome']==1][[param, 'rob_freq', 'counts']]
    df_byparam_nonrob = df_byparam[df_byparam['outcome']==0][[param, 'nonrob_freq', 'counts']]

    # renaming counts before merge
    df_byparam_rob.rename(columns={'counts': 'rob_counts'},inplace=True)
    df_byparam_nonrob.rename(columns={'counts': 'nonrob_counts'}, inplace=True)

    df_param = pd.merge(df_byparam_rob, df_byparam_nonrob, on=param)

    df_param = df_param.fillna(0)
    df_param['dfreq'] = df_param['rob_freq'] - df_param['nonrob_freq']
    df_param.sort('dfreq', ascending=False, inplace=True)

    print('\nTop ten frequency difference')
    print(df_param.head(n=10))
    print('\nLast ten frequency difference')
    print(df_param.tail(n=10))


if __name__ == "__main__":
    #get_freq_diff('ipspl1', nbid_rows=10000)
    get_freq_diff('ipspl1')
