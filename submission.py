import sys
import pandas as pd
import collections

def is_submission_ok(filename):
    """Check validity of submission file"""

    samplefname = 'data/sampleSubmission.csv'

    # Data frame from sample submission
    df_sample = pd.read_csv(samplefname)

    # Data frame of candidate submission
    df = pd.read_csv(filename)

    # Number of predictions
    if df.shape[0] != df_sample.shape[0]:
        print('\n\nError: wrong # of submissions')
        return False

    # Number of field
    if df.shape[1] != df_sample.shape[1]:
        print('\n\nError: wrong # of submission field')
        return False

    # Check that all bidders are the same
    # collections.Counter returns False if there are doubles
    if not collections.Counter(df_sample['bidder_id']) == collections.Counter(df['bidder_id']):
        return False

    print('\nSubmission looks ok!')
    return True

if __name__ == "__main__":
    is_submission_ok('submission.csv')
