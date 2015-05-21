import unittest
import sys
sys.path.append('../')


class Test_basemodel(unittest.TestCase):
    def test_init_with_fname(self):
        from basemodel import BaseModel
        a = BaseModel('../data/train.csv', verbose=False)
        self.assertSequenceEqual(a.df_train.shape, (2013, 4))

    def test_init_with_fname_with_nrows(self):
        from basemodel import BaseModel
        a = BaseModel('../data/train.csv', nrows=100, verbose=False)
        self.assertSequenceEqual(a.df_train.shape, (100, 4))

    def test_prepare(self):
        from basemodel import BaseModel
        a = BaseModel('../data4testing/train.csv', verbose=False)
        a.prepare_data('../data4testing/bids.csv', verbose=False)
        self.assertSequenceEqual(a.df_train.shape, (2, 176))
