import unittest

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal, assert_series_equal

from .nb_utils import _sort
from .cross_val import cross_val_predict_proba_df, cross_val_predict_proba
from durbango.filesystem import get_git_rev
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor, DummyClassifier


class TestShit(unittest.TestCase):
    def test_blind_descending_sort(self):
        test_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=[0, 1])
        assert_frame_equal(test_df.pipe(_sort),  # intentional use of pipe
                           test_df.sort_values('a', ascending=False))
        test_ser = test_df['b']
        assert_series_equal(test_ser.pipe(_sort),
                            test_df['b'].sort_values(ascending=False))

        # MultiIndex
        test_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=[0, 1])
        test_df.columns = pd.MultiIndex.from_tuples([('a', 'ANOTHER LEVEL'), ('b', 'ANOTHER')])
        assert_frame_equal(test_df.pipe(_sort),  # intentional use of pipe
                           test_df.sort_values(('a', 'ANOTHER LEVEL'), ascending=False))
        # Bad Inputs
        with self.assertRaises(AttributeError):
            _sort([1, 2, 3, 4])

    def test_get_git_rev(self):
        get_git_rev()
        
class TestCrossVal(unittest.TestCase):
    
    
    def test_cvp_alignment(self):
        mock_arr = np.array(range(25)).reshape(5, 5)
        mock_df = pd.concat([pd.DataFrame(mock_arr, columns=['a', 'b', 'c', 'd', 'e'])] * 5)
        
        clf = DummyClassifier()
        #preds = cross_val_predict_proba(clf, mock_df[['a', 'b', 'c', 'd']], mock_df['e'])
        mock_df_range_idx = mock_df.reset_index(drop=True)

        preds_df = cross_val_predict_proba_df(clf, mock_df_range_idx[['a', 'b', 'c', 'd', ]],
                                              mock_df_range_idx['e'] > 5,task_type='binary', stratified=True, random_state=12)

        #mock_df.index[-1] = 100  # will never get filled
        assert pd.value_counts(mock_df.index).max() > 1
        with self.assertRaises(IndexError):
            preds_df_dup = cross_val_predict_proba_df(clf, mock_df[['a', 'b', 'c', 'd']], 
                                              mock_df['e'] > 5, task_type='binary', stratified=True, random_state=12)
            
            
        
        
        #import ipdb; ipdb.set_trace()
