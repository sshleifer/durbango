"""date utilities for pandas, dataframe equality checkers, mergers and joiners"""
import resource
import sys
import logging
from math import floor, log10

import numpy as np
import pandas as pd
import re
import warnings
from collections import OrderedDict
from pandas.util.testing import assert_frame_equal

from durbango.nb_utils import to_list


logger = logging.getLogger(__name__)


def convert_dates(date_ser, date_format=None, errors='raise'):
    '''returns a pd.Timestamp series. A speed hack for pd.to_datetime'''
    if date_ser.dtype != 'O':
        return date_ser
    unq_dates = date_ser.unique()
    date_map = dict(zip(unq_dates, pd.to_datetime(unq_dates, format=date_format, errors=errors)))
    return date_ser.apply(date_map.get)


def assert_no_overlapping_dates(old_data, to_add):
    '''assert old_data and to_data have no overlapping dates'''
    unq_dates = lambda x: set(x.date if hasattr(x, 'date') else x.index)
    overlap = unq_dates(to_add).intersection(unq_dates(old_data))
    if overlap:
        raise ValueError(u'Data for dates {} has already been ingested'.format(overlap))


def enforce_date_cutoff(df, date_cutoff):
    '''return df[df.date > date_cutoff] if date_cutoff is not None. Handles date index and col '''
    if date_cutoff is None:
        return df
    cutoff = pd.Timestamp(date_cutoff) if isinstance(date_cutoff, str) else date_cutoff
    if isinstance(df.index, pd.DatetimeIndex):
        return df[df.index >= cutoff]
    check_columns(df, 'date')
    return df[df.date > cutoff]


def identify_which_level_is_datetime_index(df_or_index):
    '''Goes through the index levels of the dataframe and identifies the index of the date index

    Throws errors if not a MultiIndex, if there is no date index, if there is more than one
    date index.

    Args:
        df_or_index: (pd.DataFrame or pd.Index) the dataframe or its index

    Returns:
        (int) the index of the level
    '''
    # Pull index & check if multiindex
    index = df_or_index if isinstance(df_or_index, pd.Index) else df_or_index.index

    if not isinstance(index, pd.MultiIndex):
        raise ValueError('Function is only applicable to dataframes with MultiIndex')

    # Identify datetime index / throw if there is extra
    which_levels_are_datetime = [x.is_all_dates for x in index.levels]

    if sum(which_levels_are_datetime) == 1:
        return which_levels_are_datetime.index(True)
    else:
        msg = 'The MultiIndex has {} datetime levels; must have exactly one.'
        raise ValueError(msg.format(sum(which_levels_are_datetime)))


def get_datetime_index(df_or_index):
    '''Checks that the dataframe has a valid datetime index and returns it

    In the case that the dataframe has a MultiIndex, parses through the levels to identify
    which index is the datetime index, as well as verifying that there is one and only one
    datetime index (otherwise will raise ValueError).

    Args:
        df_or_index: (pd.DataFrame or pd.Index) the dataframe or its index

    Returns:
        a pd.Index object of type datetime
    '''
    index = df_or_index if isinstance(df_or_index, pd.Index) else df_or_index.index
    if isinstance(index, pd.MultiIndex):
        datetime_index_level = identify_which_level_is_datetime_index(index)
        return index.get_level_values(datetime_index_level)
    elif isinstance(index, pd.DatetimeIndex):
        return index
    else:
        raise ValueError('Could not find an appropriate datetime index')


def get_smallest_distance_in_days(date, event_dates):
    '''Find the number of days since the most recent event in event_dates'''
    if min(event_dates) >= date:
        return np.nan
    return (date - max(event_dates[event_dates < date])).days


def calc_days_since_last_in_dates(series, dates):
    '''return a series of ts.index: days since the most recent date in dates
    Args:
        series: pd.Series with DatetimeIndex
        dates: pd.DatetimeIndex
    '''
    if len(dates) == 0:
        return pd.Series(np.nan, index=series.index)
    return pd.Series(series.index.map(lambda x: get_smallest_distance_in_days(x, dates)),
                     index=series.index)


def assert_equal_types(left, right):
    '''raise TypeError if not identical'''
    left_types = left.dtypes.sort_index()  # NOTE: I dont understand why this sort_index is needed
    right_types = right.dtypes.sort_index()
    if not (left_types == right_types).all():
        type_df = pd.DataFrame({'left': left_types, 'right': right_types})
        to_show = type_df[type_df.left != type_df.right]
        raise TypeError(u'dtypes different for \n{}\n'.format(to_show))


def are_frames_equal(left, right, verbose=False, **kwargs):
    '''Checks if dataframes are equal (bool)
    Args:
        left: pd.DataFrame
        right: pd.DataFrame
        verbose: if True, prints the AssertionError
        **kwargs: kwargs from pandas.util.assert_frame_equal, like check_less_precise
    '''
    return _check_frame_equality(left, right, verbose=verbose, **kwargs)


def assert_sorted_frames_equal(left, right, **kwargs):
    '''Raise AssertionError if sorted DataFrames are not equal
    Args:
        left: pd.DataFrame
        right: pd.DataFrame
        **kwargs: kwargs from pandas.util.assert_frame_equal, like check_less_precise
    '''
    _check_frame_equality(left, right, raise_err=True, **kwargs)


def _check_frame_equality(left, right, verbose=False, raise_err=False, **kwargs):
    '''Checks if dataframes are equal (bool)
    Args:
        left: pd.DataFrame
        right: pd.DataFrame
        verbose: if True, prints the AssertionError
        raise_err: raise AssertionError if they're not equal
        **kwargs: kwargs from pandas.util.assert_frame_equal, like check_less_precise
    '''
    try:
        assert_frame_equal(left.sort_index().sort_index(axis=1),
                           right.sort_index().sort_index(axis=1), **kwargs)
        return True
    except AssertionError as e:
        if raise_err:
            raise e

        if verbose:
            logger.exception()

        return False


def assert_equal_length(left, right):
    '''assert len(left)==len(right) else print nice error message'''
    make_msg = 'len(left): {} != len(right): {}'.format
    assert len(left) == len(right), make_msg(len(left), len(right))


def horizontal_concat(items, concat_method='join', how='outer', **kwargs):
    '''Memory stingy horizontal concat: uses join or merge instead of pd.concat
    Args:
        items: (iter of pd.DataFrames)
        concat_method: (str) join / merge
        how: (str) passed to df.join or df.merge method
        **kwargs: passed to df.join or df.merge method
    '''
    if not is_iterable(items):
        raise TypeError(u'items should be an iterable, got {} instead'.format(type(items)))
    if len(items) == 0:
        return pd.DataFrame()
    result = items[0]
    for item in items[1:]:
        if item is None or item.empty:
            continue
        result = getattr(result, concat_method)(item, how=how, **kwargs)
    return result


def join_dfs(items, how='outer'):
    '''Horizontal concat using df.join'''
    return horizontal_concat(items, how=how)


def merge_dfs(items, on, how='outer'):
    '''Horizontal concat using df.merge'''
    return horizontal_concat(items, concat_method='merge', on=on, how=how)


def inv_dict(my_map):
    return {v: k for k, v in my_map.items()}


def join_on_multiindex(left, right, how='inner'):
    '''Takes two DataFrames and joins them (always an inner join)

    Use case: pd.DataFrame.join doesn't deal well with dataframes that don't
    have exactly the same MultiIndex levels, e.g. if `left` has (date, client, product)
    but `right` has only (date, product).

    This function takes care of some of the busy work of making sure that join goes smoothly
    by performing an inner join on the common levels.

    (Maybe require that one of the df's levels is a superset of the other?)

    Returns a dataframe with alphabetically sorted levels; also auto-sorts along indexes

    Args:
        left: pd.DataFrame
        right: pd.DataFrame
        how: how to perform the join

    Returns:
        joined pd.DataFrame
    '''

    # Require all index levels to be named
    if (None in left.index.names) or (None in right.index.names):
        raise ValueError(u'All index levels must be named (left: {}, right: {})'.format(
            left.index.names, right.index.names))

    # Require at least one shared level
    if len(set(left.index.names).intersection(right.index.names)) == 0:
        raise ValueError(u'DFs do not share any levels in common (left: {}, right: {})'.format(
            left.index.names, right.index.names))

    # Apply a couple operations:
    #   1) Identify the levels in each df not shared by the other
    #   2) Reset the unshared levels
    #   3) Sort the levels
    not_shared_levels = set([])
    streamlined_dfs = []
    for cur, other in [(left, right), (right, left)]:
        not_shared = list(set(cur.index.names).difference(other.index.names))
        not_shared_levels.update(not_shared)

        df_copy = cur.reset_index(not_shared) if len(not_shared) > 0 else cur
        if isinstance(df_copy.index, pd.MultiIndex):
            df_copy = df_copy.reorder_levels(sorted(df_copy.index.names))

        streamlined_dfs.append(df_copy.sort_index())

    # Join the cleaned levels, re-add the unshared levels, and sort levels
    return_df = streamlined_dfs[0].join(streamlined_dfs[1], how=how)
    return_df = return_df.set_index(sorted(list(not_shared_levels)), append=True)

    if isinstance(return_df.index, pd.MultiIndex):
        return_df = return_df.reorder_levels(sorted(return_df.index.names))

    return return_df.sort_index()


def join_dfs_multiindex(items, how='inner'):
    '''apply join_on_multiindex to a list of dataframes'''
    result = items[0]
    for item in items[1:]:
        result = join_on_multiindex(result, item, how=how)
    return result


def zip_to_series(a, b):
    '''Check that two iterables are of equal length, and then make a {a:b} Series'''
    if len(a) != len(b):
        raise ValueError('zip_to_series got iterables of uneven '
                         'length: {}, {}'.format(len(a), len(b)))
    return pd.Series(dict(zip(a, b)))

def safe_concat(*args, **kwargs):
    if 'sort' in kwargs:
        return pd.concat(*args, **kwargs)
    else:
        return pd.concat(*args, sort=False, **kwargs)

def _maskna_on_ends_df(df, mode):
    '''Id nan's on ends of df and return mask where they are false and all else True

    Please make sure the dataframe is already sorted; we do not check for you.

    Args:
        df: (pd.DataFrame) df to be operated upon
        mode: (before | after | both) drop only nan's before first nonnull,
              after last nonull, or both

    Returns:
        boolean mask defining which elements to keep
    '''
    if mode not in ['before', 'after', 'both']:
        raise ValueError('`mode` must be in [before|after|both]')

    if not isinstance(df, pd.DataFrame):
        raise ValueError('requires dataframe as input')

    # Chaining to try to limit memory use...
    # Count nonnull values through index t --> drop all rows w/ 0s and get min / max remaining index
    before_min_ix = (~df.isnull()).expanding().sum().where(lambda x: x > 0).dropna().index.min()
    after_min_ix = ((~df.isnull().iloc[::-1])  # This reverses the dataframe for us
                    .expanding().sum().where(lambda x: x > 0).dropna().index.max())

    min_nonnull_elements = min([df[col].dropna().shape[0] for col in df])
    if min_nonnull_elements > 0:
        before_mask = df.index >= before_min_ix
        after_mask = df.index <= after_min_ix
    else:  # e.g., if everything nan
        before_mask = pd.Series(False, index=df.index)
        after_mask = pd.Series(False, index=df.index)

    if mode == 'before':
        return before_mask
    elif mode == 'after':
        return after_mask
    else:
        return (before_mask & after_mask)


def dropna_on_ends(df_or_series, subset=None, mode='both'):
    '''Drop NAs at the beginning and at the end of the series

    Args:
        df_or_series: (pd.DataFrame|pd.Series) data to operate upon
        subset: (None|list) columns to dropna for
        mode: (before | after | both) drop only nan's before first nonnull,
              after last nonull, or both

    Returns:
        type matches input type; just without nans
    '''
    if isinstance(df_or_series, pd.Series):
        mask = _maskna_on_ends_df(df_or_series.to_frame(), mode)
    else:
        subset = df_or_series.columns if subset is None else subset
        mask = _maskna_on_ends_df(df_or_series[subset], mode)
    return df_or_series.loc[mask]


def pct_positive(x):
    '''return (x.dropna() > 0).mean()  # % positive when not zero'''
    return (x.dropna() > 0).mean()  # % positive when not zero


def int_to_float(df, exceptions=()):
    '''convert integer columns to float'''
    mask = ((df.dtypes == int) & (~df.columns.map(lambda x: x in exceptions)))
    df.loc[:, mask] = df.loc[:, mask].astype(float)
    return df


def get_last_percentile(df):
    '''For yesterdays volume was in the {}th percentile
    Args:
        df: DataFrame or Series
    '''
    ranks = df.rank()
    return ranks.iloc[-1] / ranks.shape[0]


def _split_into_max_bins(ser, max_bins):
    '''qcut series into between <max_bins> and 3 bins. Try to get as many bins as possible.'''
    if ser.dtype == 'object':
        raise ValueError('must supply a numeric input')
    for q in reversed(range(3, max_bins + 1)):
        try:
            result = pd.qcut(ser, q=q, labels=False)
            return result - np.floor(q / 2)  # center bins around 0
        except ValueError:
            pass
    return np.sign(ser)


def centered_max_bin_qcut(df, max_bins=5):
    '''Bin each series of DataFrame into between 5 and 3 bins'''
    if isinstance(df, pd.Series):
        return _split_into_max_bins(df, max_bins)
    else:
        return df.apply(lambda x: _split_into_max_bins(x, max_bins))  # to each series


def df_memory_usage(df):
    '''return the number of megabytes a dataframe/Series uses including index'''
    mem_in_bytes = df.memory_usage(index=True)
    while hasattr(mem_in_bytes, 'shape') and mem_in_bytes.shape:
        mem_in_bytes = mem_in_bytes.sum()
    return mem_in_bytes / 1024. / 1024.


def unzscore(z, original_y):
    '''change a zscore prediction back to dollars using original Series. Remember rolling!'''
    return (z * original_y.std()) + original_y.mean()


def n_levels(idx):
    '''1 if not MultiIndex else x.levelshape'''
    return len(getattr(idx, 'levels', [1]))


def save_str(str_to_write, path):
    '''save str_to_write to a file at path'''
    with open(path, 'w') as f:
        f.write(str_to_write)
    return path


def check_columns(df, required_columns):
    '''Check that required columns are in DataFrame, raise KeyError if any are missing.

    Args:
        df: dataframe instance
        required_columns: (string or list) the columns that must be in the DataFrame
    '''
    if isinstance(required_columns, str):
        required_columns_set = {required_columns}
    elif is_iterable(required_columns):
        required_columns_set = set(required_columns)
    else:
        raise TypeError(u'Unsupported type for required '
                        u'columns: "{}"'.format(type(required_columns)))

    if not isinstance(df, pd.DataFrame):
        raise TypeError(u'Expected a dataframe, but got: "{}"'.format(type(df)))

    existing_columns = set(df.columns)
    missing_columns = required_columns_set - existing_columns

    if missing_columns:
        raise KeyError(u'The given dataframe is missing the following '
                       u'required columns: "{}"'.format(missing_columns))


def replace_infs_with(df, use_value=np.nan):
    '''replace inf and -inf with use_value (default=np.nan)'''
    return df.replace([np.inf, -np.inf], use_value)


def replacer(df, maxes):
    for c in df.columns:
        df[c] = df[c].replace(np.inf, maxes[c])
    return df
def has_nan(x):
    '''Returns True if the given array or dataframe has a nan value'''
    if isinstance(x, pd.Series) or isinstance(x, pd.DataFrame):
        x = x.values
    return bool(np.sum(np.isnan(x)))


def equalize_axes(left, right):
    '''Takes two dataframes, combines their columns and indexes
    and sets combined index and columns for each
    Args:
        left: (pd.DataFrame)
        right: (pd.DataFrame)
    '''
    new_cols = left.columns.union(right.columns)
    new_idx = left.index.union(right.index)
    return left.loc[new_idx, new_cols], right.loc[new_idx, new_cols]


def filter_blank_dfs(dfs):
    '''Filters a list of dataframes to remove those without data'''
    is_blank = lambda df: df is not None and len(df.dropna())
    return filter(is_blank, dfs)


def select_subset(X=None, feature_names=None, patterns=None):
    '''Extracts a subset of X based on which of the columns match one of the patterns

    Args:
        X: (ndarray, DataFrame, or None); the data set to be filtered.
            None if just filtering a list of names
        feature_names: (list str); if X is not a pd.DataFrame, provide feature_names here
        patterns: (list, string, None) a set of regex patterns or just substrings;
                  if None, returns original df; raises error if none match
    Returns:
        ndarray or DataFrame depending on input type of X
    '''
    # Enforce input constraints
    if isinstance(X, pd.DataFrame) and feature_names is not None:
        raise ValueError('feature_names should be None when X is a pd.DataFrame')

    if isinstance(X, np.ndarray) and not isinstance(X, pd.DataFrame) and feature_names is None:
        raise ValueError('feature_names must be provided when X is not a pd.DataFrame')

    if isinstance(X, pd.DataFrame) and isinstance(X.columns, pd.MultiIndex):
        raise NotImplementedError('does not work on multi-level columns at the moment')

    if patterns is not None:
        feature_names = feature_names if not isinstance(X, pd.DataFrame) else X.columns
        matching_features = np.array([any([re.match(pat, feat) or (pat in feat)
                                           for pat in to_iterable(patterns)])
                                      for feat in feature_names])

        if X is None:
            return [fn for (fn, mask) in zip(feature_names, matching_features) if mask]
        elif isinstance(X, np.ndarray):
            sub_X = X[:, matching_features]
        elif isinstance(X, pd.DataFrame):
            sub_X = X.loc[:, matching_features]

        if sub_X.size == 0:
            raise ValueError('no columns matched any of the patterns: {}'.format(patterns))
        else:
            return sub_X
    else:
        return X


def nan_get(ser, dct) -> pd.Series:
	"""Refactor as left merge if large"""
	return ser.apply(lambda x: dct.get(x, np.nan))


def left_merge_series(df, left_col, df_or_ser, ser_name=None):
    if isinstance(df_or_ser, pd.Series): df_or_ser = df_or_ser.to_frame(ser_name)
    return df.merge(df_or_ser, left_on=left_col, right_index=True, validate='m:1',
                    how='left')


def get_index_names(df):
    '''Gets the column names of a dataframe's index'''
    return to_list(df.index.name) if df.index.name else df.index.names


def count_df(ser, sort_index_first=False):
    '''return pd.DataFrame({'n': ser.value_counts(), 'frac': ser.value_counts(normalize=True)})'''
    cts = pd.value_counts(ser)
    counts = pd.DataFrame({'n': cts, 'frac': cts/cts.sum()})
    if sort_index_first:
        counts = counts.sort_index()
    counts['cum_frac'] = counts.frac.cumsum()
    return counts.round(3)


def transform_with_nan(df, transform):
    '''Return the result of a transform while propagating NaNs.'''
    null_indices = df.isnull()
    transformed = transform(df)
    transformed[null_indices] = np.nan
    return transformed


def weighted_avg(df, col, weight_col='vol', epsilon=0):
    '''find weighted_avg of col values based on weights from weight_col
    Args:
        df: (pd.DataFrame) data that has col and weight_col
        col: (str) name of the column which will be averaged
        weight_col: (str) name of the column which will be used as weights
        epsilon: (float) value to add to weights to ensure not zero
    '''
    df[weight_col] += epsilon
    return (df[col] * df[weight_col]).sum() / df[weight_col].sum()


def round_number(num, num_digits=2):
    '''Round a number to num_digits significant digits to make it look nice'''
    if num == 0:
        return 0
    if np.isnan(num):
        return num
    total_num_digits = int(floor(log10(abs(num))))
    corrected_value = np.sign(num) * round(num, num_digits - 1 - total_num_digits)
    return corrected_value * np.sign(num)


def get_num_cols(df):
    '''Get columns where data is numeric'''
    return df.select_dtypes(
        include=[float, int, bool, np.dtype('int32'), np.dtype('int64'),
                 np.dtype('float32'), np.dtype('float64'), np.dtype('uint8')]
    ).columns


def validate_number_type(ser):
    '''Raise if the datatype in a series is numeric'''
    if not isinstance(ser, pd.Series):
        raise TypeError(u'Expected pd.Series, received "{}" instead'.format(type(ser)))
    if not np.issubdtype(ser.dtype, np.number):
        raise TypeError(u'Expected a numeric type and received "{}" instead'.format(ser.dtype))


def raise_if_not_df_or_series(df):
    '''make sure that last_rows object is pd.DataFrame / pd.Series'''
    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError('rows should be pd.DataFrame/pd.Series), got {} instead'.format(type(df)))


def is_y_one_column(y):
    '''Returns True if the given array is one dimensional or only has one column'''
    empty = y.size == 0
    too_big = (len(y.shape) > 2) or (len(y.shape) == 2 and y.shape[1] != 1)
    return not (empty or too_big)


def quantile_sample(df, target_col, n_sample=200, n_quantile=20, seed=0):
    '''sample rows of a dataframe in a stratified fashion based on continuous target column
    Args:
        df: dataframe from which we want to sample
        target_col: (str) name of column we want to use for percentile based sampling
        n_sample: (int) number of total samples we want to select
        n_quantile: (int) number of quantiles for binning (must be >= n_sample)
        seed: (int) random seed for sampling
    '''
    samples_df = df.copy()

    # floor to get even sample distribution on bins
    n_bin_sample = int(np.floor(n_sample / n_quantile))
    samples_df['quantile_bins'] = pd.qcut(samples_df[target_col], n_quantile)
    quantiles = samples_df['quantile_bins'].unique()
    samples_to_draw = {q: n_bin_sample for q in quantiles}

    # add sample in case n_sample is not integer multiple of n_quantile
    n_add = n_sample - np.sum(samples_to_draw.values())
    quantiles_increase = np.random.choice(quantiles, n_add, replace=False)
    for q in quantiles_increase:
        samples_to_draw[q] += 1

    samples_df = (samples_df.groupby('quantile_bins')[target_col]
                  .apply(lambda x: x.sample(samples_to_draw[x.name], random_state=seed)))

    sample_rows_idx = samples_df.reset_index('quantile_bins', drop=True).index
    return df.loc[sample_rows_idx].sort_index()


def abs_sum(df_or_series):
    '''type agnostic sum of absolute values'''
    return df_or_series.abs().sum()


def to_list(item):
    '''Wrap object in list; to safely pass (possibly single) items to functions expecting lists'''
    return to_iterable(item, force_list=True)


def to_iterable(symbols, force_list=False):
    '''Transform strings or iterables of strings into something that can be safely iterated
    Args:
        symbols: input string or iterable to convert
        force_list: (bool) convert iterable to list
    '''
    if is_iterable(symbols) and force_list:
        return [symbol for symbol in symbols]
    elif is_iterable(symbols) or symbols is None:
        return symbols
    return [symbols]


def frequency_string(n_precedents, expected_periods):
    '''Get the outlier trading frequency:
        Args:
            n_precedents: # of times the outlier event happened
            expected_periods: (list of tuples) of average # of periods,
                e.g. [('month', 2), ('year', 5)]
    '''
    ordered_period_names = OrderedDict(zip(('A', 'Q', 'M', 'W', 'D'),
                                           ('year', 'quarter', 'month', 'week', 'day')))
    if n_precedents == 0:
        return 'Never before'
    elif n_precedents == 1:
        return 'One precedent'
    else:
        for t, p in expected_periods:
            if n_precedents == p and p == 'D':  # trades every day
                return 'Every day'
            elif n_precedents <= p:
                return 'Once in {:.0f} {}(s)'.format(np.round(p / n_precedents),
                                                     ordered_period_names.get(t))
    raise ValueError(
        'failed for {} precedents and expected periods{}'.format(n_precedents,
                                                                 expected_periods)
    )


def humanize(n, precision=1, absolute=True):
    '''display absolute value of n with correct units
    Args:
        n: (float) to be displayed
        precision: (int) rounding precision
        absolute: (bool) should absolute the value

    >>> humanize(123456)
    => '123.4K'
    '''
    is_negative = n < 0
    n = np.abs(n)
    scaler = 1

    _dividers = {
        '': 1,
        'K': 1e3,
        'mm': 1e6,
        'B': 1e9
    }

    curr_symbols = {v: k for k, v in _dividers.items()}

    while n >= 1e3:
        n /= 1e3
        scaler *= 1e3
    rounded_abs_num = np.round(n, precision)
    formatted_value = '{:.{p}f}{}'.format(rounded_abs_num, curr_symbols.get(scaler), p=precision)
    if (not absolute) and (is_negative) and (rounded_abs_num != 0):
        formatted_value = '-' + formatted_value
    return formatted_value


def is_iterable(obj):
    '''Tests whether an object is iterable. Returns False for strings'''
    return hasattr(obj, '__iter__')


def iterable_to_pd_index(potentially_nested_iterable, index_names):
    '''make a multi index if potentially_nested_iterable is nested else a normal index'''
    if is_iterable(index_names) and len(index_names) > 1:
        # from_arrays supports both lists of tuples and lists of lists
        # if iterable is not nested will get traceback saying you screwed up length of names
        return pd.MultiIndex.from_tuples(potentially_nested_iterable, names=index_names)
    elif is_iterable(index_names) and len(index_names) == 1:
        return pd.Index(potentially_nested_iterable, name=index_names[0])
    else:
        return pd.Index(potentially_nested_iterable, name=index_names)


def convert_to_categoricals(df, columns, inplace=False):
    '''Convert requested columns to categoricals'''
    if not inplace:
        df = df.copy()

    for c in columns:
        df[c] = pd.Categorical(df[c])

    if not inplace:
        return df


def binarize_y_n_cols(df, columns):
    ''' return df.replace({c: {'Y': 1., 'N': 0., 'T': 1., 'F': 0.} for c in columns})'''
    return df.replace({c: {'Y': 1., 'N': 0., 'T': 1., 'F': 0.} for c in columns})


def capitalize_sentence(string):
    '''Capitalize first letter, but do not affect rest, e.g., 'the Brown Fox' -> 'The Brown Fox' '''
    return string[:1].upper() + string[1:]


def capitalize_sentence_series(string_series):
    '''Capitalize first letter, but do not affect rest, e.g., 'the Brown Fox' -> 'The Brown Fox' '''
    return string_series.str[:1].str.upper() + string_series.str[1:]


def _get_most_common_field_value_for_group(df, groupby, fld):
    '''Helper to get most frequent value for one field of a groupby'''
    cts = df.groupby([groupby, fld]).size().to_frame('n_rows')
    pairs = cts.sort_values('n_rows', ascending=False).reset_index().drop_duplicates(
        [groupby], keep='first')
    ser = pairs.set_index(groupby)[[fld]]
    return ser


def get_most_common_field_values_for_group(df, groupby, flds):
    '''Get most common value for each groupby, field combo.
    eg most common settlement_date for FNCL 3.5 is 1/27/2013
    Args:
        df: dataframe
        groupby: (str) colname to get most common field name for
        flds: (lst or str) columns to retrieve most common value for

    Returns:
        most_freq_unstacked: (df) with one row for each unique value of df[groupby]
        and one column for each field filled with most common values
    '''
    return pd.concat([_get_most_common_field_value_for_group(df, groupby, fld)
                      for fld in to_list(flds)], axis=1).rename_axis(groupby)


def map_to_num(ser):
    '''Replace a columns existing content with frequency rank E.g. replace names with their frequences'''
    mapping = ser.value_counts().reset_index().reset_index().set_index('index')['level_0']
    return ser.apply(mapping.get).astype(str)


def create_equal_sized_groups(input_df, ordered_gb, agg_col='qty', min_grp_size=1e6):
    '''Create most granular groups possible s.t. each group has a total <agg_col> of min_size
    Args:
        input_df: df to make groupings from
        ordered_gb: (iterable, sorted granular -> coarse) will try to make a group at [0], then [1],
            until has passed through whole list
        agg_col: (str) col used to decide how big a group is
        min_grp_size: (numeric) minimum size for a group

    Returns:
        result_df: (pd.DataFrame) with ordered_gb columns replaced with Other for too small groups
    '''
    result_df = input_df.copy()
    for i in range(1, len(ordered_gb) + 1):
        gb = ordered_gb[:i]
        group_totals = result_df.groupby(gb)[agg_col].sum()
        sum_over_grps = group_totals.to_frame('group_total').reset_index()
        result_df = result_df.merge(sum_over_grps, how='left')
        result_df.loc[result_df['group_total'] < min_grp_size, gb] = 'Other'
        result_df = result_df.drop(['group_total'], 1)
    return result_df


def set_memory_limit(limit_gb):
    '''Cap the amount of memory available to Python; useful to prevent getting OOM-killed by system
    (e.g., allows within-python debugging since will raise MemoryError rather than getting SIGKILL)

    Specifically, sets the soft limit on virtual memory allowed; to see whether or not it worked,
    call `ulimit -a` in the shell and look at the row for `vmemory`.

    Args:
        limit_gb: (int) desired memory limit; if None, sets it to the hard cap (for address space
                  defaults to "unlimited")
    '''
    if not (u'linux' in sys.platform):
        warnings.warn(u'This will not do anything since this is not a linux platform')
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit_bytes = int(limit_gb / 1.0e9) if limit_gb is not None else hard
    return resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))


class MemoryLimit(object):
    def __init__(self, limit_gb):
        '''Context manager limiting memory available to Python; tries to reset limit on exit'''
        self.limit_gb = limit_gb
        self.limit_bytes = limit_gb / 1.0e9

    def __enter__(self):
        '''Establish the temporary memory limit'''
        return set_memory_limit(self.limit_bytes)

    def __exit__(self, *args):
        '''Try to reset the memory limit (e.g., set it to the hard cap)'''
        return set_memory_limit(None)


def set_to_zero(df, set_to_zero_if_abs_below):
    '''Round values to zero if they are below a cutoff
    Args:
        df: (pd.DataFrame | pd.Series)
        set_to_zero_if_abs_below: (float) round all values below this cutoff in magnitude to zero
    Returns:
        pd.Series of rounded values
    '''
    zero_df = df.copy()
    zero_df[zero_df.abs() < set_to_zero_if_abs_below] = 0
    return zero_df


def calc_pct_of_absolute(values, epsilon=1e6):
    '''calculates value / sum of abs values, adds epsilon if sum of abs values is zero'''
    denominator = values.abs().sum()
    if denominator == 0:
        denominator = epsilon
    return values / denominator


def select_from_first_level_of_multiindex(df, selectors, axis=0):
    '''Select some values from the first level of a multi index using Index Slicer'''
    idx = pd.IndexSlice
    if axis == 0:
        return df.loc[idx[:, selectors], :]
    elif axis == 1:
        return df.loc[:, idx[:, selectors]]
    else:
        raise ValueError('axis must be 0 or 1 got {}'.format(axis))


def check_level_names(df, index_names, column_names):
    '''Assert that index and column level names are as expected'''
    if set(df.index.names) != set(index_names) or set(df.columns.names) != set(column_names):
        raise AssertionError(u'Expected dataframe with index ({}) and columns ({}), but found '
                             u'index ({}) and columns ({})'
                             .format(index_names, column_names, df.index.names, df.columns.names))
    return None


def numeric_almost_equal(a, b, thresh=1e-7):
    '''Are two numbers almost equal, broadcasts.'''
    return np.abs(a - b) <= thresh


def pandas_identity_mask(ndframe):
    '''A mask that does not alter the ndframe, e.g., ndframe.loc[pandas_identity_mask] == ndframe'''
    return [True] * ndframe.shape[0]


def stack_all(df, dropna=True, new_name=None, to_frame=False):
    '''Functional wrapper around pd.DataFrame.stack; stacks everything into a series'''
    num_levels = len(df.columns.names)
    return_series = df.stack(range(num_levels), dropna=dropna).rename(new_name)
    return return_series if not to_frame else return_series.to_frame()


def abs_sort_values(col, ascending=False):
    '''Allows sorting by absolute value; usage: df[abs_sort_values('my_column')]'''

    def make_sorted_index(df):
        '''Generated function to sort dataframes by provided column'''
        return df[col].abs().sort_values(ascending=ascending).index

    return make_sorted_index()


def combine_coefficients_in_feature_df(df, suffix_split, axis=0):
    '''If you have a bunch of lags, e.g., ar___01, ar___03, average across like-stems'''

    def drop_suffix(x):
        '''e.g., ar___01 --> ar'''
        return x.split(suffix_split)[0]

    axis_name = u'index' if axis == 0 else u'columns'
    return df.rename(**{axis_name: drop_suffix}).mean(level=0, axis=axis)


def make_outlier_quantile_mask(series, lower_quantile, upper_quantile, series_for_threshes=None):
    '''Create mask that is True for all values below / above boundaries

    Args:
        series: (pd.Series) the series to derive mask from
        lower_quantile: (float) lower bound
        upper_quantile: (float) upper bound
        series_for_threshes: (pd.Series) optionally, use a second series to get thresholds from

    Returns:
        (pd.Series) mask with same index as `series` that is True where for values outside of
        thresholds
    '''
    if series_for_threshes is None:
        series_for_threshes = series

    lower_thresh = series_for_threshes.quantile(lower_quantile)
    upper_thresh = series_for_threshes.quantile(upper_quantile)
    return (series < lower_thresh) | (series > upper_thresh)


def safe_fillna(df_or_ser, limit, method):
    '''Allow fillna to have 0 as limit value'''
    if limit == 0:
        # this is equivalent to not filling, and was removed in new pandas version
        # all other args will be ignored
        return df_or_ser
    else:
        return df_or_ser.fillna(limit=limit, method=method)


def add_level(df, new_value, axis=1, new_level_name=None):
    '''Switch to multi index for columns/index if new_name is not None or []
    Args:
        df: (pd.DataFrame)
        new_value: (str) the string that would be used in the new multi index level
        axis: (int) {0 or 'index', 1 or 'columns'}, default 1. Where the new level will be added
        new_level_name: (str) the name of the new level, if desired
    '''
    source = df.columns if axis == 1 else df.index
    source_level_names = source.names
    new_index = pd.MultiIndex.from_tuples([[new_value] + list(to_iterable(x)) for x in source],
                                          names=[new_level_name] + source_level_names)
    df_copy = df.copy()
    df_copy.set_axis(axis, new_index)  # modification in place
    return df_copy


def drop_level(df, levels_to_drop, axis=1):
    '''hack to chain the droplevel method. see droplevel docs for pd.DataFrame
    Args:
        df: a pd.DataFrame
        levels_to_drop: strings (names of levels), or ints (indices of levels)
        axis: default=1, 1 means columns, 0 means index
    '''
    df = df.copy()
    if axis == 1:
        df.columns = df.columns.droplevel(levels_to_drop)
    elif axis == 0:
        df.index = df.index.droplevel(levels_to_drop)
    else:
        raise ValueError('axis must be either 0 or 1. found {}'.format(axis))
    return df


def fld_getter(fld):
    """Allows setting pd.DataFrame.get_id = fld_getter('id')"""
    return lambda df, val: df[df[fld] == val]


def drop_duplicate_cols(df):
    """Drop duplicate column names from a DataFrame, keeping the first."""
    seen = set()
    renamer = {}
    cols = df.columns.tolist()
    for i, c in enumerate(df.columns):
        if c in seen:
            renamer[i] = c + '_dup'
            cols[i] = renamer[i]
        else:
            seen.add(c)
    df.columns = cols
    df = df.drop(renamer.values(), 1)
    return df


def drop_zero_variance_cols(df):
    keep_col_mask = df.apply(lambda x: x.nunique()) > 1
    return df.loc[:, keep_col_mask]


def groupby_softmax(df, grouper, agg_col='yhat'):
    """Transform df[agg_col] with softmax for each group."""
    log_yhat = np.exp(df[agg_col])
    sum_grp_yhat = df.groupby(grouper).log_yhat.transform('sum')
    return log_yhat / sum_grp_yhat
