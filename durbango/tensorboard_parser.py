from collections import defaultdict
import pandas as pd
from pathlib import Path

from .nb_utils import find_mode_len


def is_interesting_tag(tag):
    """Unused filter for which tags to include."""
    return ('val' in tag or 'train' in tag)


def events_file_to_df(path_to_events_file) -> pd.DataFrame:
    """Parse a tfevents file to a dataframe."""
    metrics = parse_tf_events_file(path_to_events_file)
    n_epochs = find_mode_len(metrics)
    metrics_df = pd.DataFrame({k: v for k, v in metrics.items() if len(v) == n_epochs})
    # TODO(SS): this should warn about what keys it is tossing
    return metrics_df


def parse_tf_events_file(path_to_events_file) -> defaultdict:
    """Get every {tag: [simple_val,..]} in the events file."""
    import tensorflow as tf
    metrics = defaultdict(list)
    for e in tf.train.summary_iterator(str(path_to_events_file)):
        for v in e.summary.value:
            metrics[v.tag].append(v.simple_value)
    return metrics


def get_tf_events_file(path):
    p = Path(path)
    files = list(p.rglob('*events*'))
    return files
