# durbango
python utilities for interactive matrix wrangling with minimal keystrokes

### Installation

```
pip install durbango
```


### Tutorials
WIP


### Collecting Out of Fold Predictions
see `cross_val_predict` and `cross_val_predict_proba`


### Avoiding Generators
instead of calling list on generators or `dict.keys` all the time, which is annoying for seeing results and letting progress bars
know how long things are, durbango provides wrappers that reacquaint python objects with the human visual system.
```
lmap
dhead
keys
vals
```
## Adding Magic Properties
These are not included by default, but the package has utilities to make it easy to add them to your notebook.


#### Field getters to avoid .loc in pandas
Often in pandas I repeatedly query the same field like `df[df['favorite_color'] == 'red']`.
the
The fix is `fld_getter`, which makes a lambda to query any field for equality.

At the top of your notebook define:

```
pd.DataFrame.where_color_is = fld_getter('favorite_color')
```
and then throughout the notebook, you can simply write
`df.where_color_is('red')` and everything besides 'red' autocompletes.
This doesn't cause any issues unless you have a naming collision caused by a column named `where_color_is`. Don't do that!

Another useful magic property that I use for NLP is
```
pd.DataFrame.where_text_contains = lambda df, pat: df[df['text'].str.contains(pat)]
```



### Pytorch
Instead of writing `tensor.detach().cpu().numpy()` over and over, `to_arr` will do this.
You can also make it a magic property of `torch.Tensor`



### Tensorboard -> DataFrame
```
from durbango.tensorboard_parser import *
parsed = parse_tf_events_file('/home/shleifer/conv_ai/tboard_logs/events.out.tfevents.1564283968.shleifer-gpu-3-vm')
This is a dataframe where each column is a metric and each row is an epoch.
```

### Docs ToDo:
- imports.py
- feature importance
- `cross_val_predict_proba`
- alerting: `send_sms` and `send_email`
- `count_df`
- `dsort` and `asort`
