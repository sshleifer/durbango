import pandas as pd


def read_and_clean_kibana_file(path = '/Users/shleifer/Downloads/csv.txt'):
    df = pd.read_csv(path)
    df.columns = ['day', 'model', 'n'] # n is downloads
    df['n'] = df['n'].str.replace(',','').astype(int)
    return df
    # Example: df[df.model.str.contains('t5')].groupby('day')['n'].sum()
