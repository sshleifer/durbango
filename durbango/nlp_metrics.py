from sacrebleu import corpus_bleu

def bleu_df(gen_df, a='gold', b='hf_es', c='mar'):
    import pandas as pd
    gold = gen_df[a].tolist()
    hf = gen_df[b].tolist()
    mar = gen_df[c].tolist()
    return pd.Series({'marian': corpus_bleu(mar, [gold]).score, 'hf': corpus_bleu(hf, [gold]).score, 'hf-mar':corpus_bleu(hf, [mar]).score}).round(2)
