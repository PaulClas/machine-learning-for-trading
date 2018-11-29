#!/usr/bin/env python
#-*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'


from pathlib import Path
import numpy as np
import pandas as pd


experiment_path = Path('experiments')
# dtm params
min_dfs = [50, 100, 250, 500]
max_dfs = [.1, .25, .5, 1.0]
binarys = [True, False]

results = pd.DataFrame()
coherence = pd.DataFrame()
for min_df in min_dfs:
    for max_df in max_dfs:
        for binary in binarys:
            vocab_path = experiment_path / str(min_df) / str(max_df) / str(int(binary))
            try:
                # results = pd.concat([results,
                #                      pd.read_csv(vocab_path / 'result.csv')])
                coherence = pd.concat([coherence,
                                       (pd.read_csv(vocab_path / 'coherence.csv', header=[0,1])
                                       .stack()
                                       .reset_index()
                                       .rename(columns={'level_0': 'num_topics', 'level_1': 'passes'})
                                        .assign(min_df=min_df,
                                                max_df=max_df,
                                                binary=binary))
                                       ])
            except FileNotFoundError:
                print('Missing:', min_df, max_df, binary)

with pd.HDFStore('results.h5') as store:
    # store.put('perplexity', results)
    store.put('coherence', coherence)
print(coherence.info())
# print(results.info())

