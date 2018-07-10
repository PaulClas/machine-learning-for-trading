#!/usr/bin/env python
#-*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import pandas as pd
from pathlib import Path

path = Path('user_agents')


useragents = pd.Series('\n'.join([f.read_text().strip() for f in path.glob('*.txt')]).split('\n')).str.strip().drop_duplicates()
print(len(useragents))
Path('user_agents.txt').write_text('\n'.join(useragents.tolist()))
exit()
df = pd.read_json('company_transcripts/GOOG.json')
print(df.info())