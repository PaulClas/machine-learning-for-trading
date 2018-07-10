#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

pd.set_option('display.expand_frame_repr', False)

date_pattern = re.compile(r'(\d{2})-(\d{2})-(\d{2})')
quarter_pattern = re.compile(r'(\bQ\d\b)')
string = re.compile('Earnings Call Transcript')

transcript_path = Path('transcripts')

companies = []
for html_file in (transcript_path / 'html').glob('*.html'):
    html = Path(html_file).read_text()
    soup = BeautifulSoup(html, 'lxml')

    meta, participants, content = {}, [], []
    h1 = soup.find('h1', itemprop='headline').text
    meta['company'] = h1[:h1.find('(')].strip()
    meta['symbol'] = h1[h1.find('(') + 1:h1.find(')')]

    title = soup.find('div', class_='title').text
    match = date_pattern.search(title)
    if match:
        m, d, y = match.groups()
        meta['month'] = int(m)
        meta['day'] = int(d)
        meta['year'] = int(y)

    match = quarter_pattern.search(title)
    if match:
        meta['quarter'] = match.group(0)

    qa = 0
    speaker_types = ['Executives', 'Analysts']
    for header in [p.parent for p in soup.find_all('strong')]:
        text = header.text.strip()
        if text.lower().startswith('copyright'):
            continue
        elif text.lower().startswith('question-and'):
            qa = 1
            continue
        elif any([type in text for type in speaker_types]):
            for participant in header.find_next_siblings('p'):
                if participant.find('strong'):
                    break
                else:
                    participants.append([text, participant.text])
        else:
            p = []
            for participant in header.find_next_siblings('p'):
                if participant.find('strong'):
                    break
                else:
                    p.append(participant.text)
            content.append([header.text, qa, '\n'.join(p)])

    companies.append([meta['company'], meta['symbol']])
    # path = transcript_path / 'parsed' / meta['symbol']
    # if not path.exists():
    #     path.mkdir(parents=True, exist_ok=True)
    # pd.DataFrame(content, columns=['speaker', 'q&a', 'content']).to_csv(path / 'content.csv', index=False)
    # pd.DataFrame(participants, columns=['type', 'name']).to_csv(path / 'participants.csv', index=False)
    # pd.Series(meta).to_csv(path / 'earnings.csv')
pd.DataFrame(companies, columns=['name', 'ticker']).sort_values('name').to_csv('companies.csv', index=False)