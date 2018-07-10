#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Stefan Jansen'

from pathlib import Path
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
from time import sleep
from selenium import webdriver
from pprint import pprint
import pandas as pd
from furl import furl

driver = webdriver.Firefox()
title_pattern = re.compile(r'(Q\d): (\d{2})-(\d{2})-(\d{2})')

SA_URL = 'https://seekingalpha.com/'
string = re.compile('Earnings Call Transcript')

next_page = True
page = 5
while next_page:
    url = f'{SA_URL}/earnings/earnings-call-transcripts/{page}'
    page_url = urljoin(SA_URL, url)
    print(page_url)
    driver.get(page_url)
    response = driver.page_source
    print(page)
    page += 1
    soup = BeautifulSoup(response, 'lxml')
    for tag in soup.find_all(name='a', string=string):
        sleep(.1)
        meta = {}
        content = {}

        link = tag.attrs.get('href')
        article_url = furl(urljoin(SA_URL, link)).add({'part': 'single'})
        print(article_url)
        driver.get(article_url.url)
        article = driver.page_source
        Path('test_article.html').write_text(article)

        soup = BeautifulSoup(article, 'lxml')

        h1 = soup.find('h1', itemprop='headline').text
        # title = soup.find('div', class_='title').text
        # match = title_pattern.match(title)
        # if match:
        #     meta['quarter'] = match.group(0)
        #     meta['month'] = int(match.group(1))
        #     meta['day'] = int(match.group(2))
        #     meta['year'] = int(match.group(3))
        try:
            meta['company'] = h1[:h1.find('(')].strip()
        except Exception as e:
            print('No company', e, h1)
        try:
            meta['symbol'] = h1[h1.find('(') + 1:h1.find(')')]
        except Exception as e:
            print('No symbol', e, h1)
        print(pd.Series(meta))

        participants = {}
        for type in ['Executives', 'Analysts']:
            p = soup.find('strong', string=type).parent
            participants[type] = []
            for participant in p.find_next_siblings('p'):
                if participant.find('strong'):
                    break
                else:
                    participants[type].append(participant.text)
        driver.close()

        #
        # flag = False
        # for paragraph in soup.find_all('p'):
        #     strong = paragraph.find(tag='strong')
        #     if strong:
        #         t = strong.text
        #         if t.lower().startswith('copyright policy'):
        #             break
        #         elif 'executive' in t or 'analyst' in t:
        #             flag = 'participant'
        #             participants[strong.text] = []
        #         else:
        #             flag = 'content'
        #             content[strong.text] = []
        #     else:
        #         if flag == 'participant':
        #             participants[]
        #
        #
        #     else:
        #
        #
        # transcript = [p.text for p in soup.find_all('p')]
        # pprint(transcript[:20])
        # pprint(transcript[20:])
        # print(page, h1, len(transcript))
        # driver.close()
        # exit()
