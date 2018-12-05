from pathlib import Path
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import spacy
from time import time
from pprint import pprint

pd.set_option('display.expand_frame_repr', False)
plt.style.use('fivethirtyeight')


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)


# txt = Path('reports_clean', '1.txt').read_text()
# pprint(len(txt.split('\n')))
# pprint(len(txt.split('°')))
# exit()
# df = pd.read_parquet('token_count.parquet')
# print(df.head())
# print(df.info())
# with pd.HDFStore('vocab.h5') as store:
#     df = store.get('corpus_stats')
#     print(df.clean_tokens.sum())
# exit()

with pd.HDFStore('../../data/assets.h5') as store:
    stocks = store['quandl/wiki/stocks']

filing_path = Path('reports')
clean_path = Path('reports_clean')

filings = pd.read_csv('report_index.csv').rename(columns=str.lower)
filings = (filings[filings.ticker.isin(stocks.symbol)].index + 1).tolist()
to_do = len(filings) - len(list(clean_path.glob('*.txt')))

nlp = spacy.load('en', disable=['ner'])
nlp.max_length = 6000000

vocab = Counter()
t = total_tokens = 0
stats = []
start = time()
done = 1
for filing in filing_path.glob('*.txt'):
    filing_id = int(filing.stem)
    clean_file = clean_path / filing.name
    if filing_id not in filings or clean_file.exists():
        continue

    doc = nlp(filing.read_text())
    if done % 50 == 0:
        duration = time() - start
        to_go = (to_do - done) * duration / done
        print(f'{done:>5}\t{format_time(duration)}\t{total_tokens / duration:,.0f}\t{format_time(to_go)}')
    clean_doc = []
    for s, sentence in enumerate(doc.sents):
        clean_sentence = []
        if sentence is not None and len(sentence) > 10:
            for t, token in enumerate(sentence, 1):
                if not any([token.is_stop,
                            token.is_digit,
                            not token.is_alpha,
                            token.is_punct,
                            token.is_space,
                            token.lemma_ == '-PRON-',
                            token.pos_ in ['PUNCT', 'SYM', 'X']]):
                    clean_sentence.append(token.text.lower())
            total_tokens += t
            vocab.update(clean_sentence)
            stats.append([filing_id, s, t, len(clean_sentence)])
            if len(clean_sentence) > 0:
                clean_doc.append(' '.join(clean_sentence))
    clean_file.write_text('\n'.join(clean_doc))
    done += 1

token_count = pd.DataFrame(vocab.most_common(), columns=['token', 'n'])
token_count.to_parquet('token_count.parquet')
corpus_stats = pd.DataFrame(stats, columns=['filing_id', 'sentence_id', 'tokens', 'clean_tokens'])
with pd.HDFStore('vocab.h5') as store:
    store.put('corpus_stats', corpus_stats)
