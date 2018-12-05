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


sections_path = Path('report_items')
clean_path = Path('reports_items_clean')
if not clean_path.exists():
    clean_path.mkdir(exist_ok=True)

to_do = len(list(sections_path.glob('*.csv')))
nlp = spacy.load('en', disable=['ner'])
nlp.max_length = 6000000

vocab = Counter()
t = total_tokens = 0
stats = []
start = time()
done = 1
for filing in sections_path.glob('*.csv'):
    filing_id = int(filing.stem)
    items = pd.read_csv(filing).dropna()
    if done % 50 == 0:
        duration = time() - start
        to_go = (to_do - done) * duration / done
        print(f'{done:>5}\t{format_time(duration)}\t{total_tokens / duration:,.0f}\t{format_time(to_go)}')
    clean_doc = []
    for _, (item, text) in items.iterrows():
        doc = nlp(text)
        for s, sentence in enumerate(doc.sents):
            clean_sentence = []
            if sentence is not None:
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
                stats.append([filing_id, item, s, t, len(clean_sentence)])
                if len(clean_sentence) > 0:
                    clean_doc.append([item, s, ' '.join(clean_sentence)])
    parsed = pd.DataFrame(clean_doc, columns=['item', 'sentence', 'text'])
    clean_file = clean_path / (filing.stem + '.csv')
    parsed.to_csv(clean_file)
    done += 1

    token_count = pd.DataFrame(vocab.most_common(), columns=['token', 'n'])
    token_count.to_parquet('items_token_count.parquet')
    corpus_stats = pd.DataFrame(stats, columns=['filing_id', 'item', 'sentence_id', 'tokens', 'clean_tokens'])
    with pd.HDFStore('items_vocab.h5') as store:
        store.put('corpus_stats', corpus_stats)
