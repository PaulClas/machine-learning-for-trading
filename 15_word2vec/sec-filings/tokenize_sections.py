from pathlib import Path
import pandas as pd
from collections import Counter
import spacy
from time import time


def format_time(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)


sections_path = Path('sections')
clean_path = Path('sections_clean')
if not clean_path.exists():
    clean_path.mkdir(exist_ok=True)

text_files = list(sections_path.glob('*.csv'))
cleaned_files = [int(f.stem) for f in clean_path.glob('*.csv')]
to_do = len(text_files) - len(cleaned_files)
nlp = spacy.load('en', disable=['ner'])
nlp.max_length = 6000000

vocab = Counter()
t = total_tokens = 0
stats = []
start = time()
done = 1
for text_file in text_files:
    file_id = int(text_file.stem)
    clean_file = clean_path / f'{file_id}.csv'
    if clean_file.exists():
        continue
    items = pd.read_csv(text_file).dropna()
    if done % 100 == 0:
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
                # vocab.update(clean_sentence)
                # stats.append([file_id, item, s, t, len(clean_sentence)])
                if len(clean_sentence) > 0:
                    clean_doc.append([item, s, ' '.join(clean_sentence)])
    (pd.DataFrame(clean_doc,
                  columns=['item', 'sentence', 'text'])
     .to_csv(clean_file, index=False))
    done += 1

# token_count = pd.DataFrame(vocab.most_common(), columns=['token', 'n'])
# token_count.to_parquet('items_token_count.parquet')
# corpus_stats = pd.DataFrame(stats, columns=['file_id', 'item', 'sentence_id', 'tokens', 'clean_tokens'])
# with pd.HDFStore('items_vocab.h5') as store:
#     store.put('corpus_stats', corpus_stats)
