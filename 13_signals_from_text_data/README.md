# Chapter 12: Text Processing

## Content

### NLP Applications - Examples

- **Chatbots:** Understand natural language from the user and return intelligent responses.
    - [Api.ai](https://api.ai/)
- **Information retrieval:** Find relevant results and similar results.
    - [Google](https://www.google.com/)
- **Information extraction:** Structured information from unstructured documents.
    - [Events from Gmail](https://support.google.com/calendar/answer/6084018?hl=en)
- **Machine translation:** One language to another.
    - [Google Translate](https://translate.google.com/)
- **Text simplification:** Preserve the meaning of text, but simplify the grammar and vocabulary.
    - [Rewordify](https://rewordify.com/)
    - [Simple English Wikipedia](https://simple.wikipedia.org/wiki/Main_Page)
- **Predictive text input:** Faster or easier typing.
    - [Phrase completion application](https://justmarkham.shinyapps.io/textprediction/)
    - [A much better application](https://farsite.shinyapps.io/swiftkey-cap/)
- **Sentiment analysis:** Attitude of speaker.
    - [Hater News](https://medium.com/@KevinMcAlear/building-hater-news-62062c58325c)
- **Automatic summarization:** Extractive or abstractive summarization.
    - [autotldr](https://www.reddit.com/r/technology/comments/35brc8/21_million_people_still_use_aol_dialup/cr2zzj0)
- **Natural language generation:** Generate text from data.
    - [How a computer describes a sports match](http://www.bbc.com/news/technology-34204052)
    - [Publishers withdraw more than 120 gibberish papers](http://www.nature.com/news/publishers-withdraw-more-than-120-gibberish-papers-1.14763)
- **Speech recognition and generation:** Speech-to-text, text-to-speech.
    - [Google's Web Speech API demo](https://www.google.com/intl/en/chrome/demos/speech.html)
    - [Vocalware Text-to-Speech demo](https://www.vocalware.com/index/demo)
- **Question answering:** Determine the intent of the question, match query with knowledge base, evaluate hypotheses.
    - [How did supercomputer Watson beat Jeopardy champion Ken Jennings?](http://blog.ted.com/how-did-supercomputer-watson-beat-jeopardy-champion-ken-jennings-experts-discuss/)
    - [IBM's Watson Trivia Challenge](http://www.nytimes.com/interactive/2010/06/16/magazine/watson-trivia-game.html)
    - [The AI Behind Watson](http://www.aaai.org/Magazine/Watson/watson.php)

## Technical requirements

### spaCy

- [spaCy.io](https://spacy.io/)
- [Installation instructions](https://spacy.io/usage/#installation)

### textacy


textacy is a Python library that performs a variety of natural language processing (NLP) tasks, built on the high-performance spaCy library. With the fundamentals — tokenization, part-of-speech tagging, dependency parsing, etc. — delegated to another library, textacy focuses on the tasks that come before and follow after.


- [Documentation](https://chartbeat-labs.github.io/textacy/index.html)


### TextBlob


The `TextBlob` library provides a simplified interface for common NLP tasks including part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and others.

**To install textblob run:**

> `conda install -c https://conda.anaconda.org/sloria textblob`

**Or:**

> `pip install textblob`

> `python -m textblob.download_corpora`

- [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)

- [TextBlob Sentiment Analysis](https://github.com/sloria/TextBlob/blob/dev/textblob/en/en-sentiment.xml)


### Natural Language Toolkit (NLTK)

NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries, and an active discussion forum.

- [NLTK Documentation](http://www.nltk.org/)

## References

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf), Daniel Jurafsky & James H. Martin, 3rd edition, draft, 2018

- [Statistical natural language processing and corpus-based computational linguistics](https://nlp.stanford.edu/links/statnlp.html), Annotated list of resources, Stanford University
- [Daily Market News Sentiment and Stock Prices](https://www.econstor.eu/handle/10419/125094), David E. Allen & Michael McAleer & Abhay K. Singh, 2015, Tinbergen Institute Discussion Paper

- [Predicting Economic Indicators from Web Text Using Sentiment Composition](http://www.ijcce.org/index.php?m=content&c=index&a=show&catid=39&id=358), Abby Levenberg, et al, 2014


- [JP Morgan NLP research results](https://www.jpmorgan.com/global/research/machine-learning)

### Document-Term Matrix

- [TF-IDF is about what matters](https://planspace.org/20150524-tfidf_is_about_what_matters/)

## Data

- [BBC Articles](http://mlg.ucd.ie/datasets/bbc.html), use raw text files
- [TED2013](http://opus.nlpl.eu/TED2013.php), a parallel corpus of TED talk subtitles in 15 langugages
- [Cheng-Caverlee-Lee September 2009 - January 2010 Twitter Scrape](https://archive.org/details/twitter_cikm_2010)
- [Yelp Dataset Challenge](https://www.yelp.com/dataset/challenge)
- [The New York Times Annotated Corpus](https://catalog.ldc.upenn.edu/LDC2008T19)