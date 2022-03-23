# Technical-Test
#### _Marina Lacambra's Technical Test_

This package is aimed to
- Clean a twitter dataset using https://pypi.org/project/tweet-preprocessor/ or similar tools.
- Train a sentiment analysis model using Scikit-learn.

## Functions
### To clean a twitter dataset
- _tweet_cleaner_: takes a tweet and removes its hashtags and mentions using tweet-preprocessor
- _tweet_tokenizer_: takes a tweet as a string and returns a list of its tokens.
- _tweet_stopword_remover_: takes a tokenized tweet and returns a list of its tokens without stopwords.
- _tweet_punct_remover_: takes a tokenized tweet and returns a list of its tokens without punctuation.
- _lemmatizer_: takes a tokenized tweet and returns a list of its lemmas.
- _joiner_: takes a tokenized tweet and returns it as a string.
- _tweet_dataset_cleaner_tocsv_: reads a csv with a column "tweet", and writes a new csv with all the previous functions applied to it.
- _tweet_dataset_cleaner_: reads a csv with a column "tweet", and returns it with all the previous functions applied to it.

### To train a sentiment classifier
- _tweet_dataset_cleaner_: reads a csv with a column "tweet" and a column "label", and returns the metrics from the training. It uses: CountVectorization and Tf-IDF transformation for word vectorization; and a Naive-Bayes classifier for the sentiment classification.


### Bonus
There is an attempt to do the bonus exercise, but it does not work.


## Installation
```sh
pip install -i https://test.pypi.org/simple/ TechnicalTest==0.3.3 --extra-index-url https://pypi.org/simple/
```
## Usage
```sh
from TechnicalTest import technical_test as tt
```






Based on: https://towardsdatascience.com/twitter-sentiment-analysis-classification-using-nltk-python-fa912578614c