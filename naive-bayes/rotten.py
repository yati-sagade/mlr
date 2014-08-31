'''
Movie review sentiment analysis. Data from Kaggle(which ultimately is from
the Stanford NLP lab).

'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score, recall_score, precision_score

def one_vs_all_metrics(y_true, y_pred, labels):
    '''
    Compute f1-scores for each label in ``labels`` using a one vs all
    method.

    '''
    num_labels = len(labels)
    f1_scores = np.zeros((num_labels,), dtype=float)
    precisions = np.zeros((num_labels,), dtype=float)
    recalls = np.zeros((num_labels,), dtype=float)
    for i, label in enumerate(labels):
        f1_scores[i] = f1_score(y_true==label, y_pred==label)
        precisions[i] = precision_score(y_true==label, y_pred==label)
        recalls[i] = recall_score(y_true==label, y_pred==label)
    return precisions, recalls, f1_scores


rot_df = pd.read_table('data/rotten-tomatoes-train.tsv')
train, test = train_test_split(rot_df)

train_df = pd.DataFrame(train, columns=rot_df.columns)
test_df = pd.DataFrame(test, columns=rot_df.columns)

# Use unigram, bigram and trigram information
vec = CountVectorizer(min_df=1, binary=True, ngram_range=(1, 3),
                      token_pattern=r'\b\w+\b')

# This matrix contains one row per document in our training set with binary
# indicator features which tell whether(1) or not(0) a word appears in the 
# document. The actual words can be seen using vec.get_feature_names()

X = vec.fit_transform(train_df.Phrase)
y = train_df.Sentiment.astype(int)

# Use the Bernoulli Naive Bayes classification algorithm which uses indicator
# features(unlike Multinomial NB which can use counts).
clf = BernoulliNB()
clf.fit(X, y)

X_test = vec.transform(test_df.Phrase)
y_test = test_df.Sentiment.astype(int)

y_pred = clf.predict(X_test)

p, r, f = one_vs_all_metrics(y_test, y_pred, [0, 1, 2, 3, 4])
ev = pd.DataFrame([p, r, f], columns=['negative', 's.negative', 'neutral', 's.positive', 'positive'])

# The rows are precision, recall and f1_score in that order.
print(ev) # <- Currently, we have a *horrible* performance with sub-25% recall
          # for the positive and negative classes.


