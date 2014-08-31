# Junk SMS classification using Naive Bayes.

library(tm)

# Read the data in a dataframe
sms_raw <- read.csv('data/sms_spam.csv', stringsAsFactors=FALSE)

# Convert the label to a factor(spam/ham)
sms_raw$type <- factor(sms_raw$type)

# Construct a tm::Corpus from our data's $text. Since sms_raw$text is just
# a vector, we need to wrap it in a VectorSource before passing it to the
# Corpus ctor.
sms_corpus <- Corpus(VectorSource(sms_raw$text))

# Clean up the corpus with basic text transformations.
corpus_clean <- sms_corpus

# This was a pain in the ass to get right, as the book simply maps tolower
# without using a content_transformer.
corpus_clean <- tm_map(corpus_clean, content_transformer(tolower))
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)


# Now that the data is cleaned up, we need to tokenize the sentences into
# words since we are using a bag of words model. Tokenization is provided
# by a so called DocumentTermMatrix.

# A DocumentTermMatrix is a sparse matrix with the rows representing documents
# and the columns representing terms. Each cell holds the number of times
# the term represented by the cell's column appears in the document represented
# by the cell's row.

sms_dtm <- DocumentTermMatrix(corpus_clean)

# Split the data in to train and test sets.

sms_raw_train <- sms_raw[1:4169,]
sms_raw_test <- sms_raw[4170:5559,]

sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5559,]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test <- corpus_clean[4170:5559]


# To visualize the data we have, one method is to use a wordcloud, which
# plots more frequent words in bigger fonts than less frequent ones.

library(wordcloud)

# We will visualize the training set, which contains roughly 4000 documents.
# We will also want to supress words appearing less than 40(10% of 4000) times
# to avoid clutter.


# wordcloud(sms_corpus_train, min.freq=40, random.order=FALSE)

# Another cool visualization would be the separate word clouds for spam and
# ham.
# When given raw text (and not a Corpus), wordcloud() will automatically
# apply text X-formations. 

spam <- subset(sms_raw_train, type=='spam')
ham <- subset(sms_raw_train, type=='ham')

# wordcloud(spam$text, max.words=40)
# wordcloud(ham$text, max.words=40)

# At this point, we have 7877 unique words, that appear in at least one
# sms. But this is a huge number of features and anyway, words appearing once
# or twice are unlikely to help the classifier learn much about the spam/ham
# distribution.
# 
# We will remove all the words that appear in less than about 0.1% of the
# number of messages in the corpus on average. In our case, we shall remove
# all the words that appear in less than 5 messages. The tm::findFreqTerms()
# function will take a DocumentTermMatrix/TermDocumentMatrix and return
# a vector of terms that have the frequency in the range [lowfreq, highfreq]
# where both lowfreq and highfreq are named arguments to findFreqTerms().
# 
# The Dictionary() function, that the book uses has been deprecated and a
# vector is supposed to be used when creating a DocumentTermMatrix.

sms_dict <- findFreqTerms(sms_dtm_train, lowfreq=5)
sms_train <- DocumentTermMatrix(sms_corpus_train, list(dictionary=sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test, list(dictionary=sms_dict))

# Now we have reduced the number of features to a little above 1200
# corresponding only to the words that appear at least five times in the
# corpus. But there is one problem - The DocumentTermMatrix cells contain a
# count of words - the number of times the word in the column appears in the
# document in the row. This should be changed to whether the word appears at
# all in the document corresponding to the row. This is because the naive
# Bayes classifier algorithm works with categorical variables.

convert_counts <- function(x) {
    x <- ifelse(x > 0, 1, 0)
    x <- factor(x, levels=c(0, 1), labels=c('No', 'Yes'))
    return(x)
}

sms_train <- apply(sms_train, MARGIN=2, convert_counts)
sms_test <- apply(sms_test, MARGIN=2, convert_counts)

# This contains many statistical methods, including one for naive Bayes
# classification.
library(e1071)

# Construct the naive Bayes classifier.
sms_classifier <- naiveBayes(sms_train, sms_raw_train$type)

# Predict on the test data.
sms_test_pred <- predict(sms_classifier, sms_test)

library(gmodels)
CrossTable(sms_raw_test$type, sms_test_pred, prop.chisq=FALSE, prop.t=FALSE,
           dnn=c('actual', 'predicted'))


# Check how many of these selected elements equal 1.
true_pos <- function(actual, predicted) sum(predicted[actual == 1] == 1)

# Select those elements from predicted which correspond to actual == 1
# Check how many of these selected elements equal 0.
false_neg <- function(actual, predicted) sum(predicted[actual == 1] == 0)

# Select those elements from predicted which correspond to actual == 0
# Check how many of these selected elements equal 1.
false_pos <- function(actual, predicted) sum(predicted[actual == 0] == 1)



# Recall is the fraction of Malignant cases that we successfully classified
# as such.
# recall = TP / (TP + FN)
recall <- function(actual, predicted) {
    tp <- true_pos(actual, predicted)
    fn <- false_neg(actual, predicted)
    return (tp / (tp + fn))
}


precision <- function(actual, predicted) {
    tp <- true_pos(actual, predicted)
    fp <- false_pos(actual, predicted)
    return (tp / (tp + fp))
}

f_score <- function(actual, predicted) {
    p <- precision(actual, predicted)
    r <- recall(actual, predicted)
    return (2 * p * r  / (p + r))
}

f_score(sms_raw_test$type == 'spam',
        sms_test_pred == 'spam') # 0.89

# Our recall is relatively low(~0.82), which means we potentially have
# false negatives - classifying actually spam messages as ham.
# But our recall is high(~0.96), which means that 96% of the time, when we
# say a message is spam, it is indeed spam. We might attempt to improve the
# recall by using a nonzero Laplace estimator. This number is added to all
# the word frequencies in all documents, eliminating zero frequency words,
# which will completely dominate the result of classification, since all
# the probabilities in the Bayes' Rule are multiplied.


sms_classifier2 <- naiveBayes(sms_train, sms_raw_train$type, laplace=1)

sms_test_pred2 <- predict(sms_classifier2, sms_test)

CrossTable(sms_raw_test$type, sms_test_pred2, prop.chisq=FALSE, prop.t=FALSE,
           dnn=c('actual', 'predicted'))

# This improves the recall by 1% and the precision by 2%
