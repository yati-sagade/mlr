require('tm')

# Read the TSV file in as a dataframe.
rot_df <- read.delim('data/rotten-tomatoes-train.tsv', stringsAsFactors=FALSE)

# Convert the labels to factors
rot_df$Sentiment <- factor(rot_df$Sentiment, levels=c(0, 1, 2, 3, 4),
                           labels=c('NEGATIVE', 'SOMEWHAT_NEGATIVE',
                                    'NEUTRAL',
                                    'SOMEWHAT_POSITIVE', 'POSITIVE'))

# Build a cleaned corpus.
rot_corpus <- Corpus(VectorSource(rot_df$Phrase))
corpus_clean <- rot_corpus
mklower <- content_transformer(tolower)
rmstopwords <- function(x) removeWords(x, stopwords())
ctr <- 0
corpus_clean <- tm_map(corpus_clean, function(x) {
                       print(ctr)
                       ctr <- ctr + 1
                       return(stripWhitespace(removePunctuation(rmstopwords(removeNumbers(mklower(x))))))
                })
# corpus_clean <- tm_map(corpus_clean, content_transformer(tolower))
# corpus_clean <- tm_map(corpus_clean, removeNumbers)
# corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
# corpus_clean <- tm_map(corpus_clean, removeWords, removePunctuation)
# corpus_clean <- tm_map(corpus_clean, removeWords, stripWhitespace)


# Build a DTM
rot_dtm <- DocumentTermMatrix(corpus_clean)




