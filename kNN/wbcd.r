library(class)
# Prediction of breast cancer diagnosis using the WBCD dataset.

wbcd <- read.csv('data/wisc_bc_data.csv', stringsAsFactors=FALSE)
 
# Remove the ID attribute
wbcd = wbcd[-1]

table(wbcd$diagnosis)

# Recode the diagnosis as a factor
wbcd$diagnosis <- factor(wbcd$diagnosis, levels=c('M', 'B'),
                         labels=c('Malignant', 'Benign'))


# The max-min normalization
normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
}

# We need to normalize the 30 numeric columns of the dataframe.
# lapply takes a list and a function, and returns a list of the function
# applied to each of the elements in the input list. Then we convert this
# output list to a dataframe.
wbcd_n <- as.data.frame(lapply(wbcd[-1], normalize))

# Split the data into 469 training examples and 100 test examples.
wbcd_train <- wbcd_n[1:469,]
wbcd_test <- wbcd_n[470:569,]

# Store the target labels in the same division and order.
wbcd_train_labels <- wbcd[1:469, 1]
wbcd_test_labels <- wbcd[470:569, 1]

# ---------------------------------------
# Use k = int(sqrt(469)) = 21
wbcd_test_pred <- knn(train=wbcd_train, test=wbcd_test, cl=wbcd_test_labels,
                      k=21)
# ---------------------------------------

# See the performance
CrossTable(x=wbcd_test_labels, y=wbcd_test_pred, prop.chisq=FALSE)

# Select those elements from predicted which correspond to actual == 1
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

# See how our model performed!
act <- wbcd_test_labels == 'Malignant'
pred <- wbcd_test_pred == 'Malignant'

f_score(act, pred) # A whopping 97% !!

# --- Improvement
# Since we see that the errors our model makes are dangerous false negatives(
# telling someone they don't have cancer when they in reality do),
# we need to improve.

# Use z-score standardization instead of max-min normalization.

wbcd_z <- as.data.frame(scale(wbcd[-1])) # Omit the diagnosis column.
wbcd_train_z <- wbcd_z[1:469,]
wbcd_test_z <- wbcd_z[470:569,]


# Predict using k = 21 now
wbcd_test_pred_z <- knn(train=wbcd_train_z, test=wbcd_test_z,
                        cl=wbcd_train_labels, k=21)

CrossTable(x=wbcd_test_labels, y=wbcd_test_pred_z, prop.chisq=FALSE)

# See how our model performed!
act <- wbcd_test_labels == 'Malignant'
pred <- wbcd_test_pred_z == 'Malignant'





