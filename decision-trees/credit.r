require('C50')
require('gmodels')

credit <- read.csv('data/credit.csv')

# Set the rand seed so multiple analyses return the same result
set.seed(12345)

# order(x) returns a vector of indices ordered such that they sort x in
# ascending order. e.g., order(c(10, 9, 8, 7, 6)) == c(5, 4, 3, 2, 1) 
# Doing this to a randomly generated x will get us a random index vector which
# in turn can be used to shuffle our dataset randomly.
credit_rand <- credit[order(runif(1000)),]

# Create a 90-10 train-test split
credit_train <- credit_rand[1:900,]
credit_test <- credit_rand[901:1000,]

# Ensure at this point whether each of the train and test sets contains a
# similar proportion of defaults ~ 30% default. This can be easily checked
# by inspecting the output of prop.table(table(x$default)) where 
# x <- {credit_train, credit_test}

# This is a matrix that weighs relative costs of mistakes. Since the factor
# no (in the context of $default) is 1 and yes == 2, We build a matrix that
# gives a weighting on various types of errors (nn, ny, yn, yy). Here the
# rows are predicted values and cols are actual values. Hence, we are saying
# that false-negatives are 4-times as costly as false-positives.
#
# This is because of the business rule that a defaulted loan causes more harm
# to the bank than a missed opportunity of getting interest.
error_cost <- matrix(c(0, 1, 4, 0), nrow=2)

# Create the decision tree model by excluding the label ($default)
credit_model <- C5.0(credit_train[-17], credit_train$default, costs=error_cost)

credit_pred <- predict(credit_model, credit_test[-17])

CrossTable(credit_test$default, credit_pred, prop.chisq=F)



