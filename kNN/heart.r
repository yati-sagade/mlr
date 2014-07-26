# A shot at the heart disease dataset from the UCI ML dataset repo. The dataset
# is in data/heart-disease.csv

library(class)
heart <- read.csv('data/heart-disease.csv', stringsAsFactors=FALSE)
drops <- c('num')
# TODO: Age is an ordinal or a num?

# Sex is categorical. 1=M, 0=F

# --------------------------------------------------------
# Chest pain type - nominal with 3 levels, so dummy code.
ta <- heart$cp
ta[ta!=1] = 0

heart$typical_angina <- ta

aa <- heart$cp
aa[aa!=2] = 0
aa[aa==2] = 1

heart$atypical_angina <- aa

np <- heart$cp
np[np!=3] = 0
np[np==3] = 1

heart$non_anginal_pain <- np

# Mark cp for removal.
drops <- c(drops, 'cp')

# When all the above are zero, asymptomatic is implied.
# --------------------------------------------------------

# Rest BP value - num
# Serum cholesterol in mg/dl - num

# fbs tells whether the fasting blood sugar > 120 mg/dl - nominal
# 1 is true, 0 false

#------------------------------------------------------ 
# restecg is a nominal with values 0, 1, 2 representing
# normal, ST-T wave abnormality and left-ventricular hypertrophy
# , respectively. We will dummy code this variable.

lv_hyp <- heart$restecg
lv_hyp[lv_hyp!=2] <- 0
lv_hyp[lv_hyp==2] <- 1
heart$left_ventricular_hypertrophy <- lv_hyp

stt <- heart$restecg
stt[stt!=1] = 0
heart$stt_wave_abnormality <- stt

# Mark restecg for removal
drops <- c(drops, 'restecg')

# The zeroness of both of the above implies a normal restecg.
#------------------------------------------------------ 

# thalach is the max heart rate achieved - num

# exang is whether there was exercise induced angia found
# 1 is found, 0 is not found.

# oldpeak is again a num

#----------------------------------------------------------
# slope is the slope of the peark exercise ST segment.
# 1 - upsloping, 2 - flat, 3 - downsloping
upsl <- heart$slope
upsl[upsl!=1] = 0
heart$st_upsloping <- upsl

flat <- heart$slope
flat[flat!=2] = 0
flat[flat==2] = 1
heart$st_flat <- flat

# Mark slope for removal.
drops <- c(drops, 'slope')

# both upsloping and flat being zero implies downsloping.
# ----------------------------------------------------------

# ca is a num, representing the numbe of majore vessels(0-3)
# coloured by flourosopy

# thal -> 3 = normal, 6 = fixed defect, 7 = reversible defect
tn <- heart$thal
tn[tn!=3] = 0
tn[tn==3] = 1

heart$thal_normal <- tn

tfd <- heart$thal
tfd[tfd!=6] = 0
tfd[tfd==6] = 1

heart$thal_fixed_defect <- tfd

# Both of the above being zero => reversible defect.

drops <- c(drops, 'thal')

# Presence of disease should be 1 and absence should be zero. 
heart$num[heart$num > 0] = 1

heart_n <- heart[,!(names(heart) %in% drops)]

# Remove the target variable.


# NORMALIZE all the num rows

normalize <- function(x) {
    return ((x - mean(x)) / sd(x))
}

heart_n$age <- normalize(heart_n$age)
heart_n$trestbps <- normalize(heart_n$trestbps)
heart_n$chol <- normalize(heart_n$chol)
heart_n$thalach <- normalize(heart_n$thalach)
heart_n$oldpeak <- normalize(heart_n$oldpeak)
heart_n$ca <- normalize(heart_n$ca)

labels <- heart$num

heart_train <- heart_n[1:197,]
heart_val <- heart_n[198:198+50,]
heart_test <- heart_n[198+51:297,]

heart_val_test <- heart_n[198:297,]

heart_train_labels <- labels[1:197]
heart_val_labels <- labels[198:198+50]
heart_test_labels <- labels[198+51:297]

heart_val_test_labels <- labels[198:297]

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

accuracies <- NULL
print(length(heart_val_test_labels))
for (i in seq(1, 197)) {
    p <- knn(train=heart_train,
             test=heart_val_test,
             cl=heart_train_labels,
             k=i)
    accuracies <- c(accuracies,
                    sum((p == heart_val_test_labels) == TRUE) / 100.0)
}

plot(x=seq(1, 197), y=accuracies, col='red', xlab='k',
     ylab='accuracy', main='choosing the best k')



