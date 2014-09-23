require('C50')
require('gmodels')
require('RWeka')

mushrooms <- read.csv('data/mushrooms.csv', stringsAsFactors=TRUE)

# The veil_type feature has only one value "partial" across all samples. It
# does not provide any discriminatory advantage and hence we can drop it.
mushrooms$veil_type <- NULL


# We shall build and test our model on the same dataset as here the assumption
# is that the dataset covers all possible wild mushrooms. This means we aren't
# trying to generalize on unseen examples (our assumption is that there aren't
# any), but just trying to come up with rules to describe the current class
# distribution in terms of feature values.
# ---

# We shall first fit a simple OneR model (package RWeka)
mushroom_1R <- OneR(type ~ ., data=mushrooms)

# ^^ That gives ~98% accuracy with no poisonous mushrooms classified as
# edible. That is remarkable for a single feature based classifier (here,
# odor was picked).


# Now we use the RWeka implementation of the RIPPER algorithm to improve
# model performance even better.
mushroom_JRip <- JRip(type ~ ., data=mushrooms) # That gives us a 100% accuracy!


