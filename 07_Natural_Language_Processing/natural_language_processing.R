# Title     : TODO
# Objective : TODO
# Created by: abautista
# Created on: 6/17/2018


# ------------------------------------------ Importing the dataset ----------------------------------------------#
dataset_original = read.delim("C:\\Users\\abautista\\PycharmProjects\\Machine_Learning_000\\csv_files\\Restaurant_Reviews.tsv",
          quote = '', stringsAsFactors = FALSE)

# ------------------------------------------ cleaning the datasets ----------------------------------------------#
# create each column for each word and 1000 rows that reperesent all comments
install.packages('tm')
library(tm)

corpus = VCorpus(VectorSource(dataset_original$Review))
corpus = tm_map(corpus, content_transformer(tolower))
corpus = tm_map(corpus, removeNumbers)
corpus = tm_map(corpus, removePunctuation)
install.packages('SnowballC')
library(SnowballC)
corpus = tm_map(corpus, removeWords, stopwords())
corpus = tm_map(corpus, stemDocument) # remove the tense of verbs and convert adjectives to nouns
corpus = tm_map(corpus, stripWhitespace) # remove the extra space that was left when we removed the numbers
as.character(corpus[[1]])


# ------------------------------------------ Create the bag of words ----------------------------------------------#

dtm = DocumentTermMatrix(corpus)  #sparse matrix
dtm = removeSparseTerms(dtm, 0.999) # keep the 99% of words that are the most frequent

# --------------------------------------- Apply the classification model --------------------------------------------#
# apply any classification model but in natural language processing the most common models are naive bayes, decision tree or random forest

dataset = as.data.frame(as.matrix(dtm)) # convert the matrix to a data frame
View(dataset)
dataset$Liked = dataset_original$Liked

# encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0,1))

library(caTools) # necessary library for doing the split
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# ---------------- Fit classifier to the training set ------------------- #

install.packages('randomForest')
library(randomForest)
classifier = randomForest(
             x = training_set[-692],
             y = training_set$Liked,
             ntree = 50 )

# ----------------------- Predicting the Test set results ----------------------- #

y_pred = predict(classifier, newdata = test_set[-692]) #remove the last column which is the dependent variable

# ----------------------- Making the confusion matrix -------------------------- #

cm = table(test_set[, 692], y_pred)


