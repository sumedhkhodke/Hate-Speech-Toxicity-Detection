#Importing required Libraries
library(data.table)
library(tidytext)
library(ggplot2)
library(slam)
library(tm)
library(NLP)
library(caret)
library(e1071)
library(rtweet)
library(dplyr)
library(tm) 
library(SnowballC) 
library(tidyverse) 
library(wordcloud) 
library(ggplot2) 
library(cvms)

#Setting seed
set.seed(1234)

#Importing dataset
data = fread("finalfittraining.csv")
head(data)
datas=data%>%select(label,tweet) 
head(datas) 
round(prop.table(table(datas$label)),2)

#Creating a corpus of tweets
corpus = VCorpus(VectorSource(datas$tweet)) 

#Converting to lower case
corpus = tm_map(corpus, content_transformer(tolower)) 

#Removing numbers
corpus = tm_map(corpus, removeNumbers) 

#Removing punctutation
corpus = tm_map(corpus, removePunctuation) 

#Removing stopwords
corpus = tm_map(corpus, removeWords, stopwords("english")) 

#Stemming document
corpus = tm_map(corpus, stemDocument) 

#Stripping whitespace
corpus = tm_map(corpus, stripWhitespace) 
as.character(corpus[[1]])

#Creating a Term Document Matrix
dtm = DocumentTermMatrix(corpus) 
dtm 
dim(dtm) 

#Removing sparse terms
dtm = removeSparseTerms(dtm, 0.999) 
dim(dtm)

#Creating a word cloud of non-hate speech
nohate = subset(datas,label==0) 
wordcloud(nohate$tweet, max.words = 100, colors = "blue") 
#Creating a word cloud of hate speech
hate = subset(datas,label==1) 
wordcloud(hate$tweet, max.words = 100, colors = "purple")

#Converting labels to yes or no to convert it into binary
convert <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}  

#Converting the corpus to dataset
datanaive = apply(dtm, 2, convert)
dataset = as.data.frame(as.matrix(dtm))    
dataset$Class = datas$label
str(dataset$Class)

#Splitting the dataset into train and test set
split = sample(2,nrow(dataset),prob = c(0.70,0.30),replace = TRUE)
train_set = dataset[split == 1,]
test_set = dataset[split == 2,] 

prop.table(table(train_set$Class))
prop.table(table(test_set$Class))

#Developing a model using Naive Bayes algorithm
NBclassfier=naiveBayes(Class~tweet, data=train_set)
print(NBclassfier)
plot(NBclassfier)

printALL=function(model){
  trainPred=predict(model, newdata = train_set, type = "class")
  trainTable=table(train_set$Class, trainPred)
  testPred=predict(model, newdata=test_set, type="class")
  testTable=table(test_set$Class, testPred)
  print(trainTable)
  print(testTable)
  print(confusionMatrix(trainTable))
  print(confusionMatrix(testTable))
}
printALL(NBclassfier)

#Developing a model using Linear SVM
train_control <- trainControl(method="repeatedcv", number=5, repeats=1)
svm = svm(Class~tweet, data = train_set, kernel = "linear", cost = 10, scale = FALSE)
printALL(svm)
trainPredSvm=predict(svm, newdata = train_set, type = "raw")
trainTableSvm=table(train_set$Class, trainPredSvm)
testPredSvm=predict(svm, newdata=test_set, type="raw")
testTableSvm=table(test_set$Class, testPredSvm)
print(trainTableSvm)
print(testTableSvm)

confusionMatrix(predict(svm, newdata = train_set), train_set$Class)
confusionMatrix(predict(svm, newdata = test_set), test_set$Class)

#Developing a model using XGBoost
bst <- xgboost(data = sparse_matrix, label = output_vector, max.depth = 4,
               eta = 1, nthread = 2, nrounds = 10,objective = "binary:logistic")

importance <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = bst)
head(importance)
xgb.plot.importance(importance_matrix = importance)

#Developing a model using Logistic Regression
glm_fit_title <- glmnet(x_train , y_true[training_index], family = "binomial")
predicted_glm_title <- predict(glm_fit_title, x_test, type = "class")
trainTableGlm=table(train_set$Class, predicted_glm_title)
print(trainTableSvm)
accuracy_glm_title <- sum(y_true[-training_index] == predicted_glm_title)/ length(predicted_glm_title)


