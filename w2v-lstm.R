
# load packages required
library(keras)
library(RVerbalExpressions)
library(magrittr)
library(textclean)
library(tidyverse)
library(tidytext)
library(rsample)
library(yardstick)
library(caret)

install.packages("keras", type = "source")
library(keras)
install_keras()

install.packages("dplyr")
library(dplyr)
is.vector(c(data))

#set seed keras for reproducible result
set.seed(420)
reticulate::py_set_seed(420)
tf$random$set_seed(420)

# set conda env
use_condaenv("tensorflow")

data <- read_csv("Final Training Data Set - 1.csv")
# quick check
glimpse(data)

#preprocessing
mention <- rx() %>% 
  rx_find(value = "@") %>% 
  rx_alnum() %>% 
  rx_one_or_more()
mention

hashtag <- rx() %>% 
  rx_find(value = "#") %>% 
  rx_alnum() %>% 
  rx_one_or_more()
hashtag

question <- rx() %>% 
  rx_find(value = "?") %>% 
  rx_one_or_more()
question

exclamation <- rx() %>% 
  rx_find(value = "!") %>% 
  rx_one_or_more()
exclamation

punctuation <- rx_punctuation()
punctuation

number <- rx_digit()
number

dollar <- rx() %>% 
  rx_find("$")
dollar

#data cleaning
data <- data%>% 
  mutate(
    text_clean = text %>% 
      replace_url() %>% 
      replace_emoji() %>% 
      replace_emoticon() %>% 
      replace_html() %>% 
      str_remove_all(pattern = mention) %>% 
      str_remove_all(pattern = hashtag) %>% 
      replace_contraction() %>% 
      replace_word_elongation() %>% 
      str_replace_all(pattern = question, replacement = "questionmark") %>% 
      str_replace_all(pattern = exclamation, replacement = "exclamationmark") %>% 
      str_remove_all(pattern = punctuation) %>% 
      str_remove_all(pattern = number) %>% 
      str_remove_all(pattern = dollar) %>% 
      str_to_lower() %>% 
      str_squish()
  )

data <- data %>% 
mutate(label = factor(tweet, levels = c("hate","non-hate")),
       label = as.numeric(label),
       label = label - 1) %>% 
  select(text_clean, label) %>% 
  na.omit()
head(data, 10)

# prepare tokenizers
num_words <- 1024 
tokenizer <- text_tokenizer(num_words = num_words,lower = TRUE) %>% 
fit_text_tokenizer(data$text_clean)

# data-splitting
set.seed(100)
intrain <- initial_split(data = data, prop = 0.8, strata = "label")
data_train <- training(intrain)
data_test <- testing(intrain)
set.seed(100)
inval <- initial_split(data = data_test, prop = 0.5, strata = "label")
data_val <- training(inval)
data_test <- testing(inval)

# prepare x
data_train_x <- texts_to_sequences(tokenizer, data_train$text_clean) %>%
  pad_sequences(maxlen = maxlen)
data_val_x <- texts_to_sequences(tokenizer, data_val$text_clean) %>%
  pad_sequences(maxlen = maxlen)
data_test_x <- texts_to_sequences(tokenizer, data_test$text_clean) %>%
  pad_sequences(maxlen = maxlen)

# prepare y
data_train_y <- to_categorical(data_train$label, num_classes = 3)
data_val_y <- to_categorical(data_val$label, num_classes = 3)
data_test_y <- to_categorical(data_test$label, num_classes = 3)


# Modeling ------------
# initiate keras model sequence
model <- keras_model_sequential()

# LSTM model
model %>%
  # layer input
  layer_embedding(
    name = "input",
    input_dim = num_words,
    input_length = maxlen,
    output_dim = 32, 
    embeddings_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 2)
  ) %>%
  # layer dropout
  layer_dropout(
    name = "embedding_dropout",
    rate = 0.5
  ) %>%
  # layer lstm 1
  layer_lstm(
    name = "lstm",
    units = 256,
    dropout = 0.2,
    recurrent_dropout = 0.2,
    return_sequences = FALSE, 
    recurrent_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 2),
    kernel_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 2)
  ) %>%
  # layer output
  layer_dense(
    name = "output",
    units = 3,
    activation = "softmax", 
    kernel_initializer = initializer_random_uniform(minval = -0.05, maxval = 0.05, seed = 2))

# compile the model
model %>% compile(
  optimizer = "adam",
  metrics = "accuracy",
  loss = "categorical_crossentropy" )

# model summary
summary(model)

# model fit settings
epochs <- 10
batch_size <- 512

# fit the model
history <- model %>% fit(
  data_train_x, data_train_y,
  batch_size = batch_size, 
  epochs = epochs,
  verbose = 1,
  validation_data = list(
    data_val_x, data_val_y
  ))

# history plot
plot(history)

# Model Evaluation ----------------

# predict on train
data_train_pred <- model %>%
  predict_classes(data_train_x) %>%
  as.vector()

# predict on val
data_val_pred <- model %>%
  predict_classes(data_val_x) %>%
  as.vector()

# predict on test
data_test_pred <- model %>%
  predict_classes(data_test_x) %>%
  as.vector()

# accuracy on data train
accuracy_vec(truth = factor(data_train$label,labels = c("hate","non-hate")),
  estimate = factor(data_train_pred, labels = c("hate","non-hate")))

# accuracy on data test
accuracy_vec(truth = factor(data_test$label,labels = c("hate","non-hate")),
  estimate = factor(data_test_pred, labels = c("hate","non-hate")))
