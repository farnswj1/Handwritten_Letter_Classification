# Justin Farnsworth
# Handwritten Letter Classification
# September 1, 2020


# Loading The Data
####################################################################################################

# Required packages
if (!require(tidyverse)) { install.packages("tidyverse"); library(tidyverse) }
if (!require(caret)) { install.packages("caret"); library(caret) }
if (!require(matrixStats)) { install.packages("matrixStats"); library(matrixStats) }
if (!require(keras)) { install.packages("keras"); library(keras) }
if (!require(tensorflow)) { install.packages("keras"); library(keras) }
if (!require(reticulate)) { install.packages("reticulate"); library(reticulate) }

# Connect to the Anaconda environment
use_condaenv("Keras_TensorFlow", required = TRUE)

# Load the data
data <- as.matrix(
  read_csv(
    unz("A-Z_Handwritten_Data.zip", "A-Z_Handwritten_Data.csv"), 
    col_names = FALSE
  )
)
colnames(data) <- NULL

# Compute the number of rows
nrow(data)

# Compute the number of columns
ncol(data)

# Count the frequency of each letter
tibble(Number = data[,1], Letter = sapply(as.raw(data[,1] + 65), rawToChar)) %>% 
  group_by(Number, Letter) %>% 
  summarize(Total = n()) %>% 
  ungroup() %>% 
  print(n = Inf)


# Pre-Processing The Data
####################################################################################################

# Separate the features (x) from the labels (y)
# Binarize x by converting small numbers to 0 and large numbers to 1
x <- (data[,2:785] >= 255/2) * 1
y <- data[,1]

# Save the column SDs for faster computation
SDs <- colSds(x)

# Plot the frequency of SDs
qplot(SDs, bins = 256, color = I("black"))

# Plot the pixels and their variabilities
image(1:28, 1:28, matrix(SDs, 28, 28))

# Keep columns with higher variability
x <- x[,SDs >= 0.05]

# Count the number of columns remaining
ncol(x)

# Show the variabilities of the remaining columns
image(1:20, 1:20, matrix(colSds(x), 20, 20))

# Show several images after pre-processing
image(1:20, 1:20, matrix(x[1, 400:1], 20, 20))
image(1:20, 1:20, matrix(x[50000, 400:1], 20, 20))
image(1:20, 1:20, matrix(x[100000, 400:1], 20, 20))
image(1:20, 1:20, matrix(x[200000, 400:1], 20, 20))
image(1:20, 1:20, matrix(x[300000, 400:1], 20, 20))
image(1:20, 1:20, matrix(x[350000, 400:1], 20, 20))


# Training & Test Sets
####################################################################################################

# Split the data into a training set (80%) and a test set (20%)
set.seed(2)
test_index <- createDataPartition(y, p = 0.2, list = FALSE)

train_x <- x[-test_index,]
train_y <- y[-test_index] %>% to_categorical(26)

test_x <- x[test_index,]
test_y <- y[test_index] %>% to_categorical(26)

# Count the number of rows in the training set
nrow(train_x)

# Count the number of rows in the training set
nrow(test_x)


# Sequential Model
####################################################################################################

# Generate the model
model <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(ncol(x))) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 26, activation = "softmax")

# View the model
summary(model)

# Compile the model
model %>% compile(
  optimizer = optimizer_rmsprop(), 
  loss = "categorical_crossentropy", 
  metrics = c("accuracy")
)

# Fit the model
fit <- model %>% fit(
  train_x, 
  train_y, 
  epochs = 15, 
  batch_size = 256
)

# Plot the results from fitting the model
plot(fit)

# Evaluate the accuracy using the test cases
model %>% evaluate(test_x, test_y)

# Predict the results
predictions <- model %>% predict_classes(test_x)


# Results
####################################################################################################

# Combine the observations and the predictions into one table
results <- tibble(Prediction = predictions, Observation = y[test_index])

# Compute the accuracy
mean(results$Prediction == results$Observation)

# Show the recalls by letter
results %>% 
  group_by(Prediction) %>% 
  summarize(
    Letter = rawToChar(as.raw(first(Prediction) + 65)), 
    Count = n(), 
    Recall = mean(Prediction == Observation)
  ) %>% 
  ungroup() %>% 
  print(n = Inf)

# Plot the recalls by letter
results %>% 
  group_by(Prediction) %>% 
  summarize(
    Letter = rawToChar(as.raw(first(Prediction) + 65)), 
    Count = n(), 
    Recall = mean(Prediction == Observation)
  ) %>% 
  ungroup() %>% 
  ggplot(aes(Letter, Recall)) + 
  geom_bar(aes(fill = Letter), show.legend = FALSE, stat = "identity") + 
  ggtitle("Model Recall By Letter") + 
  scale_y_continuous(breaks = seq(0, 1, 0.1), labels = seq(0, 1, 0.1))

# Show the precisions by letter
results %>% 
  group_by(Observation) %>% 
  summarize(
    Letter = rawToChar(as.raw(first(Observation) + 65)), 
    Count = n(), 
    Precision = mean(Prediction == Observation)
  ) %>% 
  ungroup() %>% 
  print(n = Inf)

# Plot the precisions by letter
results %>% 
  group_by(Observation) %>% 
  summarize(
    Letter = rawToChar(as.raw(first(Observation) + 65)), 
    Count = n(), 
    Precision = mean(Prediction == Observation)
  ) %>% 
  ungroup() %>% 
  ggplot(aes(Letter, Precision)) + 
  geom_bar(aes(fill = Letter), show.legend = FALSE, stat = "identity") + 
  ggtitle("Model Precision By Letter") + 
  scale_y_continuous(breaks = seq(0, 1, 0.1), labels = seq(0, 1, 0.1))
