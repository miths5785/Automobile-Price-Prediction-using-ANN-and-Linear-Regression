library(tidyverse)
library(readxl)
library(dplyr)
library(performance)
library(ggplot2)
library(GGally)
library(nnet)
library(NeuralNetTools)
library(Metrics)
library(MASS)


#Q1)After your EDA, what factors do you think influence a customer’s decision to
#buy a care? What are the objectives of the model that Farid plans to build?

cars <- read_excel('/Users/abhinavram/Downloads/Cars.xlsx',
                   sheet = 'page-1_table-1')

head(cars)
dim(cars) #31 rows and 28 features.
sum(is.na(cars)) #0 which mean no null values are present.
str(cars)

cars$Mfr_G <- ifelse(cars$Mfr_G == "1.0", "1", cars$Mfr_G)

cols <- c("Fuel", "MC", "Auto", "Cyl", "Drs", "Grs", "ABS", "Abag_1", "Abag_2", 
          "AC", "Comp", "CD", "Clock", "Pwi", "PStr", "Radio", "SpM", "M Rim", 
          "Tow_Bar")
cars[cols] <- lapply(cars[cols], factor)
str(cars) 
#Categorical variables converted from numerical and character to factors

cars$Price <- gsub(",", "", cars$Price) 
# Applying gsub function on Price variable as some of the input values were not 
#properly formatted and were showing a warning message.

num_cols <- c("Price", "Age", "KM", "HP", "CC", "Wght", "G P")
cars[num_cols] <- apply(cars[num_cols], 2,  as.numeric)
str(cars)

cars$Silver <- as.factor(ifelse(cars$Colour == "Silver", 1, 0))
cars$Red <- as.factor(ifelse(cars$Colour == "Red", 1, 0))
cars$Black <- as.factor(ifelse(cars$Colour == "Black", 1, 0))
cars$Grey <- as.factor(ifelse(cars$Colour == "Grey", 1, 0))
cars$Blue <- as.factor(ifelse(cars$Colour == "Blue", 1, 0))
cars$Green <- as.factor(ifelse(cars$Colour == "Green", 1, 0))
cars$Mfr_G <- ifelse(cars$Mfr_G == "1.0", "1", cars$Mfr_G)
cars$Mfr_G <- as.factor(cars$Mfr_G)
#There is no need to check for outliers in the dataset as each car may have 
#separate and unique features.

ggcorr(cars, label = T)
#Hence as we can see that the Price has a strong positive correlation with HP, 
#CC and Weight. However, price is negatively correlated with KM.
#Also we can see strong positive correlation between HP and CC, HP and weight, 
#CC and Weight.

cars <- subset(cars, select = -c(Fuel, Drs, Cyl, ABS, Abag_1, PStr, Colour))

hist(cars$Price, col="darkblue")
#As we can see, the majority of the car prices are in the range of 14000 to 
#18000

#The data has been completely cleaned and EDA has been performed.

head(cars)

#Forward Variable Selection 
null <- lm(Price~1, data=cars)
full <- lm(Price~., data=cars)

variable_sel <- step(null, scope=list(lower=null, upper=full), 
                     direction="forward")
summary(variable_sel)

#Therefore, we can conclude that factors such as HP, Clock, Red, Silver, SpM, 
#MRim, Wght, Auto strongly influence a customers decision in buying a car.
#Following are the objectives of Farid's Model:
#1. He plans to build a robust model to accurately predict the prices of cars 
#using the car features list.
#2. He plans to use and compare various ML models such as Linear Regression and 
#compare the same with Neural-Network.
#3. He plans to identify important features that play a key role in predicting 
#the prices of cars and deploy and use a single model after proper comparison 
#for computing the prices of cars.
#4. This will help Adam Ebraham to provide the optimal MSRPs to the dealers, so 
#that the dealers could decide their final price.


#Q2) Construct a neural network model. Validate and interpret the model using a 
#different number of hidden neurons.

#Normalizing the dataset
scale <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}
cars_normalize <- cars %>% mutate_if(is.numeric, scale)
cars_normalize
str(cars_normalize)

#Splitting the data into training and testing.
set.seed(1234)
ind <- sample(2, nrow(cars_normalize), replace = T, prob = c(0.6, 0.4))
train <- cars_normalize[ind == 1, ]
test <- cars_normalize[ind == 2, ]

#Building the neural network model.
dec <- seq(0.0001,1,length.out=20)
i <- 1
table_col_names <- c("Decay", "Size", "Error_Percentage")
table_matrix <- data.frame(matrix(nrow=1, ncol = length(table_col_names)))
colnames(table_matrix) <- table_col_names
table_matrix <- na.omit(table_matrix)
for (j in dec)
{
  x <- 1
  while (x <= 20) 
  {
    nn_cars_model <- nnet(Price ~ HP + Clock + Red + Silver + SpM + `M Rim` + 
                            Wght + Auto, data = train, linout = TRUE, 
                          size = x, decay = j, maxit = 1000, trace = FALSE)
    nn_cars_pred <- predict(nn_cars_model, test)
    error <- rmse(test$Price, nn_cars_pred)
    error_percent <- error/mean(test$Price)
    table_matrix[nrow(table_matrix) + 1,] <- c(j, x, error_percent)
    x <- x + 1
    i <- i + 1
    
  }
}

#In the above code, we have created a table called "table_matrix" of different 
#neural network models with different decay and size parameters along with their
#respective Error%.
#From the "table_matrix", we would be selecting that model, that has the lowest 
#Error% as that would be giving us the best performance.

head(table_matrix)

minimum_err <- min(table_matrix$Error_Percentage)
best_nn_index <- which(table_matrix$Error_Percentage == minimum_err)
best_nn_model <- table_matrix[best_nn_index,]
print(best_nn_model)

#Thus, from the above code, we got that when we set the decay to 0.05272 with a 
#size of 20, we get the best neural network model having Error% = 17.08%

#Q3)Compare your neural network models with linear regression model. Which one 
#is better?
#Building the Linear Regression Model
#Dividing the data into training and testing 
set.seed(123)
ind_reg <- sample(2, nrow(cars), replace = T, prob = c(0.6, 0.4))
train_reg <- cars[ind_reg == 1, ]
test_reg <- cars[ind_reg == 2, ]

linear_model <- lm(formula = Price ~ HP + Clock + Red + Silver + SpM + `M Rim` + 
                     Wght + Auto, data = train_reg)
linear_model_pred <- predict(linear_model, test_reg)
error_reg <- rmse(test_reg$Price, linear_model_pred)
error_percent_reg <- error_reg/mean(test_reg$Price)
print(error_percent_reg)

#Therefore, the Error% for the Linear Regression Model is 3.53% 
#So on comparing the best Neural Network Model along with the Linear Regression 
#Model, we find that the Linear Regression Model is way better than the Neural 
#Network Model as the Error% in case of Linear Regression Model is significantly
#lower than the Error% in the best Neural Network Model.

#Q4)Make a decision and offer your recommendations.
#Therefore, as concluded previously, the Linear model is better than the Neural 
#Network Model because of
#lower Error%. Therefore, we got our important variable as HP, Clock, Red, 
#Silver, SpM, MRim, Wght, Auto.
#On training the Linear Regression Model using these variables, we got the 
#following co-efficients.

summary(linear_model)

#Intercept: -11378.09
#HP: 55.03
#Clock1: 895.67
#Red1: -2215.53
#Silver1: -2029.97
#SpM1: -1164.44
#MRim1: -739.13
#Wght: 19.78
#Auto1: -795.79

#Therefore, our Linear Regression Model is:
#Price = -11378.09 + 55.03HP + 895.67Clock1 - 2215.53Red1 - 2029.97Silver1 - 
#1164.44SpM1 - 739.13MRim1 + 19.78Wght - 795.79Auto1

#Training the Neural Network Model by taking all the variables as predictors.
set.seed(1234)
ind <- sample(2, nrow(cars_normalize), replace = T, prob = c(0.6, 0.4))
train <- cars_normalize[ind == 1, ]
test <- cars_normalize[ind == 2, ]
dec <- seq(0.0001,1,length.out=20)
i <- 1
table_col_names <- c("Decay", "Size", "Error_Percentage")
table_matrix <- data.frame(matrix(nrow=1, ncol = length(table_col_names)))
colnames(table_matrix) <- table_col_names
table_matrix <- na.omit(table_matrix)
for (j in dec)
{
  x <- 1
  while (x <= 20) 
  {
    nn_cars_model <- nnet(Price ~ ., data = train, linout = TRUE, 
                          size = x, decay = j, maxit = 1000, trace = FALSE)
    nn_cars_pred <- predict(nn_cars_model, test)
    error <- rmse(test$Price, nn_cars_pred)
    error_percent <- error/mean(test$Price)
    table_matrix[nrow(table_matrix) + 1,] <- c(j, x, error_percent)
    x <- x + 1
    i <- i + 1
    
  }
}

#In the above code, we have created a table called "table_matrix" of different 
#neural network models with different decay and size parameters along with their
#respective Error%.
#From the "table_matrix", we would be selecting that model, that has the lowest 
#Error% as that would be giving us the best performance.

head(table_matrix)

minimum_err <- min(table_matrix$Error_Percentage)
best_nn_index <- which(table_matrix$Error_Percentage == minimum_err)
best_nn_model <- table_matrix[best_nn_index,]
print(best_nn_model)

#Essentially, as we can see that training the Neural Network Model again using 
#all variables as predictors, we get the least Error%
# of 18.67% for the same Decay and Size parameter as the previous one. However, 
#we would recommend using the previous Neural Network Model
#as compared to this one because the previous one had lesser Error%.

