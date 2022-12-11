---
title: "PAC_competition"
author: "Cristian Leo"
date: "2022-11-20"
output:
  html_document:
    toc: true
    toc_float: true
header-includes:
  - \usepackage{sectsty}
  - \allsectionsfont{\color{cyan}}
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(eval = FALSE,warning = FALSE, message = FALSE)
```

## Load Data
```{r}
library(caret)
file='~/Desktop/Data_Science/Kaggle/PAC_competition/Datasets/lalasongs22/analysisData.csv'
df = read.csv(file)
```

# Data Exploration
As first step for this project I used the skim function from the skimr package to understand the distribution of the data and its completeness. By looking at the distribution of each variables, we can notice how tempo presents an abnormality. Indeed, some values are registered as 0, but the tempo definition in the dataset is "Overall estimated tempo of a track in beats per minute (BPM)." In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.", therefore it must be greater than 0.

Following I plotted the distribution of some continuous variables and their correlations between them and our dependent variable, rating. This data exploration can guide us in the choice of the predicting model to use. Indeed, too many variables present highly skewed data, and no linear relationship between the dependent variable and independent variables. Therefore, the data would require complex data transformation to be normalized and the predictive model would be easily overfitting the data. Lastly, because of the complex pattern of the data, I will use a decision tree based model to be able to override the collinearity presented between all the independent variables and the complex pattern of the data.
```{r}
library(e1071);library(ggplot2)
skimr::skim(df)
boxplot(df$tempo)
```

## Track Duration
Track duration is highly skewed, and correlation between track duration and rating is not linear.
```{r}
skewness(df$track_duration)

ggplot(df, aes(track_duration, rating)) +
  geom_point(alpha=0.5) +
  geom_smooth(method='lm', formula=y~x)
```

## Danceability
Danceability is normally distributed, and there is a linear looking relationship between danceability and rating.
```{r}
skewness(df$danceability)

ggplot(df, aes(danceability)) +
  geom_histogram(bins=20)

ggplot(df, aes(danceability, rating)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method='lm', formula=y~x)
```

## Energy
Energy is slightly negative skewed, but the relationship between energy and rating is linear looking.
```{r}
skewness(df$energy)

ggplot(df, aes(energy)) +
  geom_histogram(bins=20)

ggplot(df, aes(energy, rating)) +
  geom_point(alpha=0.5) +
  geom_smooth(method='lm', formula=y~x)
```

## Loudness
Loudness is negatively skewed, and there is no linear relationship between loudness and rating.
```{r}
skewness(df$loudness)

ggplot(df, aes(loudness)) +
  geom_histogram(bins=20)

ggplot(df, aes(loudness, rating)) +
  geom_point(alpha=0.5) +
  geom_smooth(method='lm', formula=y~x)
```

## Speechiness
Speechiness is higly positve skewed, and there is no linear relationship between speechiness and rating.
```{r}
skewness(df$speechiness)

ggplot(df, aes(speechiness)) +
  geom_histogram(bins=20)

ggplot(df, aes(speechiness, rating)) +
  geom_point(alpha=0.5) +
  geom_smooth(method='lm', formula=y~x)
```

## Acousticness
Acousticness is  positvely skewed, and there is no linear relationship between acousticness and rating.
```{r}
skewness(df$acousticness)

ggplot(df, aes(acousticness)) +
  geom_histogram(bins=20)

ggplot(df, aes(acousticness, rating)) +
  geom_point(alpha=0.5) +
  geom_smooth(method='lm', formula=y~x)
```

## Instrumentalness
Instrumentalness is higly positve skewed, and there is no linear relationship between instrumentalness and rating.
```{r}
skewness(df$instrumentalness)

ggplot(df, aes(instrumentalness)) +
  geom_histogram(bins=20)

ggplot(df, aes(instrumentalness, rating)) +
  geom_point(alpha=0.5) +
  geom_smooth(method='lm', formula=y~x)
```

## Liveness
Liveness is higly positve skewed, and there is no linear relationship between liveness and rating.
```{r}
skewness(df$liveness)

ggplot(df, aes(liveness)) +
  geom_histogram(bins=20)

ggplot(df, aes(liveness, rating)) +
  geom_point(alpha=0.5) +
  geom_smooth(method='lm', formula=y~x)
```

## Valence
Valence is negatively skewed, and the distribution presents a left high tail. In addition, there is no relationship between valence and rating.
```{r}
skewness(df$valence)

ggplot(df, aes(valence)) +
  geom_histogram(bins=20)

ggplot(df, aes(valence, rating)) +
  geom_point(alpha=0.5) +
  geom_smooth(method='lm', formula=y~x)
```

## Tempo
Tempo is positve skewed, and there is no linear relationship between tempo and rating.
```{r}
skewness(df$tempo)

ggplot(df, aes(tempo)) +
  geom_histogram(bins=20)

ggplot(df, aes(tempo, rating)) +
  geom_point(alpha=0.5) +
  geom_smooth(method='lm', formula=y~x)
```

# Data Cleaning
Because less then 0.01% of the data have a values of 0, we can filter out the values with 0 as tempo. Then, we remove the duplicates in song name from the data, to avoid biases in the model.
```{r}
library(fastDummies); library(tidyverse)

#check percentage values with tempo equal to 0
df %>% 
  filter(tempo == 0) %>% 
  summarise(n_0 = n()/nrow(df))

#remove values from tempo columns equal to 0
df <- df %>% 
  filter(tempo != 0)

#remove songs duplicates
df <- df[!duplicated(df$song),]
```

In addition, we can notice the two important variables in the dataset, performer and genre, are in character format. To understand the importance of the data in a previous attempt I plotted the average rating per artist with over 10 songs produced, and average rating per genre with over 10 songs. As I assumed many of the top performers per average rating were famous artists such as Taylor Swift and Madonna. Moreover, the same happened when I plotted the average rating per genre, some genre have a higher average rating than other probably regardless of all the other variables. 

Therefore, to be able to use the two variables in the analysis we need to have a list of all the unique performers and genres both in the analysis and scoring datasets. 
I combined the two datasets with the rbind function, excluding the rating column in order to match the shapes of the dataframes. By joining the two datasets we make sure that the feature engineering is applied to both datasets.
```{r}
scoringData = read.csv('~/Desktop/Data_Science/Kaggle/PAC_competition/Datasets/lalasongs22/scoringData.csv')

data_combined <- rbind(df[1:18], scoringData)

#replace NA values and '[]' values with "No Genre"
data_combined$genre[data_combined$genre == '[]'] <- "No Genre"
data_combined$genre[is.na(data_combined$genre)] <- "No Genre"
```

# Feature Engineering
This step was one of the most challenging of the entire task.Firstly, I removed the square brackets using the gsub function to be able to have clean strings with values separated by commas. Then, I was able to separate each value by using the separate_row function. Moreover, using a pivot table I created dummy variables filling each dummy per unique genre, and filled each dummy with 1 if the genre was found in the record, and 0 if it was no. 
As for the performer, the string doesn't need any transformation. In addition, each record stores only one performer or a featuring, therefore we don't need to separate values, and we can just create the dummy variables with the pivot table as we did with the genre column.

Before coming up with the solution of the pivot table I tried several unsuccessful options. In particular, I used a for loop inside another for loop to iterate before between each record to add the extracted values in a list, and then use the final list with unique values to create a dummy variable which would assign the values 0 and 1 with an ifelse function. However, this process was highly computation intensive. And the for loop would run an entire day.
```{r}
data_dummies <- data_combined %>% 
  mutate(clean_genre = gsub("\\[|\\]", "", data_combined$genre)) %>% 
  mutate(row = row_number()) %>% 
  separate_rows(clean_genre, sep = ',') %>% 
  pivot_wider(names_from = clean_genre, values_from = clean_genre, values_fn = function(x) 1, values_fill = 0) %>% 
  pivot_wider(names_from = performer, values_from = performer, values_fn = function(x) 1, values_fill = 0) %>% 
  select(-row)
sum(is.na(data_dummies))
```

After performing this data transformation, I split again the two datasets.
```{r}
data <- data_dummies[1:16541,]
data <- cbind(data, rating = df$rating)
scoringData <- data_dummies[16542:21385,]
```

Then, I selected only the numeric data which are relevant to build a predictive model
```{r}
ncol(data)
df<- data %>% 
  select(5,7,8,10,12:8528)

scoringData <- scoringData %>% 
  select(5,7,8,10,12:8526)
```

# Split data
```{r}
set.seed(1031)
split = createDataPartition(y=df$rating, p = 0.75, list = F, groups = 100)
train = df[split,]
test = df[-split,]
```

# XGBoost
For the predictive model I decided to use a XGBoost model. But before reaching this conclusion, I fitted every model we treated in Frameworks class. 
In the beginning, In my first attempt, I didn't create dummy variables for genre and performers, and it was possible to fit linear regression with polynomial terms, GAM models with smoothing, preceeded by feature selection using a hybrid step-wise. In addition, I transformed the variables with high skewness to normalize them using combinations of log, sqrt and 1/x transformations, applying the same transformation both in the analysis data and in the scoring data. For the most complex transformations I used a BoxCox transformation, keeping the same beta for the two datasets. 

However, by implementing the dummy variables the complexity of the model increased exponentially, and it was impossible to use a linear regression or GAM. Indeed, the transformations were over-fitting the data, and even though they scored slightly better in the train data, they score much worse in the test data.
Moreover, I tried to use PCA, but most of the dummy variables had low variance and the model resulted biased. The only non-tree decision models that performed decently was a Lasso regression, for its ability to feature select by shrinking the coefficients. 

Nevertheless, once I tried decision trees based models they highly outperformed Lasso. Most of my attempts then focused on ranger and XGboost. I used ranger and not RandomForest because it's more suitable for larger dataset. In the end, XGBoost resulted the best model in terms of RMSE.
```{r}
library(vtreat); library(xgboost)
ncol(train)
trt = designTreatmentsZ(dframe = train,
                        varlist = names(train)[1:7356])
newvars = trt$scoreFrame[trt$scoreFrame$code%in% c('clean','lev'),'varName']

train_input = prepare(treatmentplan = trt, 
                      dframe = train,
                      varRestriction = newvars)
test_input = prepare(treatmentplan = trt, 
                     dframe = test,
                     varRestriction = newvars)
```

## Bayes Optimization
To tune xgboost I used Bayes Optimization which allows to train the model using different parameters. I defined a function containing the cross validation for xgboost and eta, gamma, max_depth, min_child_weight, subsample, colsample_bytree, nfold as arguments to be trained in the functioned. Also, I defined a list in the function, which will store the RMSE values of each iteration the algorithm will run. Then, I used  Bayes optimization on the function, which try several values for the parameters and adjust them in each iteration for the expected improvement. Moreover, I defined lower and upper limits to tune the parameters in a reasonable range of values. The Bayes Optimization function iterated 50 times finding the optimal hyper-parameters with lowest RMSE.
```{r}
scoring_function <- function(
    eta, gamma, max_depth, min_child_weight, subsample, colsample_bytree, nfold) {
  
  dtrain <- xgb.DMatrix(as.matrix(train_input), label = train$rating, missing = NA)
  
  pars <- list(
    eta = eta,
    gamma = gamma,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    booster = "gbtree",
    objective = 'reg:linear',
    eval_metric = "rmse",
    verbosity = 1
  )
  
  xgbcv <- xgb.cv(
    params = pars,
    data = dtrain,
    nfold = nfold,
    nrounds = 100,
    prediction = TRUE,
    showsd = TRUE,
    early_stopping_rounds = 10,
    maximize = FALSE,
    stratified = TRUE
  )
  
  return(
    list(
      Score= min(xgbcv$evaluation_log$test_rmse_mean),
      nrounds = xgbcv$best_iteration
    )
  )
}

bounds <- list(
  eta = c(0.01, 0.3),
  gamma =c(0, 100),
  max_depth = c(2L, 10L), # L means integers
  min_child_weight = c(1, 25),
  subsample = c(0.5, 0.8),
  colsample_bytree = c(0.5, 0.9),
  nfold = c(3L, 10L)
)
set.seed(2021)

library(vtreat); library(xgboost)
library(ParBayesianOptimization)
time_noparallel <- system.time(
  opt_obj <- bayesOpt(
    FUN = scoring_function,
    bounds = bounds,
    otherHalting = list(timeLimit = Inf, minUtility = 0),
    acq = 'ei',
    initPoints = 8,
    iters.n = 50,
  ))

opt_obj$scoreSummary
```

## Cross Validation
Then, I selected the hyper-parameters for Bayes Optimization with lowest RMSE. To tune the number of rounds the model should run for, I used cross validation to get the number of round for which the model perform the lowest RMSE on the test data, therefore on unseen data, in order to avoid overfitting. I ran the cross validation for 10,000 rounds but setting early_stop_rounds to 50.
```{r}
eta <- opt_obj$scoreSummary$eta[
  which(opt_obj$scoreSummary$Score
        == max(opt_obj$scoreSummary$Score))]
gamma <- opt_obj$scoreSummary$gamma[
  which(opt_obj$scoreSummary$Score
        == max(opt_obj$scoreSummary$Score))]

max_depth <- opt_obj$scoreSummary$max_depth[
  which(opt_obj$scoreSummary$Score
        == max(opt_obj$scoreSummary$Score))]

min_child_weight <- opt_obj$scoreSummary$min_child_weight[
  which(opt_obj$scoreSummary$Score
        == max(opt_obj$scoreSummary$Score))]

subsample <- opt_obj$scoreSummary$subsample[
  which(opt_obj$scoreSummary$Score
        == max(opt_obj$scoreSummary$Score))]

nfold <- opt_obj$scoreSummary$nfold[
  which(opt_obj$scoreSummary$Score
        == max(opt_obj$scoreSummary$Score))]

params <- list(eta = eta,
               gamma = gamma,
               max_depth = max_depth,
               min_child_weight = min_child_weight,
               subsample = subsample,
               nfold = nfold)

# the numrounds which gives the max Score (rmse)

xgbCV <- xgb.cv(params = params,
                data = as.matrix(train_input),
                label = train$rating,
                nrounds = 10000,
                prediction = TRUE,
                showsd = TRUE,
                early_stopping_rounds = 50,
                maximize = FALSE,
                stratified = TRUE)

```

## Fitting Model
I fit the model with the best hyperparameters selected by the Bayes function, and the best number of rounds for the cross validation. Resulting with a RMSE on train data of 12.3 and on test data for 13.7.
```{r}
numrounds <- min(which(xgbCV$evaluation_log$test_rmse_mean == min(xgbCV$evaluation_log$test_rmse_mean)))


fit_tuned <- xgboost(params = params,
                     data = as.matrix(train_input),
                     label = train$rating,
                     nrounds = numrounds,
                     prediction = TRUE,
                     showsd = TRUE,
                     early_stopping_rounds = 10,
                     eval_metric = "rmse")
```

# Predict data
## Predict train data
```{r}
pred_train = predict(fit_tuned,
                     newdata=as.matrix(train_input))
rmse_train_xgboost = sqrt(mean((pred_train - train$rating)^2))
rmse_train_xgboost
```

## Predict test data
```{r}
pred = predict(fit_tuned, 
               newdata=as.matrix(test_input))
rmse_xgboost = sqrt(mean((pred - test$rating)^2))
rmse_xgboost
```

# Predict final data
## Predict
On the final data the model scored a RMSE of 14.4.
```{r}
scoring_input = prepare(treatmentplan = trt, 
                        dframe = scoringData,
                        varRestriction = newvars)
pred = predict(fit_tuned,
               newdata=as.matrix(scoring_input))
submissionFile = data.frame(id = scoringData$id, rating = pred)
write.csv(submissionFile, 'xgb_submission.csv', row.names = F)
```
