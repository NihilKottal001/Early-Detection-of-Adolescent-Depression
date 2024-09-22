library(dplyr)
library(randomForest)
library(pROC)
library(ggplot2)
#library(glmnet)
library(MASS)
library(car)

#### Reading the dataset
da = read.csv("depression.csv", header = T)
dim(da) #6,000 by 12
str(da)

da$mdeSI = factor(da$mdeSI)
da$income = factor(da$income)
da$gender = factor(da$gender, levels = c("Male", "Female")) #Male is the reference group
da$age = factor(da$age)
da$race = factor(da$race, levels = c("White", "Hispanic", "Black", "Asian/NHPIs", "Other")) #white is the reference
da$insurance = factor(da$insurance, levels = c("Yes", "No")) #"Yes" is the reference group
da$siblingU18 = factor(da$siblingU18, levels = c("Yes", "No"))
da$fatherInHH = factor(da$fatherInHH)
da$motherInHH = factor(da$motherInHH)
da$parentInv = factor(da$parentInv)
da$schoolExp = factor(da$schoolExp, levels = c("good school experiences", "bad school experiences"))


(n = dim(da)[1])
set.seed(2024)

index = sample(1:n, 4500) #75% of training and 25% of test data
train = da[index,] 
test = da[-index,]
dim(train)
dim(test)

train <- train[,-2]
test <- test[,-2]


#Data Exploration 

##Plot to analyse the 'year' variable
ggplot(da, aes(x = year, fill = mdeSI)) +
  geom_bar(stat = "count", position = "dodge") +
  labs(title = "Response Analysis",
       x = "Year",
       y = "Count",
       fill = "mdeSI") +
  theme_minimal()


table(da$gender, da$mdeSI)

###Chi square analysis for the data###
chi_squared_results <- list()

for (feature in names(da)) {
  if (feature != "mdeSI" && is.factor(da[[feature]])) { 
    table <- table(da[[feature]], da$mdeSI)
    chi_test <- chisq.test(table)
    chi_squared_results[[feature]] <- chi_test$p.value
  }
}

p_values <- data.frame(Feature = names(chi_squared_results), P_Value = unlist(chi_squared_results))
p_values


##based on the chi square analysis it is evident that the the variables insurance 
##and motherInHH does not influence the target variable

###Generating base model###
rf_model <- randomForest(mdeSI ~ ., data = train, ntree = 500)
rf_preds <- predict(rf_model, test)
CM.test <- table(rf_preds,?test$mdeSI)
importance(rf_model)##checking the gini index
varImpPlot(rf_model)

###calculating performance metrics
(accuracy.test = (CM.test[1,1] + CM.test[2,2])/1500)
(recall.test = CM.test[2,2]/(CM.test[1,2] + CM.test[2,2]))
(precision.test = CM.test[2,2]/(CM.test[2,1] + CM.test[2,2]))

cat(sprintf("Accuracy, recall, and precision for the base model is: %f, %f, %f", accuracy.test, recall.test, precision.test))


####find the optimum ntree and mtry parameter values based on out-of-bag (oob) error####
oob.error.data <- data.frame(
  Trees=rep(1:nrow(rf_model$err.rate), Depression=3),
  Type=rep(c("OOB", "No", "Yes"), each=nrow(rf_model$err.rate)),
  Error=c(rf_model$err.rate[,"OOB"], 
          rf_model$err.rate[,"No"], 
          rf_model$err.rate[,"Yes"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

rf_model <- randomForest(mdeSI ~ ., data=train, ntree=1000, proximity=TRUE)
rf_model


oob.error.data <- data.frame(
  Trees=rep(1:nrow(rf_model$err.rate), Depression=3),
  Type=rep(c("OOB", "No", "Yes"), each=nrow(rf_model$err.rate)),
  Error=c(rf_model$err.rate[,"OOB"], 
          rf_model$err.rate[,"No"], 
          rf_model$err.rate[,"Yes"]))

ggplot(data=oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))


oob.values <- vector(length=10)

###Code to check the optimum value for mtry parameter

set.seed(007)
for(i in 1:10) {
  temp.model <- randomForest(mdeSI ~ ., data=train, mtry=i, ntree=700)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate),1]
}
oob.values

which(oob.values == min(oob.values))
set.seed(1994)
rf_model_para <- randomForest(mdeSI ~ ., 
                      data=train,
                      ntree=700, 
                      mtry=2)

###Running the model1 with updated and optimized parameters
rf_preds_para <- predict(rf_model_para, test)
CM.test <- table(rf_preds_para,?test$mdeSI)

(accuracy.test = (CM.test[1,1] + CM.test[2,2])/1500)

(recall.test = CM.test[2,2]/(CM.test[1,2] + CM.test[2,2]))
(precision.test = CM.test[2,2]/(CM.test[2,1] + CM.test[2,2]))
cat(sprintf("Accuracy, recall, and precision for the base model is: %f, %f, %f", accuracy.test, recall.test, precision.test))



###Now that we have the optimum number of trees and mtry values lets try do some feature engineering and increase the
### model performance


library(dplyr)

da <- read.csv("depression.csv")

da <- da %>%
  mutate(maritalStatus = case_when(
    fatherInHH == "father in hh" & motherInHH == "mother in hh"   ~ "married",
    fatherInHH == "fatherInHH" & motherInHH == "motherInHH"   ~ "married",
    fatherInHH == "no father in hh" & motherInHH == "mother in hh" ~ "single mother",
    fatherInHH == "father in hh" & motherInHH == "no mother in hh" ~ "single father",
    fatherInHH == "no father in hh" & motherInHH == "no mother in hh" ~ "orphan",
    TRUE ~ "unknown"
  ))

da$mdeSI = factor(da$mdeSI)
da$income = factor(da$income)
da$gender = factor(da$gender, levels = c("Male", "Female")) #Male is the reference group
da$age = factor(da$age)
da$race = factor(da$race, levels = c("White", "Hispanic", "Black", "Asian/NHPIs", "Other")) #white is the reference
da$insurance = factor(da$insurance, levels = c("Yes", "No")) #"Yes" is the reference group
da$siblingU18 = factor(da$siblingU18, levels = c("Yes", "No"))
da$fatherInHH = factor(da$fatherInHH)
da$motherInHH = factor(da$motherInHH)
da$parentInv = factor(da$parentInv)
da$schoolExp = factor(da$schoolExp, levels = c("good school experiences", "bad school experiences"))

# As the column faculty cannot be numbered we use one hot encoding to convert them into categorical features

set.seed(2024)
index = sample(1:n, 4500) #75% of training and 25% of test data
train = da[index,] 
test = da[-index,]
dim(train)
dim(test)


train$twelve <- as.numeric(model.matrix(~ age == "12-13", train)[, 2])
train$fourteen <- as.numeric(model.matrix(~ age == "14-15", train)[, 2])
train$sixteen <- as.numeric(model.matrix(~ age == "16-17", train)[, 2])
train$White <- as.numeric(model.matrix(~ race == "White", train)[, 2])
train$Asian <- as.numeric(model.matrix(~ race == "Asian/NHPIs", train)[, 2])
train$Hispanic <- as.numeric(model.matrix(~ race == "Hispanic", train)[, 2])
train$Other <- as.numeric(model.matrix(~ race == "Other", train)[, 2])
train$Black <- as.numeric(model.matrix(~ race == "Black", train)[, 2])
train$married <- as.numeric(model.matrix(~ maritalStatus == "married", train)[, 2])
train$single_mother <- as.numeric(model.matrix(~ maritalStatus == "single mother", train)[, 2])
train$single_father <- as.numeric(model.matrix(~ maritalStatus == "single father", train)[, 2])
train$orphan <- as.numeric(model.matrix(~ maritalStatus == "orphan", train)[, 2])

train <- train[,-2]
train<- train[,-3]
train<- train[,-3]
train<- train[,-5]
train<- train[,-5]

train<- train[,-8]

test$twelve <- as.numeric(model.matrix(~ age == "12-13", test)[, 2])
test$fourteen <- as.numeric(model.matrix(~ age == "14-15", test)[, 2])
test$sixteen <- as.numeric(model.matrix(~ age == "16-17", test)[, 2])
test$White <- as.numeric(model.matrix(~ race == "White", test)[, 2])
test$Asian <- as.numeric(model.matrix(~ race == "Asian/NHPIs", test)[, 2])
test$Hispanic <- as.numeric(model.matrix(~ race == "Hispanic", test)[, 2])
test$Other <- as.numeric(model.matrix(~ race == "Other", test)[, 2])
test$Black <- as.numeric(model.matrix(~ race == "Black", test)[, 2])
test$married <- as.numeric(model.matrix(~ maritalStatus == "married", test)[, 2])
test$single_mother <- as.numeric(model.matrix(~ maritalStatus == "single mother", test)[, 2])
test$single_father <- as.numeric(model.matrix(~ maritalStatus == "single father", test)[, 2])
test$orphan <- as.numeric(model.matrix(~ maritalStatus == "orphan", test)[, 2])

test <- test[,-2]
test<- test[,-3]
test<- test[,-3]
test<- test[,-5]
test<- test[,-5]

test<- test[,-8]



####running the model with the features we have selected
train1 <- train %>% 
  dplyr::select(mdeSI, gender, income, siblingU18, parentInv, schoolExp, twelve, fourteen, sixteen, White, Asian, Hispanic, Other, Black, married, single_mother, single_father, orphan)
test1 <- test %>% 
  dplyr::select(mdeSI, gender, income, siblingU18, parentInv, schoolExp, twelve, fourteen, sixteen, White, Asian, Hispanic, Other, Black, married, single_mother, single_father, orphan)

set.seed(1993)
rf_model2 <- randomForest(mdeSI ~ gender+income+parentInv+schoolExp+twelve+fourteen+White+Asian+Hispanic+Other+Black+single_mother+single_father, data = train1, ntree = 700,mtry = 2)
rf_preds <- predict(rf_model2, test1)
CM.test <- table(rf_preds,?test1$mdeSI)

(accuracy.test = (CM.test[1,1] + CM.test[2,2])/1500)

(recall.test = CM.test[2,2]/(CM.test[1,2] + CM.test[2,2]))
(precision.test = CM.test[2,2]/(CM.test[2,1] + CM.test[2,2]))


cat(sprintf("Accuracy, recall, and precision for the base model is: %f, %f, %f", accuracy.test, recall.test, precision.test))

### ROC and AUC analysis for the random forest model
predictions_rf <- predict(rf_model2, test1, type = "prob")
plot.roc(test1$mdeSI,predictions_rf[,1], percent = TRUE,print.auc = TRUE, legacy.axes = TRUE)

roc_curve <- roc(test1$mdeSI,predictions[,2],legacy.axes=TRUE, plot = TRUE, percent = TRUE)
rf_roc_curve <- plot.roc(test1$mdeSI,predictions[,2], percent = TRUE,print.auc = TRUE,legacy.axes = TRUE)
rf_roc_curve.df <- data.frame( tpp = rf_roc_curve$sensitivities,fpp = (rf_roc_curve$specificities), threshold = rf_roc_curve$thresholds)





####Model 2###Logistic Regression####


lr_model1 <- glm(mdeSI ~ gender+income+parentInv+schoolExp+twelve+fourteen+White+Asian+Hispanic+Other+Black+single_mother+single_father,family=binomial(link="logit"), data = train)
summary(lr_model1)
confint(lr_model1, level = 0.95)

predictions <- predict(lr_model1,test, type = "response")

###Assuming the general threshold of 0.5
predictions <- ifelse(predictions > 0.5,1,0)
CM.test <- table(predictions,?test$mdeSI)

(accuracy.test = (CM.test[1,1] + CM.test[2,2])/1500)

(recall.test = CM.test[2,2]/(CM.test[1,2] + CM.test[2,2]))
(precision.test = CM.test[2,2]/(CM.test[2,1] + CM.test[2,2]))

cat(sprintf("Accuracy, recall, and precision for the logistic model is: %f, %f, %f", accuracy.test, recall.test, precision.test))


###ROC and AUC for the logistic regression model

predictions_lr <- predict(lr_model1, test, type = "response")
plot.roc(test$mdeSI,predictions_lr, percent = TRUE,print.auc = TRUE, legacy.axes = TRUE)

roc_curve_lr <- roc(test$mdeSI,predictions_lr,percent = TRUE,legacy.axes=TRUE)
roc_curve_lr <- plot.roc(test1$mdeSI,predictions_lr, percent = TRUE,print.auc = TRUE,legacy.axes = TRUE )
roc_curve_lr.df <- data.frame( tpp = roc_curve_lr$sensitivities,fpp = roc_curve_lr$specificities, threshold = roc_curve_lr$thresholds)

roc_curve_lr.df[204:220,] ##filtering data to get the range of threshold values having TPP values in the range of 74-75%

###Overlapping ROC and AUC curve for logistic and random forest to compare.

plot.roc(test$mdeSI,predictions_lr, percent = TRUE,print.auc = TRUE, legacy.axes = TRUE, col = "#377eb8")
plot.roc(test1$mdeSI,predictions_rf[,1], percent = TRUE,print.auc = TRUE,print.auc.y =40, legacy.axes = TRUE, add = TRUE,col = "#4daf4a")
legend("bottomright", legend = c("Logistic Regression", "Random Forest"), col = c("#377eb8","#4daf4a"), lwd = 4)


###New recall/accuracy calculation based on custom threshold value

predictions_custom <- predict(lr_model1,test, type = "response")
predictions_custom <- ifelse(predictions_custom > 0.4771,1,0)
CM.test <- table(predictions_custom, test$mdeSI)
(accuracy.test = (CM.test[1,1] + CM.test[2,2])/1500)
(recall.test = CM.test[2,2]/(CM.test[1,2] + CM.test[2,2]))
(precision.test = CM.test[2,2]/(CM.test[2,1] + CM.test[2,2]))

###Plotting TPP, FPP vs threshold analysis curve


ggplot(roc_curve_lr.df, aes(x = threshold)) +
  geom_line(aes(y = fpp, colour = "TPP"), size = 1.2) +
  geom_line(aes(y = tpp, colour = "FPP"), size = 1.2) +
  scale_colour_manual(values = c("TPP" = "deepskyblue4", "FPP" = "tomato3"), 
                      name = "Metric",
                      labels = c("True Positive Rate", "False Positive Rate")) +
  labs(title = "Threshold Analysis",
       subtitle = "Comparison of True Positive and False Positive Rates by Threshold",
       x = "Threshold Value",
       y = "Rate (%)") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5,size = 9),
    legend.position = "bottom",
    legend.title.align = 0.5,
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),
    axis.text.x = element_text(angle = 360, vjust = 0.5, hjust = 1),
    panel.grid.major = element_line(colour = "gray80"),
    panel.grid.minor = element_blank()
  ) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "gray30", size = 0.5) +
  geom_vline(xintercept = 0.4771, linetype = "dashed", colour = "darkgreen", size = 0.5)+
  annotate("text", x = 0.5, y = max(roc_curve_lr.df$fpp, na.rm = TRUE) * 0.95, 
           label = "For threshold = 0.4771, TPP:75.06%, FPP:63.96%", hjust = 0.5, vjust = -1.2, colour = "gray30")




