# ML-for-ILLNM
Run code

1.The packages used are as follows:
1.1The "SimpleImputer" class from the "sklearn.impute" package is used for mode imputation. The "MinMaxScaler" class from the "sklearn.preprocessing" package is used for data correction and normalization of numerical data. The "train_test_split" class from the "sklearn.model_selection" package is used for splitting the dataset. The "feature_selection" module from the "sklearn" package is used for feature variable selection.

1.2For model fitting, the Support Vector Machine (SVM) in linear models is implemented using the "SVC" model class from the "sklearn.svm" package. The basic Decision Tree (DT) is implemented using the "DecisionTreeClassifier" model class from the "sklearn.tree" package. The Random Forest (RF) and Gradient Boosting (CatBoost) in ensemble learning are implemented using the "RandomForestClassifier" and "GradientBoostingClassifier" model classes from the "sklearn.ensemble" package. The K-Nearest Neighbors (KNN) algorithm is implemented using the "KNeighborsClassifier" model class from the "sklearn.neighbors" package. The Gaussian Naive Bayes (GNB) is implemented using the "GaussianNB" model class from the "sklearn.naive_bayes" package. The Neural Network (NN) is implemented using the "MLPClassifier" model class from the "sklearn.neural_network" package.

1.3The evaluation metrics are sourced from various classes in the "sklearn.metrics" package, including accuracy_score, auc, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, and auc. The DCA curve is from the "precision_score" class in the "sklearn.metrics" package. The calibration curve is from the "calibration_curve" class in the "sklearn.calibration" package and the "brier_score_loss" class in the "sklearn.metrics" package. The precision-recall curve is from the "precision_recall_curve" class in the "sklearn.metrics" package.

1.4Model tuning is performed using the "GridSearchCV" class from the "sklearn.model_selection" package. The feature importance ranking is calculated and sorted using the "feature_importances_" attribute of various prediction models.

2 R language code：
#0.import packages
install.packages("tidyverse")
install.packages("caret")
install.packages("dplyr")
install.packages("AER")
install.packages("table1")
install.packages("boot")
library(table1)
 library(boot)
library("dplyr")
library("AER")
install.packages("gtsummary")
install.packages("tidyverse")
install.packages("kableExtra")
library(dplyr)
library(gtsummary)
library(tidyverse)
library(kableExtra)

#1.Descriptive analysis of data.Assigning values to categorical variables.
##1.1 load data
```{r}
data <- read.csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量.csv')
```
##1.2 missing value imputation.
```{r}
install.packages("mice")  
library(mice)
data <-data 
data_with_missing <- data
set.seed(123)  
data_with_missing[sample(nrow(data_with_missing), 100), "prelaryngeal.LNM"] <- NA 
data_with_missing[sample(nrow(data_with_missing), 100), "prelaryngeal.LNMR"] <- NA 
data_with_missing[sample(nrow(data_with_missing), 100), "prelaryngeal.NLNM"] <- NA 
data_with_missing[sample(nrow(data_with_missing), 100), "RLNLNM"] <- NA 
data_with_missing[sample(nrow(data_with_missing), 100), "RLNLNMR"] <- NA 
data_with_missing[sample(nrow(data_with_missing), 100), "RLNNLNM"] <- NA 
data_with_missing[sample(nrow(data_with_missing), 100), "II.level.LNM"] <- NA 
vars_to_impute <- c("prelaryngeal.LNM")  
vars_to_impute <- c("prelaryngeal.LNMR")  
vars_to_impute <- c("prelaryngeal.NLNM")  
vars_to_impute <- c("RLNLNM")   
vars_to_impute <- c("RLNLNMR")  
vars_to_impute <- c("RLNNLNM")  
vars_to_impute <- c("II.level.LNM")  
imputed_data <- mice(data_with_missing, m = 5, maxit = 500, method = "pmm", seed = 123)
# extracting imputed data.
completed_data <- complete(imputed_data)

# viewing/implementing imputed data.
head(completed_data)
```

##1.2 Code for determining cutoff values of continuous numerical variables > Using ROC curve analysis to determine the optimal cutoff points for tumor size, age, and lymph node metastasis.
```{r}
# Assuming you have a data frame named "data" that includes variables for the number of pre-tracheal lymph node metastases and the number of lateral lymph node metastases.
# Import related packages
library(pROC)
roc_obj <- roc(completed_data$age, completed_data$TLNM)
plot(roc_obj, main = "ROC Curve - age vs.TLNM", xlab = "False Positive Rate", ylab = "True Positive Rate")
coords <- coords(roc_obj, "best")
coords
```

## 1.3 Assigning Values to Categorical Variables
### 1.3.1 Assigning values to categorical variables in the dataset with missing values imputed.
```{r}
completed_data$Age<-factor(completed_data$Age,levels = c(0,1),labels = c("age≤43","age>43"))
completed_data$Sex<-factor(completed_data$Sex,levels = c(0,1),labels = c("Female","Male"))
completed_data$Classification.of.BMI<-factor(completed_data$Classification.of.BMI,levels = c(0,1,2,3),labels = c("Underweight", "Normal", "Overweight","obesity"))
completed_data$Tumor.border<-factor(completed_data$Tumor.border,levels = c(0,1,2),labels = c("smooth/borderless","irregular-shape/lsharpobed","extrandular-invasion"))
completed_data$Aspect.ratio<-factor(completed_data$Aspect.ratio,levels = c(0,1),labels = c("≤1", ">1"))
 completed_data$Ingredients<-factor(completed_data$Ingredients,levels = c(0,1,2),labels = c("cystic/cavernous","Mixed cystic and solid","solid"))
completed_data$Internal.echo.pattern<-factor(completed_data$Internal.echo.pattern,levels = c(0,1,2,3),labels ="anechoic", c("high/isoechoic","hypoechoic","very hypoechoic"))
completed_data$Internal.echo.homogeneous<-factor(completed_data$Internal.echo.homogeneous,levels = c(0,1),labels = c("Non-uniform","Uniform"))
completed_data$Calcification<-factor(completed_data$Calcification,levels = c(0,1,2,3),labels = c("no/large comet tail", "coarse calcification","peripheral calcification","Microcalcification"))
completed_data$Tumor.internal.vascularization<-factor(completed_data$Tumor.internal.vascularization,levels = c(0,1),labels = c( "Without","Abundant"))
data$Tumor.Peripheral.blood.flow<-factor(data$Tumor.Peripheral.blood.flow,levels = c(0,1),labels = c("Without","Abundant"))
completed_data$Size<-factor(completed_data$Size,levels = c(0,1),labels = c("≤10.5", ">10.5"))
completed_data$Location<-factor(completed_data$Location,levels = c(1,2,3,4,5),labels = c("upper","Medium","Lower","Multiple","isthmus"))
completed_data$Mulifocality<-factor(completed_data$Mulifocality,levels = c(0,1),labels = c("Abundant", "Without"))
completed_data$Hashimoto<-factor(completed_data$Hashimoto,levels = c(0,1),labels = c("Abundant", "Without"))
completed_data$ETE<-factor(completed_data$ETE,levels = c(0,1),labels = c("Abundant", "Without"))
completed_data$T.staging<-factor(completed_data$T.staging,levels = c(0,1,2,3,4,5,6),labels = c("1a","1b","2","3a","3b","4a","4b"))
completed_data$Side.of.position<-factor(completed_data$Side.of.position,levels = c(0,1,2,3),labels = c("left","right","bilateral" ,"isthmus"))
completed_data$ICLNM<-factor(completed_data$ICLNM,levels = c(0,1),labels = c("No", "Yes"))
completed_data$prelaryngeal.LNM<-factor(completed_data$prelaryngeal.LNM,levels = c(0,1),labels = c("No", "Yes"))
completed_data$pretracheal.LNM<-factor(completed_data$pretracheal.LNM,levels = c(0,1),labels = c("No", "Yes"))
completed_data$paratracheal.LNM<-factor(completed_data$paratracheal.LNM,levels = c(0,1),labels = c("No", "Yes"))
completed_data$ILLNM<-factor(completed_data$ILLNM,levels = c(0,1),labels = c("No", "Yes"))
completed_data$II.level.LNM<-factor(completed_data$II.level.LNM,levels = c(0,1),labels = c("No", "Yes"))
completed_data$III.level.LNM<-factor(completed_data$III.level.LNM,levels = c(0,1),labels = c("No", "Yes"))
completed_data$IV.level.LNM<-factor(completed_data$IV.level.LNM,levels = c(0,1),labels = c("No", "Yes"))
completed_data$RLNLNM<-factor(completed_data$RLNLNM,levels = c(0,1),labels = c("No", "Yes"))
```

###1.3.4 exporting data.
```{r}
#install.packages("writexl")
library(writexl)
data
write.csv(data,"/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/测试集赋值的数据.csv")
```

# 2. Descriptive Statistics
## 2.1 Calculating Mean, Median, Standard Deviation, Minimum, and Maximum

```{r}
mean_value <- mean(data$RLNNLNM)
median_value <- median(data$ICNLNM)
sd_value <- sd(data$RLNNLNM)
min_value <- min(data$ICNLNM)
max_value <- max(data$ICNLNM)
mean_value
median_value
sd_value
min_value
max_value
```

# 3. Splitting the Imported Data into Random Train and Test Sets
## 3.3 Randomly Splitting into Train and Validation Sets
```{r}
data <- read.csv("/Users/zj/Desktop/4.机器学习/数据分析/R语言数据/插补缺失值后赋值的数据.csv")
set.seed(123)
index <- 1:nrow(data)
shuffled_index <- sample(index)
tra_ratio <- 0.7  
val_ratio <- 0.3  
tra_size <- round(tra_ratio * nrow(data))
val_size <- round(val_ratio * nrow(data))
tra_data <- data[shuffled_index[1:tra_size], ]
val_data <- data[shuffled_index[(tra_size + 1):(tra_size + val_size)], ]
cat("训练集观测数量:", nrow(tra_data), "\n")
cat("验证集观测数量:", nrow(val_data), "\n")
```

# 4. Univariate Analysis
## 4.1 Creating a One-Way Table for a Single Target Variable
```{r}
library(table1)
library(boot)
library(dplyr)
data_filled <-data%>%
  mutate_all(~if_else(is.na(.), "N/A", as.character(.)))

print(data_filled)
pvalue <- function(x, ...) {
  y <- unlist(x)
  g <- factor(rep(1:length(x), times=sapply(x, length)))
  if (is.numeric(y)) {
    p <- t.test(y ~ g)$p.value
  } else {
    p <- chisq.test(table(y, g))$p.value
  }
  c("", sub("<", "&lt;", format.pval(p, digits=3, eps=0.001)))
}
  table1(~Age+Sex+Classification.of.BMI+Tumor.border+Aspect.ratio+Ingredients+Internal.echo.pattern+Internal.echo.homogeneous+Calcification+Tumor.Peripheral.blood.flow+Tumor.internal.vascularization+ETE+Size+Location+Mulifocality+Hashimoto+T.staging+Side.of.position+ICLNM+ICLNMR+ICNLNM+prelaryngeal.LNM+prelaryngeal.LNMR+prelaryngeal.NLNM+pretracheal.LNM+pretracheal.LNMR+pretracheal.NLNM+paratracheal.LNM+paratracheal.LNMR+paratracheal.NLNM+RLNLNM+RLNLNMR+RLNNLNM|
         ILLNM,data=data_filled,
       overall = F,
       extra.col = list('p-value'=pvalue),
       topclass = "Rtable1-zebra")

```
###4.2 Univariate analysis for numerical variables.
```{r}
library(dplyr)
library(tidyr)
# Select the required variables
selected_vars <- c("age","size","ICLNMR", "ICNLNM", "prelaryngeal.LNMR", "prelaryngeal.NLNM", "pretracheal.LNMR", "pretracheal.NLNM", "paratracheal.LNMR", "paratracheal.NLNM", "RLNLNMR", "RLNNLNM", "II.level.LNM")
# Create the baseline information table
baseline_table <- data %>%
  select(selected_vars,II.level.LNM) %>%
  pivot_longer(cols = -II.level.LNM, names_to = "variable", values_to = "value") %>%
  group_by(variable, II.level.LNM) %>%
  summarise(mean = mean(value), sd = sd(value), n = n()) %>%
  pivot_wider(names_from =II.level.LNM, values_from = mean) %>%
  mutate(diff = `0` - `1`)
# Display the baseline information table
print(baseline_table)
```

# 5. Multivariate Analysis
## 5.1 Binary Logistic Regression Analysis
### 5.1.1 Importing Packages

```{r}
install.packages("survival")
install.packages('rrtable')
install.packages('magrittr') 
install.packages("ggplot")
install.packages("dplyr")
install.packages("AER")
library("dplyr")
library("AER")
library(openxlsx) 
library(survival) 
library(rrtable)
library(ggplot2)
```
### 5.1.2 Test values
```{r}
chisq.test(train_data$ILLNM,train_data$Age,correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Sex, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Classification.of.BMI, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Tumor.border, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Aspect.ratio, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Ingredients, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Internal.echo.pattern, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Internal.echo.homogeneous, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Calcification, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Tumor.internal.vascularization, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Size, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Location, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Mulifocality, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Hashimoto, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$ETE, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$T.staging, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Side.of.position, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$prelaryngeal.LNM, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$ICLNM, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Classification.of.prelaryngeal.NLNM, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$pretracheal.LNM, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Classification.of.pretracheal.LNMR, correct = T)$expected
  chisq.test(train_data$ILLNM,train_data$Classification.of.pretracheal.NLNM, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Ipsilateral.paratracheal.LNM, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Classification.of.Ipsilateral.paratracheal.LNMR, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Classification.of.Ipsilateral.paratracheal.NLNM, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$RLNLNM, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Classification.of.RLNLNMR, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$ICLNM, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Classification.of.ICLNMR, correct = T)$expected
chisq.test(train_data$ILLNM,train_data$Classification.of.ICNLNM, correct = T)$expected
```

### 5.1.3 Logistic Regression
Performing separate univariate analyses
```{r}
A<-glm(ILLNM~Age+Sex+Tumor.border+Aspect.ratio+Ingredients+Internal.echo.homogeneous+Calcification+Tumor.Peripheral.blood.flow+Tumor.internal.vascularization+ETE+Size+Location+Mulifocality+T.staging+Side.of.position+RLNNLNM,data=data,family = binomial())
summary(A)
coefficients(A)
exp(coefficients(A))
exp(confint(A))
coef<-summary(A)$coefficients[,1]
se<-summary(A)$coefficients[,2]
pvalue<-summary(A)$coefficients[,4]
Results<-cbind(exp(coef),exp(coef-1.96*se),exp(coef+1.96*se),pvalue)
dimnames(Results)[[2]]<-c("OR","LL","UL","p value")
Results
Results=Results[,]
View(Results)
table2docx(Results, add.rownames = FALSE)

```
# 6. Plotting Bar Charts
## 6.1 Code for plotting bar charts and validation curves
```{r}
#install.packages("foreign")
#install.packages("rms")
library(foreign) 
library(rms)
### 6.1.2 Data Integration
x<-as.data.frame(data)
dd<-datadist(data)
options(datadist='dd')
### 6.1.3 Logistic Regression
fit1<-lrm(ILLNM~Tumor.border+Aspect.ratio+Calcification+Tumor.internal.vascularization+ETE+Size+Location+Mulifocality+ICLNM+ICLNMR+ICNLNM+prelaryngeal.LNM+prelaryngeal.LNMR+prelaryngeal.NLNM+pretracheal.LNM+pretracheal.LNMR+pretracheal.NLNM+paratracheal.LNM+paratracheal.LNMR+paratracheal.NLNM+RLNLNM+RLNLNMR+RLNNLNM,data=data,x=T,y=T)
fit1
summary(fit1)
nom1 <- nomogram(fit1, fun=plogis, fun.at=c(.001, .01, .05, seq(.1,.9, by=.1), .95, .99, .999),
lp=F, funlabel="ILLNM")
plot(nom1)
### 6.1.4 Validation Curve
cal1 <- calibrate(fit1, method = 'boot', B = 1000)
plot(cal1, xlim = c(0, 1.0), ylim = c(0, 1.0))

```

## 6.2 Plotting ROC Curve for Bar Charts
### 6.2.1 Calculating Area Under Curve (AUC) and C-value Calculation
```{r}
library(foreign) 
library(rms)
x<-as.data.frame(val_data)
dd<-datadist(val_data)
options(datadist='dd')
fit1<-lrm(ILLNM~Tumor.border+Calcification+ETE+Size+Location+Mulifocality+ICLNM+ICLNMR+ICNLNM+prelaryngeal.LNM+prelaryngeal.LNMR+prelaryngeal.NLNM+pretracheal.LNM+pretracheal.LNMR+pretracheal.NLNM+paratracheal.LNM+paratracheal.LNMR+paratracheal.NLNM+RLNLNM+RLNLNMR+RLNNLNM
          ,data=val_data,x=T,y=T)
fit1
summary(fit1)

```





###### 6.2.2 Plotting ROC Curve - Single Independent Variable
```{r}
# install.packages("pROC")
data <- read.csv("/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/插补缺失值后的数据.csv")
library(pROC)
library(ggplot2)
set.seed(123)
index <- 1:nrow(data)
shuffled_index <- sample(index)
tra_ratio <- 0.7  
val_ratio <- 0.3  
tra_size <- round(tra_ratio * nrow(data))
val_size <- round(val_ratio * nrow(data))
tra_data <- data[shuffled_index[1:tra_size], ]
val_data <- data[shuffled_index[(tra_size + 1):(tra_size + val_size)], ]
cat("训练集观测数量:", nrow(tra_data), "\n")
cat("验证集观测数量:", nrow(val_data), "\n")
fit1 <- glm(ILLNM~Tumor.border+Calcification+ETE+Size+Location+Mulifocality+ICLNM+ICLNMR+ICNLNM+prelaryngeal.LNM+prelaryngeal.LNMR+prelaryngeal.NLNM+pretracheal.LNM+pretracheal.LNMR+pretracheal.NLNM+paratracheal.LNM+paratracheal.LNMR+paratracheal.NLNM+RLNLNM+RLNLNMR+RLNNLNM,
            data =tra_data, family = binomial())
probs <- predict(fit1, newdata = val_data, type = "response")
response <- val_data$ILLNM
response <- factor(response, levels = c(0, 1), labels = c("control", "case"))
roc_obj <- roc(response, probs)
roc_data <- coords(roc_obj, "all")
ggplot(roc_data, aes(x = 1 - specificity, y = sensitivity)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "ILLNM ROC Curves", x = "False positive rate", y = "Ture' positive rate") +
  theme_minimal()
cat("AUC:", auc_value, "\n")
p <- p +
  annotate("text", x = 0.9, y = 0.21, label = paste("AUC =", round(auc(roc_obj1), 2)), color = "blue")

```

### 6.2.3 Plotting ROC Curve for Multiple Independent Variables - Bar Charts
```{r}
fit1 <- glm(ILLNM~Tumor.border+Aspect.ratio+Calcification+Tumor.internal.vascularization+ETE+Size+Location+Mulifocality+ICLNM+ICLNMR+ICNLNM+prelaryngeal.LNM+prelaryngeal.LNMR+prelaryngeal.NLNM+pretracheal.LNM+pretracheal.LNMR+pretracheal.NLNM+paratracheal.LNM+paratracheal.LNMR+paratracheal.NLNM+RLNLNM+RLNLNMR+RLNNLNM,
            data = tra_data, family = binomial())

fit2 <- glm(II.level.LNM~Sex+Aspect.ratio+Calcification+Size+Location+Mulifocality+ICLNM+ICLNMR+ICNLNM+prelaryngeal.LNM+prelaryngeal.LNMR+prelaryngeal.NLNM+pretracheal.LNM+pretracheal.LNMR+pretracheal.NLNM+paratracheal.LNM+paratracheal.LNMR+paratracheal.NLNM+RLNLNM+RLNLNMR+RLNNLNM,
            data = tra_data, family = binomial())

fit3 <- glm(III.level.LNM~Age+Tumor.border+Aspect.ratio+Calcification+Tumor.internal.vascularization+ETE+Size+Mulifocality+ICLNM+ICLNMR+ICNLNM+prelaryngeal.LNM+prelaryngeal.LNMR+prelaryngeal.NLNM+pretracheal.LNM+pretracheal.LNMR+pretracheal.NLNM+paratracheal.LNM+paratracheal.LNMR+paratracheal.NLNM+RLNLNM+RLNLNMR+RLNNLNM,
            data = tra_data, family = binomial())


fit4 <- glm(IV.level.LNM~Tumor.border+Aspect.ratio+Internal.echo.homogeneous+Calcification+Tumor.internal.vascularization+Size+Location+ICLNM+ICLNMR+ICNLNM+prelaryngeal.LNM+prelaryngeal.LNMR+prelaryngeal.NLNM+pretracheal.LNM+pretracheal.LNMR+pretracheal.NLNM+paratracheal.LNM+paratracheal.LNMR+paratracheal.NLNM+RLNLNM+RLNLNMR+RLNNLNM,
            data = tra_data, family = binomial())
probs1 <- predict(fit1, newdata = val_data, type = "response")
probs2 <- predict(fit2, newdata = val_data, type = "response")
probs3 <- predict(fit3, newdata = val_data, type = "response")
probs4 <- predict(fit4, newdata = val_data, type = "response")
response <- val_data$ILLNM
roc_obj1 <- roc(response, probs1)
roc_obj2 <- roc(response, probs2)
roc_obj3 <- roc(response, probs3)
roc_obj4 <- roc(response, probs4)

roc_data1 <- coords(roc_obj1, "all")
roc_data2 <- coords(roc_obj2, "all")
roc_data3 <- coords(roc_obj3, "all")
roc_data4 <- coords(roc_obj4, "all")
p <- ggplot() +
  geom_line(data = roc_data1, aes(x = 1 - specificity, y = sensitivity), color = "blue") +
  geom_line(data = roc_data2, aes(x = 1 - specificity, y = sensitivity), color = "red") +
  geom_line(data = roc_data3, aes(x = 1 - specificity, y = sensitivity), color = "green") +
  geom_line(data = roc_data4, aes(x = 1 - specificity, y = sensitivity), color = "purple") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "ROC Curves", x = "False positive rate", y = "True positive rate") +
  theme_minimal()

print(p)
cat("AUC1:", auc(roc_obj1), "\n")
cat("AUC2:", auc(roc_obj2), "\n")
cat("AUC3:", auc(roc_obj3), "\n")
cat("AUC4:", auc(roc_obj4), "\n")
```


### 6.2.4 Annotating AUC Values on Each ROC Curve - Bottom Right Corner
```{r}
p <- ggplot() +
  geom_line(data = roc_data1, aes(x = 1 - specificity, y = sensitivity), color = "blue") +
  geom_line(data = roc_data2, aes(x = 1 - specificity, y = sensitivity), color = "red") +
  geom_line(data = roc_data3, aes(x = 1 - specificity, y = sensitivity), color = "green") +
  geom_line(data = roc_data4, aes(x = 1 - specificity, y = sensitivity), color = "purple") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "ROC Curves", x = "False positive rate", y = "True positive rate") +
  theme_minimal() +
  theme(panel.spacing = unit(1, "lines"))
p <- p +
  annotate("text", x = 0.75, y = 0.21, label = paste("ILLNM-AUC =", round(auc(roc_obj1), 3)), color = "blue") +
  annotate("text", x = 0.75, y = 0.14, label = paste("II.level.LNM-AUC =", round(auc(roc_obj2), 3)), color = "red") +
  annotate("text", x = 0.75, y = 0.07, label = paste("III.level.LNM-AUC =", round(auc(roc_obj3), 3)), color = "green") +
  annotate("text", x = 0.75, y = 0.01, label = paste("IV.level.LNM-AUC =", round(auc(roc_obj4), 3)), color = "purple")
print(p)
```






## 6.3 Evaluation of Other Types of Data
```{r}
# Calculating Confusion Matrix
confusion <- table(val_data$ILLNM,probs)
# Calculating Precision
precision <- diag(confusion) / colSums(confusion)
# Calculating Specificity
specificity <- diag(confusion) / rowSums(confusion)
# Calculating Sensitivity or Recall
recall <- diag(confusion) / rowSums(confusion)
# Calculating Negative Predictive Value (NPV）
npv <- diag(confusion) / colSums(confusion)
# Calculating Positive Predictive Value (PPV)
ppv <- diag(confusion) / (diag(confusion) + colSums(confusion) - diag(confusion))
# Calculating Accuracy
accuracy <- sum(diag(confusion)) / sum(confusion)
# Calculating F1 Score
f1 <- 2 * (precision * recall) / (precision + recall)
# Calculating False Positive Rate (FP Rate)
fpr <- 1 - specificity
# Calculating Precision
precision <- diag(confusion) / (diag(confusion) + colSums(confusion) - diag(confusion))
# Calculating Average Precision (AP)
ap <- 0  
# Printing Evaluation Metrics
cat("精确度（Precision）:", precision, "\n")
cat("特异性（Specificity）:", specificity, "\n")
cat("灵敏度（Sensitivity）或召回率（Recall）:", recall, "\n")
cat("NPV（Negative Predictive Value）:", npv, "\n")
cat("PPV（Positive Predictive Value）:", ppv, "\n")
cat("准确性（Accuracy）:", accuracy, "\n")
cat("F1评分（F1 Score）:", f1, "\n")
cat("FP率（False Positive Rate）:", fpr, "\n")
cat("AP（Average Precision）:", ap, "\n")
```

 3.Python code.
Install skleran
!pip install -U scikit-learn
!pip install pandas

Import data
import pandas as pd
data2=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/测试集赋值的数据.csv')
data.to_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/导入到爬虫的数据.csv')
data2.head
1.Encoding Categorical Variables
data2.head
data_category = data2.select_dtypes(include=['object'])
data_category
data_Number=data2.select_dtypes(exclude=['object'])
data_Number
data_Number.columns.values
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
encoder.fit(data_category)
data_category_enc = pd.DataFrame(encoder.transform(data_category), columns=data_category.columns)
data_category_enc
data_category_enc['Age'].value_counts()
data_category['Age'].value_counts()
data_enc=pd.concat([data_category_enc,data_Number],axis=1)
data_enc
data_enc.to_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/测试集编码后的数据.csv')

2、Missing Value Imputation
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data_encImpute = pd.DataFrame(imp.fit_transform(data_enc))
data_encImpute.columns = data_enc.columns
data_encImpute
data_encImpute['II.level.LNM'].value_counts()
data_encImpute.to_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/2.测试集插补缺失值后的数据.csv')
3、Data Correction and Normalization for Numerical Data
data_scale=data_encImpute
target=data_encImpute['ILLNM'].astype(int)
target.value_counts()
from sklearn import preprocessing
scaler=preprocessing.StandardScaler()
data_scaled=pd.DataFrame(scaler.fit_transform(data_scale))
data_scaled.columns=data_scale.columns
data_scaled
data_encImpute.to_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据.csv')
4、Dimensionality Reduction (Reducing the issue of multicollinearity among factors)
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量_ILLNM.csv')
data.iloc[:,[0,2,4]]
data.shape
data.info()
#4.1Removing Low Variance Features
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8))) 
data_sel = sel.fit_transform(data)
data_sel
a=sel.get_support(indices=True)
a
data.iloc[:,a]
data_sel=data.iloc[:,a]
data_sel.info()
#4.2Univariate Feature Selection
from sklearn.feature_selection import SelectKBest, chi2
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量_ILLNM.csv')
data_feature = data[['Age','Sex','Classification.of.BMI','Tumor.border','Aspect.ratio','Ingredients','Internal.echo.pattern','Internal.echo.homogeneous','Calcification','Tumor.Peripheral.blood.flow','Tumor.internal.vascularization','ETE','Size','Location','Mulifocality','Hashimoto','T.staging','Side.of.position',
'ICLNM','prelaryngeal.LNM','pretracheal.LNM','paratracheal.LNM','RLNLNM',
'age','size',
'ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM',
'paratracheal.LNMR','paratracheal.NLNM',
'RLNLNMR','RLNNLNM']]
data_feature.shape
data_target=data['ILLNM']
data_target.unique()
set_kit=SelectKBest(chi2,k=20)
data_sel=set_kit.fit_transform(data_feature,data_target)
data_sel.shape
a=set_kit.get_support(indices=True)
a
data_sel=data_feature.iloc[:,a]
data_sel
data_sel.info()
#4.3Recursive Feature Elimination (RFE) - Linear Models
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR 
from sklearn.model_selection import cross_val_score 
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量_ILLNM.csv')
data_feature = data[['Age','Sex','Classification.of.BMI','Tumor.border','Aspect.ratio','Ingredients','Internal.echo.pattern','Internal.echo.homogeneous','Calcification','Tumor.Peripheral.blood.flow','Tumor.internal.vascularization','ETE','Size','Location','Mulifocality','Hashimoto','T.staging','Side.of.position',
'ICLNM','prelaryngeal.LNM','pretracheal.LNM','paratracheal.LNM','RLNLNM',
'age','size',
'ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM',
'paratracheal.LNMR','paratracheal.NLNM',
'RLNLNMR','RLNNLNM']]
data_feature.shape
estimator=SVR(kernel='linear')
sel=RFE(estimator,n_features_to_select=25,step=1) 
data_target=data['ILLNM']
data_target.unique()
sel.fit(data_feature,data_target)
a=sel.get_support(indices=True)
a
data_sel=data_feature.iloc[:,a]
data_sel=data_feature.iloc[:,a]
data_sel
data_sel.info()
#4.4RFECV (Recursive Feature Elimination with Cross-Validation)
RFC_ = RandomForestClassifier() 
RFC_.fit(data_sel, data_target)
c = RFC_.feature_importances_
print('重要性：')
print(c)
selector = RFECV(RFC_, step=1, cv=10,min_features_to_select=10) 
selector.fit(data_sel, data_target)
X_wrapper = selector.transform(data_sel)
score = cross_val_score(RFC_, X_wrapper, data_target, cv=10).mean() 
print(score)
print('最佳数量和排序')
print(selector.support_)
print(selector.n_features_)
print(selector.ranking_)
a = selector.get_support(indices=True)
data_sel.iloc[:,a]
data_sel=data_feature.iloc[:,a]
data_sel.info()
import matplotlib.pyplot as plt
score = []
best_score = 0
best_features = 0
for i in range(1, 8):
X_wrapper = RFE(RFC_, n_features_to_select=i, step=1).fit_transform(data_sel, data_target) # 最优特征
once = cross_val_score(RFC_, X_wrapper, data_target, cv=10).mean()
score.append(once)
if once > best_score:
best_score = once
best_features = i
print("当前最高得分:", best_score)
print("最佳特征数量:", best_features)
print("得分列表:", score)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 8), score)
plt.show()
from sklearn.model_selection import StratifiedKFold
rfecv=RFECV(estimator=RFC_,step=1,cv=StratifiedKFold(2),scoring='accuracy')
rfecv.fit(data,data_target)
print("最优特征数量：%d" % rfecv.n_features_)
print("选择的特征：", rfecv.support_)
print("特征排名：", rfecv.ranking_)
print("Optimal number of features: %d" % selector.n_features_)
# plot number of features vs. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (number of correct classifications)")
plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), selector.cv_results_['mean_test_score'])
plt.show()
print("Optimal number of features: %d" % rfecv.n_features_)
# plot number of features vs. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (number of correct classifications)")
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
plt.show()
rfecv.get_support(indices=True)
a
data.iloc[:,a]
data_sel=data.iloc[:,a]
data_sel.info()
#4.5 L1 Regularization-based Feature Selection - SelectFromModel with Linear Regression and LinearSVC for Classification
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
clf = LogisticRegression()
clf.fit(data_feature, data_target)

model = SelectFromModel(clf, prefit=True)
data_new = model.transform(data_feature)
model.get_support(indices=True)
a=model.get_support(indices=True)
data_features=pd.DataFrame(data_feature)
data_features.columns=data_feature.columns
data_featurenew=data_features.iloc[:,a]
data_featurenew
data_featurenew.info()
#4.6Tree-based Feature Selection
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量_ILLNM.csv')
data_feature = data[['Age','Sex','Classification.of.BMI','Tumor.border','Aspect.ratio','Ingredients','Internal.echo.pattern','Internal.echo.homogeneous','Calcification','Tumor.Peripheral.blood.flow','Tumor.internal.vascularization','ETE','Size','Location','Mulifocality','Hashimoto','T.staging','Side.of.position',
'prelaryngeal.LNM','pretracheal.LNM','paratracheal.LNM','RLNLNM','ICLNM',
'age','size',
'ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM',
'paratracheal.LNMR','paratracheal.NLNM',
'RLNLNMR','RLNNLNM']]
data_target=data['ILLNM']
data_target.unique()
clf = ExtraTreesClassifier()
clf.fit(data_feature, data_target)
clf.feature_importances_
model=SelectFromModel(clf,prefit=True)
x_new=model.transform(data_feature)
x_new
model.get_support(indices=True)
a=model.get_support(indices=True)
data_features=pd.DataFrame(data_feature)
data_features.columns=data_feature.columns
data_featurenew=data_features.iloc[:,a]
data_featurenew
data_featurenew.info()
#5、Various Prediction Models - Ridge Regression and Lasso Regression
## 5.1.1 Loading and Correcting Data
from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量_ILLNM.csv')
data_featureCata=data[['Sex','Aspect.ratio','Calcification','Size','Location','Mulifocality',
'ICLNM','prelaryngeal.LNM',
'pretracheal.LNM','paratracheal.LNM','RLNLNM',
]]
data_featureNum=data[['ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM','paratracheal.LNMR',
'paratracheal.NLNM','RLNLNMR','RLNNLNM']]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureNum.shape
data_featureCata=np.array(data_featureCata)
data_featureCata
import numpy as np
data_feature = np.hstack((data_featureCata, data_featureNum))
data_feature.shape
## 5.1.2 Splitting into Training and Validation Sets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
data_targetclass = data['ILLNM']
data_targetNum=data['ILLNM']

class_x_tra, class_x_val, class_y_tra, class_y_val = train_test_split(data_feature, data_targetclass, test_size=0.3, random_state=0)
reg_x_tra, reg_x_val, reg_y_tra, reg_y_val = train_test_split(data_feature, data_targetNum, test_size=0.3, random_state=0)
print(class_x_tra.shape,class_y_tra.shape)
## 5.1.3 Model Fitting
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
clf=RidgeClassifier(alpha=1,fit_intercept=True)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
accuracy = clf.score(class_x_val, class_y_val)
print("Validation Accuracy:", accuracy)
clf=RidgeClassifierCV(alphas=[1e-3,1e-2,1e-1,1],cv=10)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
clf.predict(class_x_val)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import numpy as np
n_alphas =200
alphas = np.logspace(-10 -2, n_alphas)
print(alphas)
coefs=[]
for a in alphas:
ridge=Ridge(alpha=a,fit_intercept=False)
ridge.fit(class_x_tra,class_y_tra)
coef=ridge.coef_
coefs.append(ridge.coef_) 
coefs len(coefs)
# Ridge Trace Plot
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
coefs = []
for a in alphas:
ridge = Ridge(alpha=a, fit_intercept=False)
ridge.fit(class_x_tra, class_y_tra)
coef = ridge.coef_
coefs.append(coef)
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) 
plt.xlabel('ILLNM alpha')
plt.ylabel('coefficient')
plt.title('Ridge Coefficient Plot')
plt.axis('tight')
plt.show()
array=([1,101,201,301,401,501,601,701,801,901])
Ridge_=RidgeCV(alphas=np.arange(1,1001,100),store_cv_values=True)
Ridge_.fit(reg_x_tra,reg_y_tra )
Ridge_.score(reg_x_val,reg_y_val)
# Cross-Validation Results
Ridge_.score(reg_x_val,reg_y_val)
Ridge_.cv_values_.shape
Ridge_.cv_values_.mean(axis=0)
Ridge_.alpha_
## 5.1.4 Evaluation Metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
y_pred = clf.predict(class_x_val)
y_pred_proba = clf.decision_function(class_x_val)
accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, y_pred_proba)
conf_matrix = confusion_matrix(class_y_val, y_pred)
precision = precision_score(class_y_val, y_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
sensitivity = recall_score(class_y_val, y_pred)
npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
ppv = precision
recall = sensitivity
f1_score = f1_score(class_y_val, y_pred)
false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0])
print("Accuracy:", accuracy)
print("AUC:", auc)
print("DCA:", net_benefit) 
print("Precision:", precision)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
print("Negative Predictive Value:", npv)
print("Positive Predictive Value:", ppv)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("False Positive Rate:", false_positive_rate)
labels = ['Accuracy', 'AUC', 'DCA', 'Precision', 'Specificity', 'Sensitivity', 'NPV', 'PPV', 'Recall', 'F1 Score', 'FPR']
values = [accuracy, auc, net_benefit, precision, specificity, sensitivity, npv, ppv, recall, f1_score, false_positive_rate]
plt.figure(figsize=(8, 6))
plt.bar(labels, values)
plt.xlabel('Evaluation Metric')
plt.ylabel('Value')
plt.title('Model Evaluation Metrics')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
y_pred_proba = clf.decision_function(class_x_val)

# Calculating ROC Curve
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
auc = roc_auc_score(class_y_val, y_pred_proba)
# Plotting ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM Ridge regression ROC curve')
plt.legend(loc="lower right")
plt.show()
####LASSO
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV

clf = Lasso(alpha=0.1, fit_intercept=True)
clf.fit(class_x_tra, class_y_tra)

accuracy = clf.score(class_x_val, class_y_val)
print("Validation Accuracy:", accuracy)
clf=LassoCV(alphas=[1e-3,1e-2,1e-1,1],cv=5,max_iter=10000)

clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
# 5.2 Decision Tree
## 5.2.1 Loading Data
from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量_ILLNM.csv')
data_feature = data[['Tumor.border','Aspect.ratio','Calcification',
'Tumor.internal.vascularization','ETE','Size','Location',
'Mulifocality','ICLNM','prelaryngeal.LNM',
'pretracheal.LNM','paratracheal.LNM','RLNLNM',
'ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM','pretracheal.LNMR','pretracheal.NLNM','paratracheal.LNMR','paratracheal.NLNM','RLNLNMR','RLNNLNM'
]]
data_target=data['ILLNM']
data_target.unique()
data_featureCata=data[['Tumor.border','Aspect.ratio','Calcification',
'Tumor.internal.vascularization','ETE','Size','Location',
'Mulifocality','ICLNM','prelaryngeal.LNM',
'pretracheal.LNM','paratracheal.LNM','RLNLNM',
]]
data_featureNum=data[['ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM','paratracheal.LNMR',
'paratracheal.NLNM','RLNLNMR','RLNNLNM']]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureCata=np.array(data_featureCata)
import numpy as np
data_feature = np.hstack((data_featureCata, data_featureNum))
## 5.2.2 Splitting Data
data_feature.shape
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

class_x_tra, class_x_val, class_y_tra, class_y_val = train_test_split(data_feature, data_target, test_size=0.3, random_state=0)
print(class_x_tra.shape,class_x_val.shape)
### 5.2.3 Decision Tree Analysis Code
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
import graphviz
clf = DecisionTreeClassifier(min_samples_split=2)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
clf = ExtraTreeClassifier()
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
feature_name=['Tumor.border','Aspect.ratio','Calcification',
'Tumor.internal.vascularization','ETE','Size','Location',
'Mulifocality','ICLNM','prelaryngeal.LNM',
'pretracheal.LNM','paratracheal.LNM','RLNLNM',
'ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM','pretracheal.LNMR','pretracheal.NLNM','paratracheal.LNMR','paratracheal.NLNM','RLNLNMR','RLNNLNM']
[*zip(feature_name,clf.feature_importances_)]
tree_dot = export_graphviz(clf, out_file=None, 
feature_names=feature_names, 
class_names=['No', 'Yes'], 
filled=True, rounded=True, 
special_characters=True)
### 5.2.3 Pruning the Decision Tree
clf=DecisionTreeClassifier(min_samples_split=3,max_depth=5,min_samples_leaf=10,max_features=20)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
clf.feature_importances_
[*zip(feature_name,clf.feature_importances_)]
### 5.2.4 Evaluation Metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
y_pred = clf.predict(class_x_val)
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]
accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, y_pred_proba)
conf_matrix = confusion_matrix(class_y_val, y_pred)
precision = precision_score(class_y_val, y_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
sensitivity = recall_score(class_y_val, y_pred)
npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
ppv = precision
recall = sensitivity
f1_score = f1_score(class_y_val, y_pred)
false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0])
print("Accuracy:", accuracy)
print("AUC:", auc)
print("DCA:", net_benefit) 
print("Precision:", precision)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
print("Negative Predictive Value:", npv)
print("Positive Predictive Value:", ppv)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("False Positive Rate:", false_positive_rate)
labels = ['Accuracy', 'AUC', 'DCA', 'Precision', 'Specificity', 'Sensitivity', 'NPV', 'PPV', 'Recall', 'F1 Score', 'FPR']
values = [accuracy, auc, net_benefit, precision, specificity, sensitivity, npv, ppv, recall, f1_score, false_positive_rate]

plt.figure(figsize=(8, 6))
plt.bar(labels, values)
plt.xlabel('Evaluation Metric')
plt.ylabel('Value')
plt.title('Model Evaluation Metrics')
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)
# Plotting ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM Decision tree ROC curve')
plt.legend(loc="lower right")
plt.show()
# 5.3 Random Forest (Ensemble Learning - Part 1)
## 5.3.1 Loading Data
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量.csv')
data_feature = data[['Tumor.border','Aspect.ratio','Calcification',
'Tumor.internal.vascularization','ETE','Size','Location',
'Mulifocality','ICLNM','prelaryngeal.LNM',
'pretracheal.LNM','paratracheal.LNM','RLNLNM',

'ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM','paratracheal.LNMR',
'paratracheal.NLNM','RLNLNMR','RLNNLNM'
]]
data_target=data['ILLNM']
data_target.unique()
data_featureCata=data[['Tumor.border','Aspect.ratio','Calcification',
'Tumor.internal.vascularization','ETE','Size','Location',
'Mulifocality','ICLNM','prelaryngeal.LNM',
'pretracheal.LNM','paratracheal.LNM','RLNLNM',
]]


data_featureNum=data[['ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM','paratracheal.LNMR',
'paratracheal.NLNM','RLNLNMR','RLNNLNM']]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureCata=np.array(data_featureCata)
import numpy as np
data_feature = np.hstack((data_featureCata, data_featureNum))
### 5.3.2 Categorical Data
data_feature.shape
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

class_x_tra, class_x_val, class_y_tra, class_y_val = train_test_split(data_feature, data_target, test_size=0.3, random_state=0)
print(class_x_tra.shape,class_x_val.shape)
### 5.3.3 Model Code
clf=RandomForestClassifier(n_estimators=25000,max_depth=5,min_samples_leaf=5,min_samples_split=5,)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
clf.feature_importances_
feature_name=['Tumor.border','Aspect.ratio','Calcification',
'Tumor.internal.vascularization','ETE','Size','Location',
'Mulifocality','ICLNM','prelaryngeal.LNM',
'pretracheal.LNM','paratracheal.LNM','RLNLNM','ICLNMR','ICNLNM','prelaryngeal.LNMR','prelaryngeal.NLNM',
'pretracheal.LNMR','pretracheal.NLNM','paratracheal.LNMR','paratracheal.NLNM','RLNLNMR','RLNNLNM']
[*zip(feature_name,clf.feature_importances_)]
### 5.3.4 Feature Importance Selection
import matplotlib.pyplot as plt
import numpy as np
importances = clf.feature_importances_
positive_importances = [imp for imp in importances if imp > 0]
negative_importances = [imp for imp in importances if imp < 0]
positive_indices = np.argsort(positive_importances)[::-1]
negative_indices = np.argsort(negative_importances)
positive_features = [feature_name[i] for i in positive_indices]
positive_importances = [positive_importances[i] for i in positive_indices]
negative_features = [feature_name[i] for i in negative_indices]
negative_importances = [abs(negative_importances[i]) for i in negative_indices]
plt.figure(figsize=(10, 6))
plt.barh(range(len(positive_features)), positive_importances, tick_label=positive_features, color='blue', label='Positive Importance')
plt.barh(range(len(negative_features)), negative_importances, tick_label=negative_features, color='red', label='Negative Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
for i in range(len(positive_features)):
plt.text(positive_importances[i], i, f'{positive_importances[i]:.2f}', ha='left', va='center')
for i in range(len(negative_features)):
plt.text(negative_importances[i], i + len(positive_features), f'{negative_importances[i]:.2f}', ha='left', va='center')
plt.gca().invert_yaxis()
plt.legend()
plt.show()
import matplotlib.pyplot as plt
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = [feature_name[i] for i in indices]
importances = importances[indices]
plt.figure(figsize=(10, 6))
plt.barh(range(len(features)), importances, tick_label=features, color='palegoldenrod') 
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('ILLNM Feature Importances')
for i in range(len(features)):
plt.text(importances[i], i, f'{importances[i]:.2f}', ha='left', va='center')
plt.gca().invert_yaxis() plt.show()
import matplotlib.pyplot as plt
colors = ['palegreen', 'hotpink', 'palegoldenrod', 'skyblue']
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = [feature_name[i] for i in indices[:10]]
top_importances = importances[indices[:10]]
plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), top_importances, tick_label=top_features, color='palegoldenrod') plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('ILLNM Top 10 Feature Importances')
for i in range(len(top_features)):
plt.text(top_importances[i], i, f'{top_importances[i]:.2f}', ha='left', va='center')
plt.gca().invert_yaxis() plt.show()
import matplotlib.pyplot as plt
### 5.3.4 Evaluation Metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
y_pred = clf.predict(class_x_val)
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]
accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, y_pred_proba)
conf_matrix = confusion_matrix(class_y_val, y_pred)
precision = precision_score(class_y_val, y_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
sensitivity = recall_score(class_y_val, y_pred)
npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
ppv = precision
recall = sensitivity
f1_score = f1_score(class_y_val, y_pred)
false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0])
print("Accuracy:", accuracy)
print("AUC:", auc)
print("DCA:", net_benefit) print("Precision:", precision)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
print("Negative Predictive Value:", npv)
print("Positive Predictive Value:", ppv)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("False Positive Rate:", false_positive_rate)
labels = ['Accuracy', 'AUC', 'DCA', 'Precision', 'Specificity', 'Sensitivity', 'NPV', 'PPV', 'Recall', 'F1 Score', 'FPR']
values = [accuracy, auc, net_benefit, precision, specificity, sensitivity, npv, ppv, recall, f1_score, false_positive_rate]
plt.figure(figsize=(8, 6))
plt.bar(labels, values)
plt.xlabel('Evaluation Metric')
plt.ylabel('Value')
plt.title('Model Evaluation Metrics')
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM Random forest ROC curve')
plt.legend(loc="lower right")
plt.show()
# 5.4 Gradient Boosting
### 5.4.4 Model Code - Similar to Random Forest and Decision Tree
clf=GradientBoostingClassifier(n_estimators=25000,max_depth=7,learning_rate=0.01,)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
y_pred = clf.predict(class_x_val)
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]
accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, y_pred_proba)
conf_matrix = confusion_matrix(class_y_val, y_pred)
precision = precision_score(class_y_val, y_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
sensitivity = recall_score(class_y_val, y_pred)
npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
ppv = precision
recall = sensitivity
f1_score = f1_score(class_y_val, y_pred)
false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0])
print("Accuracy:", accuracy)
print("AUC:", auc)
print("DCA:", net_benefit) print("Precision:", precision)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
print("Negative Predictive Value:", npv)
print("Positive Predictive Value:", ppv)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("False Positive Rate:", false_positive_rate)
labels = ['Accuracy', 'AUC', 'DCA', 'Precision', 'Specificity', 'Sensitivity', 'NPV', 'PPV', 'Recall', 'F1 Score', 'FPR']
values = [accuracy, auc, net_benefit, precision, specificity, sensitivity, npv, ppv, recall, f1_score, false_positive_rate]

plt.figure(figsize=(8, 6))
plt.bar(labels, values)
plt.xlabel('Evaluation Metric')
plt.ylabel('Value')
plt.title('Model Evaluation Metrics')
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
auc = roc_auc_score(class_y_val, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Gradient Boosting (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM Gradient boosting ROC curve')
plt.legend(loc='lower right')
plt.show()


# 5.5 Support Vector Machines (SVM) - SVC (Support Vector Classification) for Binary Classification
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据删除一些不必要的变量的数据.csv')
data_feature = data[['Tumor.border', 'Aspect.ratio', 'Calcification', 'Tumor.internal.vascularization', 'Tumor.Peripheral.blood.flow',
'Size','Location','Hashimoto','ETE',
'ICLNM','Classification.of.ICNLNM',
'Classification.of.prelaryngeal.LNMR','Classification.of.pretracheal.LNMR',
'Classification.of.paratracheal.LNMR','ICLNMR','prelaryngeal.LNMR','pretracheal.LNMR',
'RLNLNMR',
]]
data_target=data['ILLNM']
data_target.unique()
data_featureCata=data[['Tumor.border', 'Aspect.ratio', 'Calcification', 'Tumor.internal.vascularization', 'Tumor.Peripheral.blood.flow',
'Size','Location','Hashimoto','ETE',
'ICLNM','Classification.of.ICNLNM',
'Classification.of.prelaryngeal.LNMR','Classification.of.pretracheal.LNMR','Classification.of.paratracheal.LNMR',
]]
data_featureNum=data[['ICLNMR','prelaryngeal.LNMR','pretracheal.LNMR',
'RLNLNMR',]]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureCata=np.array(data_featureCata)
import numpy as np
data_feature = np.hstack((data_featureCata, data_featureNum))from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import feature_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据删除一些不必要的变量的数据.csv')
from sklearn.model_selection import train_test_split
data_feature=data[['Age','Sex','Tumor.border','Aspect.ratio','Internal.echo.pattern','Internal.echo.homogeneous','Calcification','Tumor.Peripheral.blood.flow',
'Size','Location','Mulifocality','ETE','Side.of.position',
'ICLNM','Classification.of.ICLNMR','Classification.of.ICNLNM',
'prelaryngeal.LNM','Classification.of.prelaryngeal.LNMR','Classification.of.prelaryngeal.NLNM',
'pretracheal.LNM','Classification.of.pretracheal.LNMR','Classification.of.pretracheal.NLNM',
'paratracheal.LNM','Classification.of.paratracheal.NLNM','Classification.of.paratracheal.LNMR',
'RLNLNM','Classification.of.RLNLNMR','Classification.of.RLNNLNM']]
### 5.5.3 Model Fitting
from sklearn.svm import SVR
from sklearn.svm import SVC
clf=SVC(kernel='linear',gamma=0.2)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
### 5.5.4 Evaluation Metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
y_pred = clf.predict(class_x_val)
y_pred_proba = clf.decision_function(class_x_val)
accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, y_pred_proba)
conf_matrix = confusion_matrix(class_y_val, y_pred)
precision = precision_score(class_y_val, y_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
sensitivity = recall_score(class_y_val, y_pred)
npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
ppv = precision
recall = sensitivity
f1_score = f1_score(class_y_val, y_pred)
false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0])
print("Accuracy:", accuracy)
print("AUC:", auc)
print("DCA:", net_benefit) print("Precision:", precision)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
print("Negative Predictive Value:", npv)
print("Positive Predictive Value:", ppv)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("False Positive Rate:", false_positive_rate)
labels = ['Accuracy', 'AUC', 'DCA', 'Precision', 'Specificity', 'Sensitivity', 'NPV', 'PPV', 'Recall', 'F1 Score', 'FPR']
values = [accuracy, auc, net_benefit, precision, specificity, sensitivity, npv, ppv, recall, f1_score, false_positive_rate]

plt.figure(figsize=(8, 6))
plt.bar(labels, values)
plt.xlabel('Evaluation Metric')
plt.ylabel('Value')
plt.title('Model Evaluation Metrics')
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
y_pred_proba = clf.decision_function(class_x_val)
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
auc = roc_auc_score(class_y_val, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='SVC (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--') # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of SVM')
plt.legend(loc='lower right')
plt.show()


# 5.6 K-Nearest Neighbors (KNN)
### 5.6.1 Loading Data
data=pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据删除一些不必要的变量的数据.csv')
data_feature = data[['Tumor.border', 'Aspect.ratio', 'Calcification', 'Tumor.internal.vascularization', 'Tumor.Peripheral.blood.flow',
'Size','Location','Hashimoto','ETE',
'ICLNM','Classification.of.ICNLNM',
'Classification.of.prelaryngeal.LNMR','Classification.of.pretracheal.LNMR',
'Classification.of.paratracheal.LNMR','ICLNMR','prelaryngeal.LNMR','pretracheal.LNMR',
'RLNLNMR',
]]
data_target=data['ILLNM']
data_target.unique()
data_featureCata=data[['Tumor.border', 'Aspect.ratio', 'Calcification', 'Tumor.internal.vascularization', 'Tumor.Peripheral.blood.flow',
'Size','Location','Hashimoto','ETE',
'ICLNM','Classification.of.ICNLNM',
'Classification.of.prelaryngeal.LNMR','Classification.of.pretracheal.LNMR','Classification.of.paratracheal.LNMR',
]]
data_featureNum=data[['ICLNMR','prelaryngeal.LNMR','pretracheal.LNMR',
'RLNLNMR',]]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureCata=np.array(data_featureCata)
import numpy as np
data_feature = np.hstack((data_featureCata, data_featureNum))
### 5.6.2 Data Classification
data_feature.shape
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
class_x_tra, class_x_val, class_y_tra, class_y_val = train_test_split(data_feature, data_target, test_size=0.3, random_state=0)
print(class_x_tra.shape,class_x_val.shape)

### 5.6.3 Model Fitting
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
best_score = 0
best_k = None
for i in range(1, 51):
clf = KNeighborsClassifier(n_neighbors=i, leaf_size=40, n_jobs=6)
clf.fit(class_x_tra, class_y_tra)
score = clf.score(class_x_val, class_y_val)
if score > best_score:
best_score = score
best_k = i
print("Best score:", best_score)
print("Best k:", best_k)
clf=KNeighborsClassifier(n_neighbors=11,leaf_size=40,n_jobs=6)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
5.6.4 Evaluation indicators
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
y_pred = clf.predict(class_x_val)
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]

accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, y_pred_proba)
conf_matrix = confusion_matrix(class_y_val, y_pred)
precision = precision_score(class_y_val, y_pred)
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
sensitivity = recall_score(class_y_val, y_pred)
npv = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
ppv = precision
recall = sensitivity
f1_score = f1_score(class_y_val, y_pred)
false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0])
print("Accuracy:", accuracy)
print("AUC:", auc)
print("DCA:", net_benefit) # 假设已经计算了DCA的net_benefit
print("Precision:", precision)
print("Specificity:", specificity)
print("Sensitivity:", sensitivity)
print("Negative Predictive Value:", npv)
print("Positive Predictive Value:", ppv)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("False Positive Rate:", false_positive_rate)
labels = ['Accuracy', 'AUC', 'DCA', 'Precision', 'Specificity', 'Sensitivity', 'NPV', 'PPV', 'Recall', 'F1 Score', 'FPR']
values = [accuracy, auc, net_benefit, precision, specificity, sensitivity, npv, ppv, recall, f1_score, false_positive_rate]
plt.figure(figsize=(8, 6))
plt.bar(labels, values)
plt.xlabel('Evaluation Metric')
plt.ylabel('Value')
plt.title('Model Evaluation Metrics')
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
auc = roc_auc_score(class_y_val, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='KNN (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], 'k--') # 绘制对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM KNN ROC Curve ')
plt.legend(loc='lower right')
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(class_y_val, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ILLNM KNN Precision-Recall Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True)
plt.show()
# 5.7 Naive Bayes Classifier - Bernoulli Naive Bayes for Binary Classification
### 5.7.3 Model Fitting (Similar to previous code, no need to repeat)
from sklearn.naive_bayes import BernoulliNB #二分类的
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0], 'fit_prior': [True, False]}
clf = BernoulliNB()
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(class_x_tra, class_y_tra)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters: ", best_params)
print("Best Score: ", best_score)
best_clf = BernoulliNB(alpha=best_params['alpha'], fit_prior=best_params['fit_prior'])
best_clf.fit(class_x_tra, class_y_tra)
accuracy = best_clf.score(class_x_val, class_y_val)
print("Validation Accuracy: ", accuracy)
5.7.4 Evaluation indicators
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
y_pred = best_clf.predict(class_x_val)
y_pred_proba = best_clf.predict_proba(class_x_val)[:, 1]
accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, y_pred_proba)
dca = 2 * (auc - 0.5)
precision = precision_score(class_y_val, y_pred)
recall = recall_score(class_y_val, y_pred)
f1 = f1_score(class_y_val, y_pred)
tn, fp, fn, tp = confusion_matrix(class_y_val, y_pred).ravel()
specificity = tn / (tn + fp)
npv = tn / (tn + fn)
ppv = tp / (tp + fp)
fpr = fp / (fp + tn)
print("Accuracy: ", accuracy)
print("AUC: ", auc)
print("DCA: ", dca)
print("Precision: ", precision)
print("Specificity: ", specificity)
print("Sensitivity/Recall: ", recall)
print("Negative Predictive Value: ", npv)
print("Positive Predictive Value: ", ppv)
print("F1 Score: ", f1)
print("False Positive Rate: ", fpr)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' ROC CurvILLNM Bayesian-modele')
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(class_y_val, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ILLNM Bayesian-model Precision-Recall Curve')
plt.show()
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
y_pred = clf.predict(class_x_val)
y_pred_proba = clf.predict_proba(class_x_val)[:, 1]
accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, y_pred_proba)
dca = 2 * (auc - 0.5)
precision = precision_score(class_y_val, y_pred)
recall = recall_score(class_y_val, y_pred)
f1 = f1_score(class_y_val, y_pred)
tn, fp, fn, tp = confusion_matrix(class_y_val, y_pred).ravel()
specificity = tn / (tn + fp)
npv = tn / (tn + fn)
ppv = tp / (tp + fp)
fpr = fp / (fp + tn)

print("Accuracy: ", accuracy)
print("AUC: ", auc)
print("DCA: ", dca)
print("Precision: ", precision)
print("Specificity: ", specificity)
print("Sensitivity/Recall: ", recall)
print("Negative Predictive Value: ", npv)
print("Positive Predictive Value: ", ppv)
print("F1 Score: ", f1)
print("False Positive Rate: ", fpr)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(class_y_val, y_pred_proba)
auc = roc_auc_score(class_y_val, y_pred_proba)
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM GNB ROC Curve')
plt.legend(loc="lower right")
plt.text(0.6, 0.2, 'AUC = %0.2f' % auc)
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(class_y_val, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ILLNM GMN Precision-Recall Curve')
plt.show()
# 5.8 Neural Network Model
### 5.8.3 Model Fitting
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
clf=MLPClassifier(hidden_layer_sizes=(10,50),activation='relu',solver='adam',
alpha=0.0001,batch_size='auto',learning_rate='constant',
learning_rate_init=0.001)
clf.fit(class_x_tra,class_y_tra)
clf.score(class_x_val,class_y_val)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {
'hidden_layer_sizes': [(10,), (20,), (30,)],
'activation': ['relu', 'tanh'],
'solver': ['adam', 'sgd'],
'alpha': [0.0001, 0.001, 0.01],
'learning_rate': ['constant', 'adaptive'],
'learning_rate_init': [0.001, 0.01, 0.1]
}
clf = MLPClassifier()
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(class_x_tra, class_y_tra)
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)
best_clf = grid_search.best_estimator_
best_clf.fit(class_x_tra, class_y_tra)
score = best_clf.score(class_x_val, class_y_val)
print("Validation Score: ", score)
### 5.8.4 Feature Variable Histogram
#### 5.8.4.1 Without Importance Values
import matplotlib.pyplot as plt
best_clf.fit(class_x_tra, class_y_tra)
weights = best_clf.coefs_
feature_importance = np.sum(np.abs(weights[0]), axis=1)
all_indices = range(len(feature_importance))
all_importance = feature_importance[all_indices]
all_features = [feature_name[i] for i in all_indices]
sorted_indices = sorted(all_indices, key=lambda i: all_importance[i], reverse=True)
sorted_importance = [all_importance[i] for i in sorted_indices]
sorted_features = [all_features[i] for i in sorted_indices]
for feature, importance in zip(sorted_features, sorted_importance):
print(f"Variable: {feature}, Importance: {importance}")
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_features)), sorted_importance, tick_label=sorted_features, color='pink')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances - Neural Network')
plt.gca().invert_yaxis()
plt.show()
#### 5.8.4.2 With Importance Values at the End
import matplotlib.pyplot as plt
best_clf.fit(class_x_tra, class_y_tra)
weights = best_clf.coefs_
feature_importance = np.sum(np.abs(weights[0]), axis=1)
all_indices = range(len(feature_importance))
all_importance = feature_importance[all_indices]
all_features = [feature_name[i] for i in all_indices]
sorted_indices = sorted(all_indices, key=lambda i: all_importance[i], reverse=True)
sorted_importance = [all_importance[i] for i in sorted_indices]
sorted_features = [all_features[i] for i in sorted_indices]
for feature, importance in zip(sorted_features, sorted_importance):
print(f"Variable: {feature}, Importance: {importance}")
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(sorted_features)), sorted_importance, tick_label=sorted_features, color='pink')
for i, bar in enumerate(bars):
importance = sorted_importance[i]
plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{importance:.2f}', va='center')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances - Neural Network')
plt.gca().invert_yaxis()
plt.show()
#### 5.8.4.3 Top Ten Ranked
import matplotlib.pyplot as plt
best_clf.fit(class_x_tra, class_y_tra)
weights = best_clf.coefs_
feature_importance = np.sum(np.abs(weights[0]), axis=1)
top_indices = np.argsort(feature_importance)[::-1][:10]
top_importance = feature_importance[top_indices]
top_features = [feature_name[i] for i in top_indices]
for feature, importance in zip(top_features, top_importance):
print(f"Variable: {feature}, Importance: {importance}")
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(top_features)), top_importance, tick_label=top_features, color='pink')
for i, bar in enumerate(bars):
importance = top_importance[i]
plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{importance:.2f}', va='center')

plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Top 10 Feature Importances - Neural Network')
plt.gca().invert_yaxis()
plt.show()
####5.8.4.4 Importance Ratio rankings
import matplotlib.pyplot as plt
best_clf.fit(class_x_tra, class_y_tra)
weights = best_clf.coefs_
feature_importance = np.sum(np.abs(weights[0]), axis=1)
total_importance = np.sum(feature_importance)
importance_ratio = feature_importance / total_importance
all_indices = range(len(importance_ratio))
all_ratio = importance_ratio[all_indices]
all_features = [feature_name[i] for i in all_indices]
sorted_indices = sorted(all_indices, key=lambda i: all_ratio[i], reverse=True)
sorted_ratio = [all_ratio[i] for i in sorted_indices]
sorted_features = [all_features[i] for i in sorted_indices]

for feature, ratio in zip(sorted_features, sorted_ratio):
print(f"Variable: {feature}, Ratio: {ratio}")
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(sorted_features)), sorted_ratio, tick_label=sorted_features, color='pink')
for i, bar in enumerate(bars):
ratio = sorted_ratio[i]
plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{ratio:.2}', va='center')

plt.xlabel('Importance Ratio')
plt.ylabel('Features')
plt.title('Feature Importance Ratios - Neural Network')
plt.gca().invert_yaxis()
plt.show()
###5.8.4 Evaluation indicators
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
y_pred = best_clf.predict(class_x_val)
accuracy = accuracy_score(class_y_val, y_pred)
auc = roc_auc_score(class_y_val, best_clf.predict_proba(class_x_val)[:, 1])
tn, fp, fn, tp = confusion_matrix(class_y_val, y_pred).ravel()
precision = tp / (tp + fp)
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
npv = tn / (tn + fn)
ppv = tp / (tp + fp)
recall = sensitivity
f1_score = 2 * (precision * recall) / (precision + recall)
fpr = fp / (fp + tn)
print("Accuracy: ", accuracy)
print("AUC: ", auc)
print("Precision: ", precision)
print("Specificity: ", specificity)
print("Sensitivity: ", sensitivity)
print("Negative Predictive Value: ", npv)
print("Positive Predictive Value: ", ppv)
print("Recall: ", recall)
print("F1 Score: ", f1_score)
print("False Positive Rate: ", fpr)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(class_y_val, best_clf.predict_proba(class_x_val)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.fill_between(fpr, tpr, alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ILLNM MLP ROC Curve')
plt.legend(loc='lower right')
plt.show()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(class_y_val, best_clf.predict_proba(class_x_val)[:, 1])
smooth_precision = np.linspace(0, 1, 100)
smooth_recall = np.interp(smooth_precision, precision, recall)
plt.figure(figsize=(8, 6))
plt.plot(smooth_recall, smooth_precision, color='b', lw=2, label='Precision-Recall Curve')
plt.fill_between(smooth_recall, smooth_precision, alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('ILLNM MLP Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()
import numpy as np

weights = best_clf.coefs_
feature_contributions = np.abs(weights).mean(axis=0)
feature_contributions_dict = dict(zip(data.columns, feature_contributions))
sorted_contributions = sorted(feature_contributions_dict.items(), key=lambda x: x[1], reverse=True)
for feature, contribution in sorted_contributions:
print(f"{feature}: {contribution}")
import matplotlib.pyplot as plt
features = [feature for feature, _ in sorted_contributions]
contributions = [contribution for _, contribution in sorted_contributions]
plt.figure(figsize=(10, 6))
plt.bar(features, contributions)
plt.xticks(rotation=90)
plt.xlabel('Feature Variables')
plt.ylabel('Contribution')
plt.title('Feature Variable Contributions')
plt.tight_layout()
plt.show()
### 5.8.6 Plotting Network Calculator
# Obtaining Feature Names and Weights based on Feature Importance Ranking
feature_names = ['feature1', 'feature2', 'feature3']
feature_weights = [0.4, 0.3, 0.3]

def network_calculator(input_features):
prediction = 0
for feature, weight in zip(input_features, feature_weights):
prediction += feature * weight
return prediction
user_input = [2, 3, 4]
prediction = network_calculator(user_input)
print("Prediction:", prediction)
from flask import Flask, render_template, request
app = Flask(__name__)

feature_names = ['ETE', 'Size', 'ICLNM']
feature_weights = [0.4, 0.3, 0.3]

@app.route('/')
def home():
return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
input_features = [float(request.form.get('ETE')),
float(request.form.get('Size')),
float(request.form.get('ICLNM'))]
prediction = 0
for feature, weight in zip(input_features, feature_weights):
prediction += feature * weight
return render_template('result.html', prediction=prediction)
if __name__ == '__main__':
app.run(debug=True)
# 6. Overall Evaluation Code
## 6.1 Plotting Overall ROC Curve
### 6.1.1 Method 1: Pay attention to parameter settings! Especially for neural networks and random forests, use the code with the parameter settings learned earlier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
data = pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据删除一些不必要的变量的数据.csv')
data_feature = data[['Age','Tumor.border', 'Aspect.ratio', 
'Internal.echo.homogeneous','Calcification', 
'Tumor.internal.vascularization', 'Size','Location',
'Side.of.position',
'ICLNM','Classification.of.ICLNMR',
'pretracheal.LNM','Classification.of.pretracheal.LNMR',
'paratracheal.LNM','Classification.of.paratracheal.LNMR',
'RLNLNM','prelaryngeal.NLNM', 'pretracheal.NLNM','paratracheal.NLNM','RLNNLNM' ]]
data_target=data['ILLNM']
data_target.unique()#二分类
data_featureCata=data[['Age','Tumor.border', 'Aspect.ratio', 'Internal.echo.homogeneous','Calcification', 'Tumor.internal.vascularization', 'Size',
'Side.of.position','Location',
'ICLNM','Classification.of.ICLNMR',
'pretracheal.LNM','Classification.of.pretracheal.LNMR',
'paratracheal.LNM','Classification.of.paratracheal.LNMR',
'RLNLNM',
]]
data_featureNum=data[['ICNLNM', 'prelaryngeal.NLNM', 'pretracheal.NLNM','paratracheal.NLNM','RLNNLNM' ]]
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureCata=np.array(data_featureCata)
import numpy as np
data_feature = np.hstack((data_featureCata, data_featureNum))
data_target = data['ILLNM']
class_x_tra, class_x_val, class_y_tra, class_y_val = train_test_split(data_feature, data_target, test_size=0.3, random_state=0)
models = [
LogisticRegression(),
RidgeClassifier(alpha=1, fit_intercept=True),
DecisionTreeClassifier(min_samples_split=3, max_depth=5, min_samples_leaf=10, max_features=20),
RandomForestClassifier(n_estimators=25000, max_depth=5, min_samples_leaf=5, min_samples_split=5),
GradientBoostingClassifier(n_estimators=25000, max_depth=7, learning_rate=0.01),
SVC(kernel='linear', gamma=0.2, probability=True),
KNeighborsClassifier(n_neighbors=11, leaf_size=40, n_jobs=6),
GaussianNB(),
MLPClassifier()
]
model_names = [
'Logistic Regression',
'Ridge Regression',
'Decision Tree',
'Random Forest',
'Gradient Boosting',
'Support Vector Machine',
'K-Nearest Neighbors',
'Gaussian Naive Bayes',
'Neural Network'
]
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan']
nn_param_grid = {
'hidden_layer_sizes': [(10,), (20,), (30,)],
'activation': ['relu', 'tanh'],
'solver': ['adam', 'sgd'],
'alpha': [0.0001, 0.001, 0.01],
'learning_rate': ['constant', 'adaptive'],
'learning_rate_init': [0.001, 0.01, 0.1]
}
nn_grid_search = GridSearchCV(estimator=MLPClassifier(), 
param_grid=nn_param_grid, cv=5)
plt.figure(figsize=(8, 6))
for model, name, color in zip(models, model_names, colors):
if name == 'Neural Network':
nn_grid_search.fit(class_x_tra, class_y_tra)
best_nn_model = nn_grid_search.best_estimator_
model = best_nn_model
else:
model.fit(class_x_tra, class_y_tra)
if hasattr(model, 'predict_proba'):
y_pred_prob = model.predict_proba(class_x_val)[:, 1]
else:
y_pred_prob = model.decision_function(class_x_val)
auc = roc_auc_score(class_y_val, y_pred_prob)
fpr, tpr, _ = roc_curve(class_y_val, y_pred_prob)
plt.plot(fpr, tpr, color=color, label='%s (AUC = %0.3f)' % (name, auc))

plt.plot([0, 1], [0, 1], 'k--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('IV.level.LNM Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
# Displaying My Results
## 1.1.1 ROC Curve for Training Set
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
data = pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量.csv')
data_feature = data[['Sex','Tumor.border','Aspect.ratio',
'Calcification','Tumor.internal.vascularization',
'ETE','Size','Hashimoto',
'prelaryngeal.LNM','paratracheal.LNM','RLNLNM','ICLNM',
'ICLNMR','pretracheal.NLNM',
'paratracheal.LNMR',
'RLNLNMR',
]]
data_target=data['IV.level.LNM']
data_target.unique()
data_featureCata=data[['Sex','Tumor.border','Aspect.ratio',
'Calcification','Tumor.internal.vascularization',
'ETE','Size','Hashimoto',
'prelaryngeal.LNM','paratracheal.LNM','RLNLNM','ICLNM',]]


data_featureNum=data[['ICLNMR','pretracheal.NLNM',
'paratracheal.LNMR',
'RLNLNMR',]]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data_featureNum=scaler.fit_transform(data_featureNum)
data_featureCata=np.array(data_featureCata)
import numpy as np

data_feature = np.hstack((data_featureCata, data_featureNum))
data_target = data['IV.level.LNM']
class_x_tra, class_x_val, class_y_tra, class_y_val = train_test_split(data_feature, data_target, test_size=0.3, random_state=0)

models = [
LogisticRegression(),
DecisionTreeClassifier(min_samples_split=3, max_depth=5, min_samples_leaf=10, max_features=20),
RandomForestClassifier(n_estimators=25000, max_depth=5, min_samples_leaf=5, min_samples_split=20),
GradientBoostingClassifier(n_estimators=5, max_depth=4, learning_rate=0.1),
SVC(kernel='linear', gamma=0.2, probability=True),
KNeighborsClassifier(n_neighbors=13),
GaussianNB(),
MLPClassifier()
]
model_names = [
'Logistic Regression',
'Decision Tree',
'Random Forest',
'Gradient Boosting',
'Support Vector Machine',
'K-Nearest Neighbors',
'Gaussian Naive Bayes',
'Neural Network'
]
colors = ['blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'cyan']
nn_param_grid = {
'hidden_layer_sizes': [(10,), (20,), (30,)],
'activation': ['relu', 'tanh'],
'solver': ['adam', 'sgd'],
'alpha': [0.0001, 0.001, 0.01],
'learning_rate': ['constant', 'adaptive'],
'learning_rate_init': [0.001, 0.01, 0.1]
}
nn_grid_search = GridSearchCV(estimator=MLPClassifier(), 
param_grid=nn_param_grid, cv=10)
plt.figure(figsize=(8, 6))
for model, name, color in zip(models, model_names, colors):
if name == 'Neural Network':
nn_grid_search.fit(class_x_tra, class_y_tra)
best_nn_model = nn_grid_search.best_estimator_
model = best_nn_model
else:
model.fit(class_x_tra, class_y_tra)
if hasattr(model, 'predict_proba'):
y_train_pred_prob = model.predict_proba(class_x_tra)[:, 1]
else:
y_train_pred_prob = model.decision_function(class_x_tra)
auc = roc_auc_score(class_y_tra, y_train_pred_prob)
fpr, tpr, _ = roc_curve(class_y_tra, y_train_pred_prob)
plt.plot(fpr, tpr, color=color, label='%s (AUC = %0.3f)' % (name, auc))
plt.grid(True)
plt.plot([0, 1], [0, 1], 'k--') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('IV.level.LNM ROC (Training set)')
plt.legend(loc='lower right')
plt.show()
##1.1.2 Evaluation index of training set
train_accuracy_scores = []
train_auc_scores = []
train_precision_scores = []
train_specificity_scores = []
train_sensitivity_scores = []
train_npv_scores = []
train_ppv_scores = []
train_recall_scores = []
train_f1_scores = []
train_fpr_scores = []
for model, name in zip(models, model_names):
if name == 'Neural Network':
nn_grid_search.fit(class_x_tra, class_y_tra)
best_nn_model = nn_grid_search.best_estimator_
model = best_nn_model
else:
model.fit(class_x_tra, class_y_tra)
train_y_pred = model.predict(class_x_tra)
if hasattr(model, 'predict_proba'):
train_y_pred_prob = model.predict_proba(class_x_tra)[:, 1]
else:
train_y_pred_prob = model.decision_function(class_x_tra)
train_accuracy = accuracy_score(class_y_tra, train_y_pred)
train_auc = roc_auc_score(class_y_tra, train_y_pred_prob)
train_precision = precision_score(class_y_tra, train_y_pred)
train_cm = confusion_matrix(class_y_tra, train_y_pred)
train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
train_specificity = train_tn / (train_tn + train_fp)
train_sensitivity = recall_score(class_y_tra, train_y_pred)
train_npv = train_tn / (train_tn + train_fn)
train_ppv = train_tp / (train_tp + train_fp)
train_recall = train_sensitivity
train_f1 = f1_score(class_y_tra, train_y_pred)
train_fpr = train_fp / (train_fp + train_tn)
train_accuracy_scores.append(train_accuracy)
train_auc_scores.append(train_auc)
train_precision_scores.append(train_precision)
train_specificity_scores.append(train_specificity)
train_sensitivity_scores.append(train_sensitivity)
train_npv_scores.append(train_npv)
train_ppv_scores.append(train_ppv)
train_recall_scores.append(train_recall)
train_f1_scores.append(train_f1)
train_fpr_scores.append(train_fpr)
train_metrics_df = pd.DataFrame({
'Model': model_names,
'Accuracy': train_accuracy_scores,
'AUC': train_auc_scores,
'Precision': train_precision_scores,
'Specificity': train_specificity_scores,
'Sensitivity': train_sensitivity_scores,
'Negative Predictive Value': train_npv_scores,
'Positive Predictive Value': train_ppv_scores,
'Recall': train_recall_scores,
'F1 Score': train_f1_scores,
'False Positive Rate': train_fpr_scores
})
print(train_metrics_df)
train_metrics_df.to_csv('/Users/zj/Desktop/4.机器学习/结果/4.评价指标表格/4.1 IV.level.LNM训练集的评价指标.csv', index=False)
##1.1.3 DCA curve of training set
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
thresholds = np.linspace(0, 1, 100)
train_net_benefit = []
for model, model_name, color in zip(models, model_names, colors):
model.fit(class_x_tra, class_y_tra)
train_model_predictions = model.predict_proba(class_x_tra)[:, 1]
train_model_net_benefit = []
for threshold in thresholds:
train_predictions = (train_model_predictions >= threshold).astype(int)
train_net_benefit_value = (precision_score(class_y_tra, train_predictions) - threshold * (1 - precision_score(class_y_tra, train_predictions))) / (threshold + 1e-10)
train_model_net_benefit.append(train_net_benefit_value)
train_net_benefit.append(train_model_net_benefit)
train_net_benefit = np.array(train_net_benefit)
train_all_predictions = np.ones_like(class_y_tra) train_all_net_benefit = (precision_score(class_y_tra, train_all_predictions) - thresholds * (1 - precision_score(class_y_tra, train_all_predictions))) / (thresholds + 1e-10)
for i in range(train_net_benefit.shape[0]):
plt.plot(thresholds, train_net_benefit[i], color=colors[i], label=model_names[i])

plt.plot(thresholds, np.zeros_like(thresholds), color='black', linestyle='-', label='None')
plt.plot(thresholds, train_all_net_benefit, color='gray', linestyle='--', label='All')
plt.xlim(0, 0.8)
plt.ylim(-0.5,6)
plt.xlabel('Threshold')
plt.ylabel('Net Benefit')
plt.title('IV.level.LNM Decision Curve Analysis (Training set)')
plt.legend(loc='upper right')
plt.show()
1.1.4 Calibration curve of the training set
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy.stats import ttest_ind
train_calibration_curves = []
train_brier_scores = []
for model, model_name, color in zip(models, model_names, colors):
model.fit(class_x_tra, class_y_tra)
y_proba = model.predict_proba(class_x_tra)[:, 1]
train_fraction_of_positives, train_mean_predicted_value = calibration_curve(class_y_tra, y_proba, n_bins=10)
train_calibration_curves.append((train_fraction_of_positives, train_mean_predicted_value, model_name, color))
train_brier_score = brier_score_loss(class_y_tra, y_proba)
train_brier_scores.append((model_name, train_brier_score))
print(f'{model_name} - Training Brier Score: {train_brier_score:.4f}')
fig, ax1 = plt.subplots(figsize=(10, 6))

for curve in train_calibration_curves:
train_fraction_of_positives, train_mean_predicted_value, model_name, color = curve
train_brier_score = next((score for model, score in train_brier_scores if model == model_name), None)
if train_brier_score is not None:
model_name += f' (Training Brier Score: {train_brier_score:.4f})'
ax1.plot(train_mean_predicted_value, train_fraction_of_positives, "s-", label=model_name, color=color)
ax1.plot([0, 1], [0, 1], "k:",label="Perfectly calibrated")
ax1.set_ylabel("Fraction of positives")
ax1.set_xlabel("Mean predicted value")
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")

plt.title("IV.level.LNM Calibration Curves (Training set)")
plt.tight_layout()
plt.show()
##1.1.5 Exact recall curve for training set
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
def plot_pr_curve(y_true, y_prob, model_name, color):
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
aupr = auc(recall, precision)

plt.plot(recall, precision, color=color, label=model_name + f' (AUPR = {aupr:.3f})')
train_y_true_list = [class_y_tra] * len(models)
train_y_prob_list = [model.fit(class_x_tra, class_y_tra).predict_proba(class_x_tra)[:, 1] for model in models]
plt.figure(figsize=(10, 8))
for i, (train_y_true, train_y_prob, model_name, color) in enumerate(zip(train_y_true_list, train_y_prob_list, model_names, colors)):
plot_pr_curve(train_y_true, train_y_prob, model_name, color)
plt.plot([0, 1], [class_y_tra.mean(), class_y_tra.mean()], linestyle='--', color='black', label='Random Guessing')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('IV.level.LNM Precision-Recall Curve (Training set)')
plt.legend()
plt.show()
# 7. Important Feature Ranking
## 7.1 Feature Ranking for Each Model
import matplotlib.pyplot as plt
fig, axs = plt.subplots(len(models), 1, figsize=(10, 6 * len(models)))
for i, (model, model_name, color) in enumerate(zip(models, model_names, colors)):
model.fit(class_x_tra, class_y_tra)
if hasattr(model, 'coef_'):
feature_importance = np.abs(model.coef_[0])
else:
feature_importance = np.zeros(len(data.columns))
sorted_idx = np.argsort(feature_importance)
axs[i].barh(range(len(sorted_idx)), feature_importance[sorted_idx], color=color)
axs[i].set_yticks(range(len(sorted_idx)))
axs[i].set_yticklabels(np.array(data.columns)[sorted_idx])
axs[i].set_xlabel('Relative Importance')
axs[i].set_title(f'{model_name} - Feature Importances')
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
for i, (model, model_name, color) in enumerate(zip(models, model_names, colors)):
fig, ax = plt.subplots(figsize=(10, 6))
model.fit(class_x_tra, class_y_tra)
if hasattr(model, 'coef_'):
feature_importance = np.abs(model.coef_[0])
else:
feature_importance = np.zeros(len(data.columns))
sorted_idx = np.argsort(feature_importance)
ax.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color=color)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(np.array(data.columns)[sorted_idx])
ax.set_xlabel('Relative Importance')
ax.set_title(f'{model_name} - Feature Importances')
plt.show()
import matplotlib.pyplot as plt
fig, axs = plt.subplots(len(models), 1, figsize=(10, 6 * len(models)))
for i, (model, model_name, color) in enumerate(zip(models, model_names, colors)):
model.fit(class_x_tra, class_y_tra)
if hasattr(model, 'coef_'):
feature_importance = np.abs(model.coef_[0])
else:
feature_importance = np.zeros(len(data.columns))
sorted_idx = np.argsort(feature_importance)
axs[i].barh(range(len(sorted_idx)), feature_importance[sorted_idx], color=color)
axs[i].set_yticks(range(len(sorted_idx)))
axs[i].set_yticklabels(np.array(data.columns)[sorted_idx])
axs[i].set_xlabel('Relative Importance')
axs[i].set_title(f'{model_name} - Feature Importances')
axs[i].set_xlim(0, 2) # 设置x轴的取值范围为0-4
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

# Set subplot layout
fig, axs = plt.subplots(len(models), 1, figsize=(10, 6 * len(models)))

# Iterate over each model
for i, (model, model_name, color) in enumerate(zip(models, model_names, colors)):
# Train the model
model.fit(class_x_tra, class_y_tra)
# Calculate feature importance
if hasattr(model, 'coef_'):
feature_importance = np.abs(model.coef_[0])
else:
feature_importance = np.zeros(len(data.columns))
# Sort feature importance
sorted_idx = np.argsort(feature_importance)
# Plot bar chart
axs[i].barh(range(len(sorted_idx)), feature_importance[sorted_idx], color=color)
axs[i].set_yticks(range(len(sorted_idx)))
axs[i].set_yticklabels(np.array(data.columns)[sorted_idx])
axs[i].set_xlabel('Relative Importance')
axs[i].set_title(f'{model_name} - Feature Importances')

# Adjust subplot layout
plt.tight_layout()
plt.show()
Network calculator drawing
from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
top_features = ['ICLNMR', 'ETE', 'Tumor.border', 'ICNLNM', 'Tumor.internal.vascularization',
'Calcification', 'ICLNM', 'paratracheal.LNMR', 'Size', 'paratracheal.NLNM']
top_importances = [0.14, 0.12, 0.12, 0.1, 0.08, 0.08, 0.07, 0.06, 0.05, 0.02]

@app.route('/')
def index():
return render_template('calculator.html', features=top_features, importances=top_importances)

@app.route('/calculate', methods=['POST'])
def calculate_result():
feature_values = [request.form[feature] for feature in top_features]
data = pd.read_csv('/Users/zj/Desktop/4.机器学习/数据分析/PYTHON数据/矫正后的数据-纯数值变量.csv')
data_feature = data[top_features]
data_target = data['ILLNM']
model = RandomForestClassifier(n_estimators=25000, max_depth=5, min_samples_leaf=5, min_samples_split=5)
model.fit(data_feature, data_target)
result = model.predict([feature_values])
return result
if __name__ == '__main__':
app.run()


