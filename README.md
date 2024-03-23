# ML-for-ILLNM
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



