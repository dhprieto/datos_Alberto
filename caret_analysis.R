library(tidyverse)
library(readxl)
library(caret)

## Import the data

excel_sheets("./01-literature-review/growth/Datos estudio mod prob S. aureus.xlsx")

d1 <- read_excel("./01-literature-review/growth/Datos estudio mod prob S. aureus.xlsx",
                 sheet = "Training") %>%
  select(temp = Temperatura...1,
         pH = pH...2,
         aw = Aw...3,
         bw = Bw,
         Crec)

d2 <- read_excel("./01-literature-review/growth/Datos estudio mod prob S. aureus.xlsx",
                 sheet = "Validation") %>%
  select(temp = Temperatura,
         pH,
         aw = Aw,
         bw = Bw,
         Crec)

d <- bind_rows(d1, d2) %>% mutate(Crec = factor(Crec))
d

## Training and test sets

in_training <- createDataPartition(d$Crec, p = .8, list = FALSE)

d_train <- d[ in_training,]
d_test  <- d[-in_training,]

## Training

fitControl <- trainControl(
  method = "cv",
  number = 10,
  search = "grid")

set.seed(825)
gbmFit1 <- train(Crec ~ ., data = d_train,
                 method = "gbm",
                 trControl = fitControl)

plot(gbmFit1)

## Comparison

d_train %>%
  mutate(p = gbmFit1 %>% predict()) %>% View()

d_test %>%
  mutate(p = predict(gbmFit1, newdata = .)) %>%
  select(Crec, p) %>% table()


d_test %>%
  mutate(p = predict(gbmFit1, newdata = ., type = "prob")$`51`) %>%
  ggplot() +
  geom_boxplot(aes(x = Crec, y = p))

d_train %>%
  mutate(p = predict(gbmFit1, newdata = ., type = "prob")$`1`) %>%
  ggplot() +
  geom_boxplot(aes(x = Crec, y = p))

## Training an SVM

svmFit <- train(Crec ~ ., data = d_train,
                method = "svmRadial",
                trControl = trainControl(
                  method = "cv",
                  number = 10,
                  search = "grid"),
                preProc = c("center", "scale"),
                tuneLength = 8)

svmFit
plot(svmFit)

svmFit2 <- train(Crec ~ ., data = d_train,
                method = "svmPoly",
                trControl = trainControl(
                  method = "cv",
                  number = 10,
                  search = "grid"),
                preProc = c("center", "scale"),
                tuneLength = 8)

svmFit2
plot(svmFit2)


## Comparison

d_test %>%
  mutate(p = predict(svmFit, newdata = .)) %>%
  select(Crec, p) %>% table()

predict(svmFit, newdata = d_test, type = "prob")

d_test %>%
  mutate(p = predict(svmFit, newdata = ., type = "prob")) %>%
  ggplot() +
  geom_boxplot(aes(x = Crec, y = p))

d_train %>%
  mutate(p = predict(gbmFit1, newdata = ., type = "prob")$`1`) %>%
  ggplot() +
  geom_boxplot(aes(x = Crec, y = p))

# ##
#
# mylogit <- glm(Crec ~ ., data = d_train, family = "binomial")
# mylogit

