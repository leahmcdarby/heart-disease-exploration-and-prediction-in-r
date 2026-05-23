## Heart disease exploration and prediction in R
#Leah McDarby
#Dataset: UCI Heart Disease Cleveland dataset
#Aim: explore clinical variables that associate with heart disease and build a logistic regression model for interpretation

# 1. load packafes
library(ggplot2)

# 2. Load data
url <- "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

heart <- read.csv(url, header = FALSE, na.strings = "?")

colnames(heart) <- c(
  "age", "sex", "chest_pain", "resting_bp", "cholesterol",
  "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
  "exercise_angina", "oldpeak", "slope", "major_vessels",
  "thal", "diagnosis"
)

# 3. Create ouputs folder
if (!dir.exists("outputs")) {
  dir.create("outputs")
}

# 4. Check missing data
missing_table <- data.frame(
  variable = names(heart),
  missing_count = colSums(is.na(heart)),
  missing_percent = round(colSums(is.na(heart)) / nrow(heart) * 100, 2)
)

missing_table

write.csv(
  missing_table,
  "outputs/missing_data_summary.csv",
  row.names = FALSE
)

# 5. Clean data

# Remove rows with missing values
heart <- na.omit(heart)

# Create binary outcome for modelling:
# 0 = no heart disease
# 1 = heart disease present
heart$disease_binary <- ifelse(heart$diagnosis == 0, 0, 1)

# Create label version for plots
heart$heart_disease <- ifelse(heart$disease_binary == 1, "Disease", "No disease")

# Convert categorical variables to factors
heart$sex <- factor(heart$sex, levels = c(0, 1), labels = c("Female", "Male"))
heart$chest_pain <- factor(heart$chest_pain)
heart$fasting_blood_sugar <- factor(heart$fasting_blood_sugar)
heart$resting_ecg <- factor(heart$resting_ecg)
heart$exercise_angina <- factor(heart$exercise_angina)
heart$slope <- factor(heart$slope)
heart$thal <- factor(heart$thal)

# Check cleaned dataset
nrow(heart)
table(heart$heart_disease)

# 6. Exploratory summaries
# Outcome balance
outcome_summary <- as.data.frame(table(heart$heart_disease))
colnames(outcome_summary) <- c("heart_disease", "count")

outcome_summary$percent <- round(
  outcome_summary$count / sum(outcome_summary$count) * 100,
  1
)

outcome_summary

write.csv(
  outcome_summary,
  "outputs/outcome_summary.csv",
  row.names = FALSE
)
# Numerical summaries by disease status
numeric_summary <- aggregate(
  cbind(age, resting_bp, cholesterol, max_heart_rate, oldpeak) ~ heart_disease,
  data = heart,
  FUN = mean
)

numeric_summary

write.csv(
  numeric_summary,
  "outputs/numeric_summary_by_disease.csv",
  row.names = FALSE
)

# 7. Exploratory visualisation
png("outputs/age_histogram.png", width = 800, height = 600)

# Age distribution
hist(
  heart$age,
  breaks = 30,
  main = "Age distribution",
  xlab = "Age"
)

dev.off()

# Outcome counts
ggplot(heart, aes(x = heart_disease)) +
  geom_bar() +
  labs(
    title = "Heart disease outcome counts",
    x = "Heart disease status",
    y = "Count"
  )

ggsave("outputs/outcome_counts.png", width = 6, height = 4)


# Age by diagnosis level
ggplot(heart, aes(x = factor(diagnosis), y = age)) +
  geom_boxplot() +
  labs(
    title = "Age by diagnosis level",
    x = "Diagnosis level",
    y = "Age"
  )

ggsave("outputs/age_by_diagnosis_level.png", width = 6, height = 4)


# Maximum heart rate by heart disease status
ggplot(heart, aes(x = heart_disease, y = max_heart_rate)) +
  geom_boxplot() +
  labs(
    title = "Maximum heart rate by heart disease status",
    x = "Heart disease status",
    y = "Maximum heart rate"
  )

ggsave("outputs/max_heart_rate_by_disease.png", width = 6, height = 4)


# Oldpeak by heart disease status
ggplot(heart, aes(x = heart_disease, y = oldpeak)) +
  geom_boxplot() +
  labs(
    title = "Oldpeak by heart disease status",
    x = "Heart disease status",
    y = "Oldpeak"
  )

ggsave("outputs/oldpeak_by_disease.png", width = 6, height = 4)


# Exercise-induced angina by heart disease status
ggplot(heart, aes(x = exercise_angina, fill = heart_disease)) +
  geom_bar(position = "fill") +
  labs(
    title = "Heart disease proportion by exercise-induced angina",
    x = "Exercise-induced angina",
    y = "Proportion",
    fill = "Heart disease status"
  )

ggsave("outputs/exercise_angina_by_disease.png", width = 6, height = 4)
# Spearman correlation between age and diagnosis level
age_diagnosis_test <- cor.test(
  heart$age,
  heart$diagnosis,
  method = "spearman"
)

age_diagnosis_test

# 8. Feature choice 

# Based on exploratory analysis and clinical relevance, the model uses:
# age, sex, chest_pain, max_heart_rate, exercise_angina, oldpeak,
# major_vessels and thal.
#
# The original diagnosis variable is not included in the model because it was
# used to define the binary disease outcome.

# 9. Train/test split of the data
set.seed(123)

n <- nrow(heart)
train_rows <- sample(1:n, size = round(0.7 * n))

train_data <- heart[train_rows, ]
test_data <- heart[-train_rows, ]

# 10. Logistic regression model
log_model <- glm(
  disease_binary ~ age + sex + chest_pain + max_heart_rate +
    exercise_angina + oldpeak + major_vessels + thal,
  data = train_data,
  family = binomial
)

summary(log_model)

# 11. Prediction on test data
# Predict probability of heart disease
test_prob <- predict(log_model, newdata = test_data, type = "response")

# Convert probabilities into 0/1 predictions
test_pred <- ifelse(test_prob >= 0.5, 1, 0)

# 12. Model evaluation
conf_matrix <- table(
  Predicted = test_pred,
  Actual = test_data$disease_binary
)

conf_matrix

accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

sensitivity <- conf_matrix["1", "1"] / sum(conf_matrix[, "1"])
specificity <- conf_matrix["0", "0"] / sum(conf_matrix[, "0"])

accuracy
sensitivity
specificity

model_performance <- data.frame(
  accuracy = accuracy,
  sensitivity = sensitivity,
  specificity = specificity
)

model_performance

# 13. save model outputs
write.csv(conf_matrix, "outputs/confusion_matrix.csv")

write.csv(
  model_performance,
  "outputs/model_performance.csv",
  row.names = FALSE
)

sink("outputs/model_summary.txt")

cat("Logistic regression model summary\n\n")
print(summary(log_model))

cat("\nConfusion matrix\n\n")
print(conf_matrix)

cat("\nModel performance\n\n")
print(model_performance)

cat("\nSpearman correlation between age and diagnosis level\n\n")
print(age_diagnosis_test)

cat("\nInterpretation note\n\n")
cat("The model was evaluated on a held-out test set. Performance should be interpreted cautiously because this is a small public dataset from a single source. Future work could use repeated cross-validation and external validation for a more robust estimate of performance.\n")

sink()
