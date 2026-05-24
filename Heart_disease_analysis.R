## Heart disease exploration and prediction in R
# Leah McDarby
# Dataset: UCI Heart Disease Cleveland dataset
# Aim: explore clinical variables associated with heart disease and build a simple logistic regression model


# 1. Load package 

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


# 3. Create outputs folder 

if (!dir.exists("outputs")) {
  dir.create("outputs")
}


# 4. Check missing data -

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

# Mean numerical values by heart disease status
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


# 7. Exploratory visualisation -

# Age distribution
png("outputs/age_histogram.png", width = 800, height = 600)

hist(
  heart$age,
  breaks = 30,
  main = "Age distribution",
  xlab = "Age"
)

dev.off()


# Diagnosis level counts
ggplot(heart, aes(x = factor(diagnosis))) +
  geom_bar() +
  labs(
    title = "Number of patients by diagnosis level",
    x = "Diagnosis level",
    y = "Count"
  )

ggsave("outputs/diagnosis_level_counts.png", width = 6, height = 4)


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


# 8. PCA exploration 

# Select numerical variables
pca_data <- heart[, c("age", "resting_bp", "cholesterol", "max_heart_rate", "oldpeak")]

# Scale variables before PCA
pca_scaled <- scale(pca_data)

# Run PCA
pca_model <- prcomp(pca_scaled)

# Create dataframe for plotting
pca_plot_data <- data.frame(
  PC1 = pca_model$x[, 1],
  PC2 = pca_model$x[, 2],
  heart_disease = heart$heart_disease
)

# PCA plot coloured by disease status
ggplot(pca_plot_data, aes(x = PC1, y = PC2, colour = heart_disease)) +
  geom_point() +
  labs(
    title = "PCA of numerical clinical variables",
    x = "Principal component 1",
    y = "Principal component 2",
    colour = "Heart disease status"
  )

ggsave("outputs/pca_disease_plot.png", width = 6, height = 4)


# 9. K-means clustering exploration 

# K-means is used as an exploratory method to assess whether numerical
# variables form broad groups. These clusters are not treated as clinical groups.

set.seed(123)

km_model <- kmeans(
  pca_scaled,
  centers = 2,
  nstart = 20
)

# Add cluster labels
heart$cluster <- factor(km_model$cluster)

# Compare clusters with heart disease status
cluster_summary <- table(
  Cluster = heart$cluster,
  Heart_disease = heart$heart_disease
)

cluster_summary

write.csv(
  cluster_summary,
  "outputs/kmeans_cluster_summary.csv"
)

# Add cluster labels to PCA dataframe
pca_plot_data$cluster <- heart$cluster

# PCA plot coloured by k-means cluster and shaped by disease status
ggplot(pca_plot_data, aes(x = PC1, y = PC2, colour = cluster, shape = heart_disease)) +
  geom_point() +
  labs(
    title = "K-means clusters shown using PCA",
    x = "Principal component 1",
    y = "Principal component 2",
    colour = "K-means cluster",
    shape = "Heart disease status"
  )

ggsave("outputs/kmeans_pca_plot.png", width = 6, height = 4)


# 10. Investigate cluster characteristics 

# Mean numerical variables by cluster
cluster_numeric_summary <- aggregate(
  cbind(age, resting_bp, cholesterol, max_heart_rate, oldpeak) ~ cluster,
  data = heart,
  FUN = mean
)

cluster_numeric_summary

write.csv(
  cluster_numeric_summary,
  "outputs/cluster_numeric_summary.csv",
  row.names = FALSE
)

# Heart disease status by cluster
cluster_disease_summary <- as.data.frame(
  table(heart$cluster, heart$heart_disease)
)

colnames(cluster_disease_summary) <- c(
  "cluster",
  "heart_disease",
  "count"
)

cluster_disease_summary

write.csv(
  cluster_disease_summary,
  "outputs/cluster_disease_summary.csv",
  row.names = FALSE
)

# Diagnosis level composition by cluster
diagnosis_by_cluster <- table(
  Cluster = heart$cluster,
  Diagnosis = heart$diagnosis
)

diagnosis_by_cluster

diagnosis_by_cluster_prop <- prop.table(diagnosis_by_cluster, margin = 1)

round(diagnosis_by_cluster_prop, 3)

diagnosis_by_cluster_df <- as.data.frame(diagnosis_by_cluster_prop)

colnames(diagnosis_by_cluster_df) <- c(
  "cluster",
  "diagnosis_level",
  "proportion"
)

diagnosis_by_cluster_df$proportion <- round(
  diagnosis_by_cluster_df$proportion,
  3
)

write.csv(
  diagnosis_by_cluster_df,
  "outputs/diagnosis_composition_by_cluster.csv",
  row.names = FALSE
)

# Diagnosis composition by cluster plot
ggplot(heart, aes(x = cluster, fill = factor(diagnosis))) +
  geom_bar(position = "fill") +
  labs(
    title = "Diagnosis level composition by k-means cluster",
    x = "K-means cluster",
    y = "Proportion",
    fill = "Diagnosis level"
  )

ggsave("outputs/diagnosis_composition_by_cluster.png", width = 6, height = 4)


# 11. Feature choice 

# Exploratory analysis was used to inspect how clinical variables differed
# between patients with and without heart disease.
#
# The final logistic regression model includes variables that showed visible
# differences in exploratory plots or were clinically relevant:
# age, sex, chest_pain, max_heart_rate, exercise_angina, oldpeak,
# major_vessels and thal.
#
# Resting blood pressure and cholesterol were explored, but were not included
# in the final model to keep it simple and interpretable.
#
# The original diagnosis variable is not included because it was used to define
# the binary disease outcome.


# 12. Train/test split 

set.seed(123)

n <- nrow(heart)
train_rows <- sample(1:n, size = round(0.7 * n))

train_data <- heart[train_rows, ]
test_data <- heart[-train_rows, ]


# 13. Logistic regression model 

log_model <- glm(
  disease_binary ~ age + sex + chest_pain + max_heart_rate +
    exercise_angina + oldpeak + major_vessels + thal,
  data = train_data,
  family = binomial
)

summary(log_model)


# 14. Prediction on test data 

# Predict probability of heart disease
test_prob <- predict(log_model, newdata = test_data, type = "response")

# Convert probabilities into 0/1 predictions
test_pred <- ifelse(test_prob >= 0.5, 1, 0)


# 15. Model evaluation 

conf_matrix <- table(
  Predicted = test_pred,
  Actual = test_data$disease_binary
)

conf_matrix

# Create labelled confusion matrix for saving
conf_matrix_labelled <- as.data.frame(conf_matrix)

colnames(conf_matrix_labelled) <- c(
  "Predicted",
  "Actual",
  "Count"
)

conf_matrix_labelled$Predicted <- ifelse(
  conf_matrix_labelled$Predicted == 0,
  "No disease",
  "Disease"
)

conf_matrix_labelled$Actual <- ifelse(
  conf_matrix_labelled$Actual == 0,
  "No disease",
  "Disease"
)

write.csv(
  conf_matrix_labelled,
  "outputs/confusion_matrix.csv",
  row.names = FALSE
)

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

write.csv(
  model_performance,
  "outputs/model_performance.csv",
  row.names = FALSE
)


# 16. Save model summary 

sink("outputs/model_summary.txt")

cat("Logistic regression model summary\n\n")
print(summary(log_model))

cat("\nConfusion matrix\n\n")
print(conf_matrix)

cat("\nModel performance\n\n")
print(model_performance)

cat("\nSpearman correlation between age and diagnosis level\n\n")
print(age_diagnosis_test)

cat("\nK-means cluster summary\n\n")
print(cluster_summary)

cat("\nDiagnosis level composition by cluster\n\n")
print(round(diagnosis_by_cluster_prop, 3))

cat("\nInterpretation note\n\n")
cat("Exploratory PCA and k-means clustering were used to assess whether numerical clinical variables showed broad structure. The clusters did not perfectly separate disease and no-disease patients, so they were not treated as clinical groups. However, cluster composition was explored to understand whether some clusters contained more lower-severity or no-disease patients. The logistic regression model was evaluated on a held-out test set. Performance should be interpreted cautiously because this is a small public dataset from a single source. Future work could use repeated cross-validation and external validation for a more robust estimate of performance.\n")

sink()


# 17. End note 

cat("Analysis complete. Outputs saved in the outputs folder.\n")
