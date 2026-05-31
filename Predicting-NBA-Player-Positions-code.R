set.seed(123)

packages <- c("tidyverse", "class", "e1071", "caret", "cluster", "factoextra")
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) install.packages(packages[!installed])

library(tidyverse)
library(class)
library(e1071)
library(caret)
library(cluster)
library(factoextra)


# 1. Load Data
# ------------------------------------------------

nba <- read.csv("nba_data.csv", check.names = TRUE)

cat("Dataset dimensions:\n")
print(dim(nba))

cat("\nColumn names:\n")
print(names(nba))

cat("\nMissing values before cleaning:\n")
print(colSums(is.na(nba)))

names(nba) <- make.names(names(nba), unique = TRUE)

# Remove blank Excel column if it exists
if ("X" %in% names(nba)) {
  nba <- nba %>% select(-X)
}

# Fix missing 3P% values: players with no 3PA get 0 for 3P%
nba$X3P.[is.na(nba$X3P.)] <- 0

cat("\nMissing values after cleaning:\n")
print(colSums(is.na(nba)))


# 2. Create Guard vs Non-Guard Response
# ------------------------------------------------

nba$GuardClass <- ifelse(grepl("G", nba$Pos), "Guard", "NonGuard")
nba$GuardClass <- as.factor(nba$GuardClass)

cat("\nClass distribution:\n")
print(table(nba$GuardClass))


# 3. Select Predictors
# ------------------------------------------------

predictor_data <- nba %>%
  select(where(is.numeric))

cat("\nNumeric predictor columns:\n")
print(names(predictor_data))

model_data <- cbind(predictor_data, GuardClass = nba$GuardClass)

cat("\nMissing values in model data:\n")
print(colSums(is.na(model_data)))

if (any(is.na(model_data))) {
  stop("There are still missing values in model_data. Check the missing value table above.")
}

cat("\nFinal model data dimensions:\n")
print(dim(model_data))


# 4. Train/Test Split
# ------------------------------------------------

set.seed(123)

train_index <- createDataPartition(model_data$GuardClass, p = 0.80, list = FALSE)

train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]

x_train <- train_data %>% select(-GuardClass)
x_test  <- test_data %>% select(-GuardClass)

y_train <- train_data$GuardClass
y_test  <- test_data$GuardClass

preprocess <- preProcess(x_train, method = c("center", "scale"))

x_train_scaled <- predict(preprocess, x_train)
x_test_scaled  <- predict(preprocess, x_test)

cat("\nTraining size:\n")
print(dim(x_train_scaled))

cat("\nTesting size:\n")
print(dim(x_test_scaled))


# 5. KNN Classification
# ------------------------------------------------

k_values <- 1:25
knn_results <- data.frame(K = k_values, Accuracy = NA)

for (k in k_values) {
  pred <- knn(
    train = x_train_scaled,
    test = x_test_scaled,
    cl = y_train,
    k = k
  )
  
  knn_results$Accuracy[knn_results$K == k] <- mean(pred == y_test)
}

cat("\nKNN tuning results:\n")
print(knn_results)

# Select best K based on accuracy; if tied, choose smallest K greater than 1
max_acc <- max(knn_results$Accuracy)
candidate_k <- knn_results$K[knn_results$Accuracy == max_acc]
best_k <- ifelse(any(candidate_k > 1), min(candidate_k[candidate_k > 1]), min(candidate_k))

cat("\nChosen K:\n")
print(best_k)

knn_pred <- knn(
  train = x_train_scaled,
  test = x_test_scaled,
  cl = y_train,
  k = best_k
)

cat("\nKNN Confusion Matrix:\n")
knn_cm <- confusionMatrix(knn_pred, y_test)
print(knn_cm)

knn_accuracy <- knn_cm$overall["Accuracy"]

ggplot(knn_results, aes(x = K, y = Accuracy)) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept = best_k, linetype = "dashed") +
  labs(
    title = "KNN Accuracy by K",
    x = "K",
    y = "Test Accuracy"
  )

ggsave("knn_accuracy_plot.png", width = 7, height = 5)


# 6. SVM Classification
# ------------------------------------------------

svm_train <- data.frame(x_train_scaled, GuardClass = y_train)
svm_test  <- data.frame(x_test_scaled, GuardClass = y_test)

set.seed(123)

svm_tune <- tune(
  svm,
  GuardClass ~ .,
  data = svm_train,
  kernel = "radial",
  ranges = list(
    cost = c(0.1, 1, 10, 100),
    gamma = c(0.001, 0.01, 0.1, 1)
  )
)

cat("\nSVM tuning results:\n")
print(svm_tune)

best_svm <- svm_tune$best.model

cat("\nBest SVM model summary:\n")
print(summary(best_svm))

svm_pred <- predict(best_svm, svm_test)

cat("\nSVM Confusion Matrix:\n")
svm_cm <- confusionMatrix(svm_pred, y_test)
print(svm_cm)

svm_accuracy <- svm_cm$overall["Accuracy"]

# 2D SVM boundary plot using two interpretable predictors
svm_2d_train <- data.frame(
  AST = x_train_scaled$AST,
  TRB = x_train_scaled$TRB,
  GuardClass = y_train
)

svm_2d <- svm(
  GuardClass ~ AST + TRB,
  data = svm_2d_train,
  kernel = "radial",
  cost = best_svm$cost,
  gamma = best_svm$gamma
)

png("svm_boundary_plot.png", width = 900, height = 700)
plot(
  svm_2d,
  svm_2d_train,
  main = "SVM Boundary Plot Using AST and TRB"
)
dev.off()


# 7. Compare Supervised Models
# ------------------------------------------------

supervised_results <- data.frame(
  Model = c("KNN", "SVM"),
  Test_Accuracy = c(knn_accuracy, svm_accuracy)
)

cat("\nSupervised Model Comparison:\n")
print(supervised_results)

write.csv(supervised_results, "supervised_model_comparison.csv", row.names = FALSE)
write.csv(knn_results, "knn_tuning_results.csv", row.names = FALSE)


# 8. Full Data Scaling for Clustering
# ------------------------------------------------

full_x <- model_data %>% select(-GuardClass)

full_preprocess <- preProcess(full_x, method = c("center", "scale"))
full_scaled <- predict(full_preprocess, full_x)


# 9. K-Means Clustering
# ------------------------------------------------

set.seed(123)

k_cluster_values <- 2:10

wss <- numeric(length(k_cluster_values))
sil_scores <- numeric(length(k_cluster_values))

for (i in seq_along(k_cluster_values)) {
  k <- k_cluster_values[i]
  
  km <- kmeans(full_scaled, centers = k, nstart = 25)
  wss[i] <- km$tot.withinss
  
  sil <- silhouette(km$cluster, dist(full_scaled))
  sil_scores[i] <- mean(sil[, 3])
}

kmeans_results <- data.frame(
  K = k_cluster_values,
  Total_Within_SS = wss,
  Avg_Silhouette = sil_scores
)

cat("\nK-Means Results:\n")
print(kmeans_results)

best_kmeans_k <- kmeans_results$K[which.max(kmeans_results$Avg_Silhouette)]

cat("\nBest K for K-Means:\n")
print(best_kmeans_k)

set.seed(123)
best_kmeans <- kmeans(full_scaled, centers = best_kmeans_k, nstart = 25)

cat("\nK-Means cluster sizes:\n")
print(best_kmeans$size)

cat("\nK-Means cluster vs GuardClass:\n")
print(table(best_kmeans$cluster, model_data$GuardClass))

write.csv(kmeans_results, "kmeans_results.csv", row.names = FALSE)

ggplot(kmeans_results, aes(x = K, y = Total_Within_SS)) +
  geom_line() +
  geom_point() +
  labs(
    title = "K-Means Elbow Plot",
    x = "Number of Clusters K",
    y = "Total Within-Cluster Sum of Squares"
  )

ggsave("kmeans_elbow_plot.png", width = 7, height = 5)

ggplot(kmeans_results, aes(x = K, y = Avg_Silhouette)) +
  geom_line() +
  geom_point() +
  labs(
    title = "K-Means Average Silhouette by K",
    x = "Number of Clusters K",
    y = "Average Silhouette"
  )

ggsave("kmeans_silhouette_plot.png", width = 7, height = 5)

fviz_cluster(
  best_kmeans,
  data = full_scaled,
  geom = "point",
  main = "K-Means Clustering of NBA Player Profiles"
)

ggsave("kmeans_cluster_plot.png", width = 7, height = 5)


# 10. Hierarchical Clustering
# ------------------------------------------------

dist_matrix <- dist(full_scaled)

linkages <- c("complete", "average", "ward.D2")

hier_results <- data.frame(
  Linkage = character(),
  K = numeric(),
  Avg_Silhouette = numeric()
)

for (link in linkages) {
  hc <- hclust(dist_matrix, method = link)
  
  for (k in 2:10) {
    clusters <- cutree(hc, k = k)
    sil <- silhouette(clusters, dist_matrix)
    
    hier_results <- rbind(
      hier_results,
      data.frame(
        Linkage = link,
        K = k,
        Avg_Silhouette = mean(sil[, 3])
      )
    )
  }
}

cat("\nHierarchical Clustering Results:\n")
print(hier_results)

best_hier <- hier_results[which.max(hier_results$Avg_Silhouette), ]

cat("\nBest Hierarchical Clustering Setting:\n")
print(best_hier)

write.csv(hier_results, "hierarchical_results.csv", row.names = FALSE)

best_hc <- hclust(dist_matrix, method = best_hier$Linkage)
best_hc_clusters <- cutree(best_hc, k = best_hier$K)

cat("\nHierarchical cluster vs GuardClass:\n")
print(table(best_hc_clusters, model_data$GuardClass))

png("hierarchical_dendrogram.png", width = 900, height = 700)
plot(
  best_hc,
  labels = FALSE,
  main = paste("Hierarchical Clustering Dendrogram -", best_hier$Linkage),
  xlab = "",
  sub = ""
)
rect.hclust(best_hc, k = best_hier$K, border = 2:6)
dev.off()


# 11. Compare Clustering Models
# ------------------------------------------------

best_kmeans_sil <- max(kmeans_results$Avg_Silhouette)
best_hier_sil <- best_hier$Avg_Silhouette

clustering_comparison <- data.frame(
  Model = c("K-Means", "Hierarchical"),
  Best_Setting = c(
    paste("K =", best_kmeans_k),
    paste("Linkage =", best_hier$Linkage, ", K =", best_hier$K)
  ),
  Avg_Silhouette = c(best_kmeans_sil, best_hier_sil)
)

cat("\nClustering Model Comparison:\n")
print(clustering_comparison)

write.csv(clustering_comparison, "clustering_model_comparison.csv", row.names = FALSE)


# 12. Fit Best Supervised Model on Full Dataset
# ------------------------------------------------

if (svm_accuracy > knn_accuracy) {
  
  cat("\nBest supervised model: SVM\n")
  
  full_supervised <- data.frame(full_scaled, GuardClass = model_data$GuardClass)
  
  final_svm <- svm(
    GuardClass ~ .,
    data = full_supervised,
    kernel = "radial",
    cost = best_svm$cost,
    gamma = best_svm$gamma
  )
  
  cat("\nFinal SVM model on full dataset:\n")
  print(summary(final_svm))
  
  final_pred <- predict(final_svm, full_supervised)
  
  cat("\nFinal SVM full-data confusion matrix:\n")
  print(confusionMatrix(final_pred, model_data$GuardClass))
  
} else {
  
  cat("\nBest supervised model: KNN\n")
  
  final_knn_pred <- knn(
    train = full_scaled,
    test = full_scaled,
    cl = model_data$GuardClass,
    k = best_k
  )
  
  cat("\nFinal KNN full-data confusion matrix:\n")
  print(confusionMatrix(final_knn_pred, model_data$GuardClass))
}


# 13. Save Cleaned Dataset
# ------------------------------------------------

final_output <- nba
final_output$GuardClass <- nba$GuardClass

write.csv(final_output, "cleaned_nba_data_with_guardclass.csv", row.names = FALSE)

cat("\nAnalysis complete. Saved all plots and CSV result files.\n")