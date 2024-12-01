# Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.feature_selection import mutual_info_classif, RFE
from umap import UMAP
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from xgboost import XGBClassifier

# Load Dataset
df = pd.read_csv('creditcard.csv')

# Reduce dataset size for faster processing
df = df.sample(frac=0.2, random_state=42)  # Use 20% of the data
print("Dataset Subsampled to 20% of Original Size")

# Exploratory Data Analysis
print("Dataset Information:")
print(df.info())

# Checking Distribution of Fraud Cases
fraud_count = df['Class'].value_counts()
print("\nFraud Cases Distribution:\n", fraud_count)

# Visualizing Class Imbalance
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.savefig('class_distribution.png')
plt.close()

# Correlation Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.savefig('correlation_heatmap.png')
plt.close()

# Correlation with Target
correlation_with_target = df.corr()['Class'].sort_values(ascending=False)
print("\nCorrelation with Target:\n", correlation_with_target)

# Mutual Information to Test Independence Assumptions
X = df.drop('Class', axis=1)
y = df['Class']
mi_scores = mutual_info_classif(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns)
plt.figure(figsize=(12, 6))
mi_scores.sort_values(ascending=False).plot.bar()
plt.title("Mutual Information Scores")
plt.savefig('mutual_information_scores.png')
plt.close()

# Data Preprocessing
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])
X_scaled = scaler.fit_transform(X)

# Dimensionality Reduction (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.5)
plt.title("PCA Visualization")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.savefig('pca_visualization.png')
plt.close()

# Dimensionality Reduction (t-SNE) with Subsampling
subsample_size = 5000
indices = np.random.choice(len(X_scaled), subsample_size, replace=False)
X_sample = X_scaled[indices]
y_sample = y.iloc[indices]

tsne = TSNE(n_components=2, random_state=42, perplexity=50)
X_tsne = tsne.fit_transform(X_sample)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='coolwarm', alpha=0.5)
plt.title("t-SNE Visualization (Subsampled)")
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")
plt.savefig('tsne_visualization_subsampled.png')
plt.close()

# Apply UMAP on the subsampled data
umap = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap.fit_transform(X_sample)
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_sample, cmap='coolwarm', alpha=0.5)
plt.title("UMAP Visualization (Subsampled)")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.savefig('umap_visualization_subsampled.png')
plt.close()

# Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Unsupervised Learning: K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='coolwarm', alpha=0.5)
plt.title("K-Means Clustering")
plt.savefig('kmeans_clustering.png')
plt.close()

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Baseline Model: Logistic Regression
log_model = LogisticRegression(max_iter=2000, random_state=42)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# Evaluation for Logistic Regression
print("\nLogistic Regression Evaluation:")
print("Classification Report:\n", classification_report(y_test, y_pred_log))
print("ROC-AUC Score:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]))

# Feature Selection Using RFE
rfe_selector = RFE(log_model, n_features_to_select=10)
rfe_selector.fit(X_train, y_train)
selected_features = X.columns[rfe_selector.support_]
print("\nSelected Features Using RFE:\n", selected_features)

# Advanced Model 1: Random Forest with Cross-Validation
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
cv_rf = StratifiedKFold(n_splits=3)  # Reduced number of folds for speed
grid_rf = GridSearchCV(rf_model, param_grid=param_grid_rf, cv=cv_rf, scoring='roc_auc')
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

# Random Forest Evaluation
print("\nRandom Forest Evaluation:")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("ROC-AUC Score:", roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]))

# Advanced Model 2: XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# XGBoost Evaluation
print("\nXGBoost Evaluation:")
print("Classification Report:\n", classification_report(y_test, y_pred_xgb))
print("ROC-AUC Score:", roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]))

# Model Comparison
print("\nModel Comparison:")
print(f"Logistic Regression AUC: {roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]):.4f}")
print(f"Random Forest AUC: {roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1]):.4f}")
print(f"XGBoost AUC: {roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1]):.4f}")

# Feature Importance (Random Forest)
importances_rf = best_rf.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1]
plt.figure(figsize=(12, 6))
plt.bar(range(X.shape[1]), importances_rf[indices_rf])
plt.xticks(range(X.shape[1]), X.columns[indices_rf], rotation=90)
plt.title("Feature Importance (Random Forest)")
plt.savefig('feature_importance_rf.png')
plt.close()

# ROC Curve Comparison
plt.figure(figsize=(10, 6))
# Logistic Regression
fpr_log, tpr_log, _ = roc_curve(y_test, log_model.predict_proba(X_test)[:, 1])
plt.plot(fpr_log, tpr_log, label='Logistic Regression (AUC = {:.4f})'.format(roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1])))
# Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, best_rf.predict_proba(X_test)[:, 1])
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = {:.4f})'.format(roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])))
# XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (AUC = {:.4f})'.format(roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])))
# Random Line
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig('roc_curve_comparison.png')
plt.close()
