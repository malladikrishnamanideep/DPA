

# **Fraud Detection System**

This project demonstrates a fraud detection system using a Python script. It processes the `creditcard.csv` dataset, which contains anonymized credit card transactions, to classify fraudulent transactions. The dataset is sourced from Kaggle.

---

## **Prerequisites**

- Python 3.8 or higher installed on your system.
- Required Python libraries installed (see below).
- `creditcard.csv` dataset downloaded from Kaggle.

---

## **Setup and Usage**

### 1. Download the Dataset
- Download the `creditcard.csv` dataset from [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- Place the `creditcard.csv` file in the same folder as the Python script.

### 2. Install Required Libraries
Install the required libraries by running:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost umap-learn
```

### 3. Run the Script
Navigate to the folder containing the script and dataset, and execute the script:
```bash
python your_script_name.py
```

---

## **Project Files**

- **`your_script_name.py`**: The Python script for fraud detection.
- **`creditcard.csv`**: The dataset file containing credit card transactions.

---

## **Script Overview**

1. **Data Loading**:
   - Loads the `creditcard.csv` dataset.
   - Reduces the dataset size for faster processing.

2. **Exploratory Data Analysis (EDA)**:
   - Displays dataset information and class distribution.
   - Generates visualizations such as:
     - Class distribution plot.
     - Correlation heatmap.

3. **Feature Engineering**:
   - Scales numerical features (`Time` and `Amount`).
   - Extracts insights using PCA, t-SNE, and UMAP for dimensionality reduction.

4. **Model Training and Evaluation**:
   - Trains models including Logistic Regression, Random Forest, and XGBoost.
   - Handles class imbalance using SMOTE.
   - Evaluates models with classification reports and ROC-AUC scores.

5. **Unsupervised Learning**:
   - Implements K-Means clustering for unsupervised fraud detection.

6. **Visualization**:
   - Saves visualizations such as PCA, t-SNE, UMAP, and ROC curves.

---

## **Output**

- **Saved Visualizations**:
  - `class_distribution.png`: Class distribution plot.
  - `correlation_heatmap.png`: Correlation heatmap.
  - `pca_visualization.png`: PCA visualization.
  - `tsne_visualization_subsampled.png`: t-SNE visualization.
  - `umap_visualization_subsampled.png`: UMAP visualization.
  - `roc_curve_comparison.png`: ROC curve comparison.

- **Model Performance**:
  - Prints classification reports and ROC-AUC scores for Logistic Regression, Random Forest, and XGBoost.

---

## **Troubleshooting**

1. **Missing Libraries**:
   - Ensure all required Python libraries are installed. Use:
     ```bash
     pip install -r requirements.txt
     ```
     Create a `requirements.txt` file if needed:
     ```plaintext
     pandas
     numpy
     matplotlib
     seaborn
     scikit-learn
     imbalanced-learn
     xgboost
     umap-learn
     ```

2. **Dataset Not Found**:
   - Ensure the `creditcard.csv` file is in the same folder as the Python script.

3. **Large Dataset Processing**:
   - The script uses a subsample for faster processing. Modify the fraction in `df.sample(frac=0.2)` if needed.

---

## **Future Enhancements**

- Implement a dashboard for real-time fraud detection.
- Add additional models for comparison.
- Deploy the solution as a web service using FastAPI.

