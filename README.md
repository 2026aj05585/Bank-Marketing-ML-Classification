# Bank Marketing Subscription Prediction — Five-Model Comparison

## a. Problem Statement

The objective of this project is to build, evaluate, and compare five supervised machine learning classification models to predict whether a client of a Portuguese banking institution subscribes to a term deposit following a direct-marketing campaign.

The target variable is `y`, which contains two possible classes: `yes` (subscribed) and `no` (not subscribed).

The project covers dataset preparation, data preprocessing, model training, performance evaluation, saving trained model pipelines, and deploying an interactive Streamlit application for model evaluation.

## b. Dataset Description

- **Dataset:** Bank Marketing
- **Source:** UCI Machine Learning Repository
- **Dataset ID:** 222
- **Selected file:** `bank-additional-full.csv`
- **Number of records:** 41,188
- **Original input features:** 20
- **Selected predictor features:** 19
- **Target variable:** `y` (binary classification: yes/no)
- **License:** Creative Commons Attribution 4.0 (CC BY 4.0)

**Dataset citation:**

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

### Data Preprocessing

The `duration` feature is excluded from the selected predictors because it represents the duration of the marketing call. This information is only known after the call and would introduce information leakage for a prediction intended to be made before the call.

The dataset is divided into training and testing partitions using an 80:20 stratified split with a random state of 42.

The preprocessing steps are:

- Numerical features: missing-value imputation using the median and standardization.
- Categorical features: missing-value imputation using the most frequent value and one-hot encoding.
- Target variable: `y`, with `yes` treated as the positive class.

The preprocessing transformations are incorporated into the model pipelines and saved with the trained classifiers. This ensures consistent preprocessing during model evaluation.

## c. GitHub Repository Link

**GitHub Repository:**  
https://github.com/2026aj05585/Bank-Marketing-ML-Classification

The repository contains the source code, saved model artifacts, test dataset, evaluation reports, and application dependencies.

### Repository Structure

```text
Bank-Marketing-ML-Classification/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── support_vector_machine.joblib
│
└── reports/
    ├── model_metrics.csv
    └── confusion matrices and evaluation reports
```

The model filenames and report contents shown above describe the intended repository organization; the actual files should be checked against the public repository before submission.

### Project Files

- **`train.py`** — Downloads the UCI dataset, performs preprocessing, trains the five classifiers, evaluates their performance, and generates the saved models and evaluation artifacts.
- **`app.py`** — Provides the Streamlit application for evaluating the saved classifiers using uploaded labelled test data.
- **`requirements.txt`** — Lists the Python dependencies required to execute the project.
- **`README.md`** — Documents the problem, dataset, models, results, and application.
- **`test_data.csv`** — Contains the held-out labelled test records used for model evaluation.
- **`models/`** — Stores the five fitted model pipelines.
- **`reports/`** — Stores model comparison metrics and evaluation reports.

## d. Models Used and Evaluation Metrics

Five supervised machine learning classification algorithms were implemented and evaluated using the Bank Marketing dataset.

The classifiers are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (LinearSVC)

### Evaluation Metrics

The following six evaluation metrics are calculated for each classifier:

1. **Accuracy:** Measures the proportion of correctly classified observations.
2. **AUC (ROC-AUC):** Measures the model's ability to distinguish between the positive and negative classes.
3. **Precision:** Measures the proportion of positive predictions that are actually positive.
4. **Recall:** Measures the proportion of actual positive observations correctly identified.
5. **F1 Score:** Represents the harmonic mean of precision and recall.
6. **Matthews Correlation Coefficient (MCC):** Measures the quality of binary classifications using all four confusion-matrix categories.

The positive class is defined as `yes`.

### Model Comparison Table

The following results are the recorded five-model comparison results for the held-out test set. All five classifiers are compared using the same stated 80:20 stratified split and random state 42.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8350 | 0.8009 | 0.3679 | 0.6466 | 0.4689 | 0.4011 |
| Decision Tree | 0.8418 | 0.7812 | 0.3770 | 0.6196 | 0.4688 | 0.3986 |
| KNN | 0.8933 | 0.7460 | 0.5481 | 0.3006 | 0.3883 | 0.3535 |
| Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| Support Vector Machine | 0.8331 | 0.8007 | 0.3641 | 0.6455 | 0.4656 | 0.3973 |

*Note: These values are taken from the five available model-comparison CSV exports. Confirm them against the latest `reports/model_metrics.csv` generated by `train.py` before final submission.*

### Observations on Model Performance

| ML Model Name | Observation About Model Performance |
|---|---|
| **Logistic Regression** | Achieved an accuracy of 83.50% and the highest AUC (0.8009), recall (0.6466), F1 score (0.4689), and MCC (0.4011). It provides the strongest overall balance among the five classifiers, particularly for identifying clients who subscribe to a term deposit. |
| **Decision Tree** | Achieved an accuracy of 84.18%, with an F1 score of 0.4688, which is very close to Logistic Regression. However, its AUC (0.7812) and MCC (0.3986) are slightly lower, indicating marginally weaker overall performance. |
| **KNN** | Achieved the highest accuracy (89.33%) and precision (0.5481). However, its recall (0.3006) is considerably lower, indicating that it identifies a smaller proportion of actual subscribers. Its F1 score (0.3883) and MCC (0.3535) are also lower than those of Logistic Regression. |
| **Naive Bayes** | Achieved an accuracy of 80.49%, the lowest among the five classifiers. Its recall (0.6347) is relatively high, but its precision (0.3172) is the lowest, indicating that many of its positive predictions are false positives. |
| **Support Vector Machine** | Achieved an accuracy of 83.31% and an AUC of 0.8007, close to Logistic Regression. Its recall (0.6455), F1 score (0.4656), and MCC (0.3973) are also competitive, although slightly below Logistic Regression. |

### Overall Winner for This Dataset

**Logistic Regression is the strongest overall classifier based on the recorded comparison results.**

It achieves the highest AUC (0.8009), recall (0.6466), F1 score (0.4689), and MCC (0.4011) among the five evaluated classifiers. These results indicate a better balance between identifying actual term-deposit subscribers and maintaining overall classification quality.

Although KNN achieves the highest accuracy (0.8933) and precision (0.5481), its recall is considerably lower (0.3006). Therefore, it misses a larger proportion of actual subscribers.

Considering the importance of identifying the positive `yes` class, Logistic Regression is selected as the overall winner for this dataset.

## Streamlit Application

**Live Streamlit Application:**  
https://bank-marketing-ml-2026aj05585.streamlit.app/

The interactive application allows users to evaluate the five trained classification models using labelled CSV test data.

### Application Features

- CSV upload for labelled test data.
- Classifier selection through a dropdown menu.
- Evaluation of all five available saved classifiers.
- Display of Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Comparison table showing the performance of all five classifiers.
- Confusion matrix and classification report for the selected classifier.
- Download option for the evaluated metrics in CSV format.

The application loads the previously trained model pipelines from the `models/` directory. It uses the uploaded labelled test data to generate predictions and calculate evaluation metrics without retraining the classifiers.

The uploaded CSV must contain the required 19 predictor columns and the target column `y`, containing `yes` or `no` labels.

## Run Instructions

### 1. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Train the Classification Models

Execute the training script:

```bash
python train.py
```

This downloads the UCI Bank Marketing dataset, performs preprocessing, trains and evaluates all five classifiers, and generates the saved model pipelines, evaluation reports, and held-out test dataset.

### 3. Run the Streamlit Application Locally

Execute:

```bash
streamlit run app.py
```

The application opens in a web browser, allowing users to upload labelled test data, select a classifier, and review the evaluation results.

### 4. Deploy on Streamlit Community Cloud

1. Push the complete project to the GitHub repository.
2. Open Streamlit Community Cloud.
3. Sign in using the GitHub account.
4. Select **New App**.
5. Choose the project repository and branch.
6. Set the main application file to `app.py`.
7. Deploy the application.

## Reproducibility and Academic Integrity

- The random state is fixed at 42.
- The dataset is split using stratified 80:20 training and testing partitions.
- All five classifiers use the same dataset split and target definition.
- Preprocessing transformations are stored within the fitted model pipelines.
- The Streamlit application loads saved models and evaluates uploaded test data without retraining.
- The evaluation metrics are calculated using the positive class `yes`.
- The source code, model artifacts, dependencies, and test dataset are maintained in the GitHub repository.
- The assignment execution proof is captured separately from the BITS Virtual Lab environment.

The project demonstrates an end-to-end machine learning classification workflow, from dataset preparation and model comparison to saved model deployment and interactive evaluation.
