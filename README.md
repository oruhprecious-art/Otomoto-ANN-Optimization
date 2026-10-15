# Otomoto ANN Optimization for Customer Churn Prediction

## 1. Project Overview

This project focuses on developing and optimizing an Artificial Neural Network (ANN) to predict customer churn using the customer dataset supplied for the Nexford University assignment.

Customer churn occurs when customers stop using a company's products or services. Identifying customers who are likely to leave can help businesses make better decisions about customer retention and marketing activities.

The main objective of this project was to build a baseline ANN, compare different optimization algorithms, investigate methods for improving model performance, and select a suitable model based on experimental results.

The project also examined the effects of learning-rate selection, classification thresholds, and class imbalance on the model's ability to identify customers who churn.

**Note:** This project evaluates customer churn prediction. It does not directly measure the effectiveness of marketing segmentation or retention campaigns.

## 2. Project Objectives

The objectives of this project were to:

- Develop a baseline ANN for customer churn prediction.
- Compare Adam, Stochastic Gradient Descent (SGD), and L-BFGS-B optimization methods.
- Investigate different learning rates for the Adam optimizer.
- Examine the effects of classification threshold selection.
- Compare unweighted and class-weighted ANN models.
- Evaluate model performance using appropriate classification metrics.
- Identify a preferred model based on experimental evidence.
- Discuss the business implications and limitations of the findings.

## 3. Dataset Description

The project used the customer churn dataset supplied for the assignment.

The original dataset contained 7,043 customer records and 21 columns, including customer characteristics, service information, billing details, and the churn target.

### Target Distribution

| Customer Class | Number of Customers | Percentage |
|----------------|--------------------:|-----------:|
| No Churn       | 5,174               | 73.46%     |
| Churn          | 1,869               | 26.54%     |
| Total          | 7,043               | 100%       |

The dataset was imbalanced because the number of customers who stayed was substantially greater than the number who churned.

This imbalance made recall and F1-score important evaluation metrics, alongside accuracy and ROC-AUC.

## 4. Data Preparation

The following data preparation steps were performed:

1. **Missing values:** The `TotalCharges` column was converted to a numerical data type. Eleven blank values were identified and handled using median imputation.

2. **Categorical encoding:** One-hot encoding was used to convert categorical variables into numerical features suitable for ANN training.

3. **Target encoding:** The churn target was converted into binary values, with 0 representing customers who stayed and 1 representing customers who churned.

4. **Feature selection:** The `customerID` column was removed because it was an identifier rather than a meaningful predictive feature.

5. **Feature scaling:** StandardScaler was used to standardize numerical features.

6. **Data splitting:** The dataset was divided into training, validation, and test sets using a stratified split.

### Dataset Split

| Dataset | Number of Records |
|---------|------------------:|
| Training | 4,930 |
| Validation | 1,056 |
| Testing | 1,057 |
| Total | 7,043 |

The training set was used to fit the models. The validation set was used to compare configurations and select classification thresholds. The test set was reserved for final performance evaluation.

Preprocessing transformations were fitted using training data to reduce the risk of data leakage.

After preprocessing, the dataset contained 45 input features.

## 5. ANN Architecture

The baseline ANN used the following architecture and training configuration:

| Component | Configuration |
|-----------|---------------|
| Input | 45 preprocessed features |
| Hidden layer | 32 neurons |
| Hidden activation | ReLU |
| Output layer | 1 neuron |
| Output activation | Sigmoid |
| Loss function | Binary cross-entropy |
| Baseline optimizer | Adam |
| Baseline learning rate | 0.001 |
| Batch size | 32 |
| Maximum epochs | 50 |
| Early stopping | Validation loss, patience of 5 epochs |
| Baseline classification threshold | 0.50 |

Early stopping was used to reduce unnecessary training when validation loss stopped improving.

## 6. Optimization Experiments

### 6.1 Optimizer Comparison

Three optimization methods were evaluated:

- Adam
- Stochastic Gradient Descent (SGD) with momentum
- L-BFGS-B

The purpose was to compare how different optimization approaches affected the model's predictive performance.

### 6.2 Adam Learning-Rate Tuning

Three learning rates were tested:

| Learning Rate | Best Validation Loss | Epochs |
|--------------:|---------------------:|-------:|
| 0.0003 | 0.423003 | 37 |
| 0.0010 | 0.421941 | 16 |
| 0.0030 | 0.419409 | 11 |

The learning rate of 0.003 produced the lowest validation loss among the three tested values and was selected for subsequent experiments.

This result identifies the best value among those tested, not necessarily the globally optimal learning rate.

### 6.3 Classification Threshold Optimization

The default classification threshold of 0.50 was investigated because it did not identify enough actual churners in the baseline experiment.

Different thresholds were evaluated using validation data. The threshold was selected based on validation F1-score, with ties resolved in favor of the higher threshold.

The final unweighted model used a threshold of 0.30.

### 6.4 Class-Imbalance Experiment

An unweighted ANN was compared with a class-weighted ANN.

Class weighting assigned greater importance to the minority churn class during training. The purpose was to investigate whether this approach would improve the identification of customers likely to churn.

The two approaches were evaluated using their validation-selected classification thresholds and final test results.

## 7. Model Evaluation

### 7.1 Baseline ANN Results

The baseline model was evaluated using a classification threshold of 0.50.

| Metric | Baseline Result |
|--------|----------------:|
| Accuracy | 79.75% |
| Precision | 67.36% |
| Recall | 46.26% |
| F1-score | 54.85% |
| ROC-AUC | 84.24% |

Although the baseline achieved reasonable accuracy, its recall indicated that it missed many customers who actually churned.

This motivated further experiments focused on identifying more churners.

### 7.2 Optimizer Comparison Results

The following results were obtained using a classification threshold of 0.50.

| Metric | Adam | SGD | L-BFGS-B |
|--------|-----:|----:|--------:|
| Accuracy | 79.75% | 80.23% | 75.88% |
| Precision | 67.36% | 69.78% | 55.12% |
| Recall | 46.26% | 45.20% | 49.82% |
| F1-score | 54.85% | 54.86% | 52.34% |
| ROC-AUC | 84.24% | 84.37% | 78.82% |

SGD achieved the highest accuracy, precision, and ROC-AUC in this comparison. Its F1-score was almost identical to Adam's.

L-BFGS-B performed less favorably overall and did not converge within the configured iteration limit.

These results suggest that changing the optimizer alone did not produce a substantial improvement in churn detection.

### 7.3 Final Validation-Selected Model Comparison

The final comparison evaluated the unweighted and class-weighted models using thresholds selected from validation data.

| Metric | Unweighted ANN | Class-Weighted ANN |
|--------|---------------:|-------------------:|
| Classification threshold | 0.30 | 0.55 |
| Accuracy | 78.05% | 77.29% |
| Precision | 56.90% | 55.77% |
| Recall | 71.89% | 70.46% |
| F1-score | 63.52% | 62.26% |
| ROC-AUC | 84.42% | 84.26% |

The unweighted ANN achieved slightly better results across all the listed final test metrics.

It correctly identified 202 churners and missed 79. The class-weighted model identified 198 churners and missed 83.

Based on these results, the unweighted ANN with a classification threshold of 0.30 was selected as the preferred model for this experiment.

The classification threshold was selected using validation data before evaluating the final model on the test set.

## 8. Key Findings

The experiments produced several important findings:

1. **Optimizer selection matters, but improvements can be small.** SGD achieved slightly better overall results than Adam in the fixed-threshold optimizer comparison, but the F1-scores were almost identical.

2. **Learning-rate tuning improved the training configuration.** A learning rate of 0.003 produced the lowest validation loss among the three tested values.

3. **Classification threshold selection affected churn detection.** Using a threshold of 0.30 improved recall and F1-score compared with the baseline model.

4. **Class weighting did not improve the final results.** The unweighted model performed slightly better than the class-weighted alternative in the final validation-selected comparison.

5. **Accuracy alone was insufficient.** The final preferred model had lower accuracy than the baseline but identified a greater proportion of actual churners.

## 9. Business Implications

The selected ANN could help a business identify customers who may be at risk of leaving and prioritize them for further investigation or retention activities.

However, the model's predictions should not be treated as certain outcomes. Some customers predicted to churn may remain, while some customers predicted not to churn may leave.

A business considering deployment should evaluate:

- The cost of contacting customers who would have stayed.
- The financial impact of failing to identify customers who churn.
- The potential value of successfully retaining customers.
- The effectiveness of retention campaigns.
- The model's performance on new customer data.
- Potential fairness concerns and changes in customer behavior over time.

A cost-benefit analysis would help determine whether the selected classification threshold is appropriate for the business.

## 10. Limitations

This project has several limitations:

- The experiments used the supplied dataset and may not represent future customer behavior.
- The learning-rate search covered only three values.
- The optimizer comparison used a fixed classification threshold of 0.50, while the final models used validation-selected thresholds.
- The experiments did not directly measure the financial return of retention campaigns.
- The project evaluated churn prediction rather than directly measuring improvements in marketing segmentation.
- Further testing would be required to establish how well the model generalizes to new data.

## 11. Conclusion

This project investigated how optimization algorithms, learning rates, classification thresholds, and class weighting affected ANN performance for customer churn prediction.

The final unweighted ANN, using an Adam learning rate of 0.003 and a classification threshold of 0.30, achieved a test F1-score of 63.52%, recall of 71.89%, and ROC-AUC of 84.42%.

I selected this model because it provided the strongest results in the final comparison of the unweighted and class-weighted models.

Further evaluation, including financial analysis, testing on new data, fairness assessment, and ongoing monitoring, would be necessary before using the model in a real business environment.

## 12. Project Structure

The repository is organized to separate model scripts, results, visualizations, and saved models.

```text
Otomoto-ANN-Optimization/
│
├── README.md
├── baseline_ann.py
├── optimizer_comparison.py
├── adam_tuning.py
├── final_threshold_check.py
├── class_imbalance_experiment.py
├── requirements.txt
│
├── results/
│   ├── final_validation_selected_test_results.csv
│   └── validation_threshold_search.csv
│
├── figures/
│   ├── unweighted_ann_final_test_confusion_matrix.png
│   └── class_weighted_ann_final_test_confusion_matrix.png
│
└── models/
    ├── unweighted_ann_validation_tuned.keras
    └── class_weighted_ann_validation_tuned.keras
