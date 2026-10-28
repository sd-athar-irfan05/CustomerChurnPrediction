# Customer Churn Prediction

## Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a customer is likely to leave a company's services based on their account details, service usage, contract type, and billing information.

The project uses the Telco Customer Churn dataset to analyse customer behaviour and build a classification model.

## Features

* Customer churn data analysis
* Exploratory Data Analysis (EDA)
* Data cleaning and preprocessing
* Categorical feature encoding
* Logistic Regression model
* Random Forest model comparison
* Model evaluation using classification metrics
* Interactive Streamlit web application
* Customer churn prediction
* Churn probability display

## Technologies Used

* Python
* Pandas
* Matplotlib
* Scikit-learn
* Streamlit
* Joblib
* Jupyter Notebook

## Machine Learning Models

Two classification algorithms were trained and evaluated:

1. Logistic Regression
2. Random Forest Classifier

Logistic Regression was selected as the primary model based on the evaluation results.

## Model Performance

### Logistic Regression

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 80.41% |
| Precision | 65.41% |
| Recall    | 55.61% |
| F1-Score  | 60.12% |

### Random Forest

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 76.93% |
| Precision | 55.88% |
| Recall    | 62.30% |
| F1-Score  | 58.91% |

These scores were obtained on the project's test dataset. Performance may vary on new customer data.

## Project Structure

```text
CustomerChurnPrediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── app.py
├── main.py
├── logistic_regression_model.pkl
└── README.md
```

## How to Run the Project

### 1. Clone the Repository

```bash
git https://github.com/sd-athar-irfan05/CustomerChurnPrediction.git

```



### 2. Open the Project Folder

Open the project folder in VS Code.

### 3. Install Required Libraries

```bash
pip install pandas matplotlib scikit-learn streamlit joblib
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## How to Use the Application

1. Open the Streamlit application.
2. Enter the customer's details.
3. Click the **Predict Customer Churn** button.
4. View the predicted result.
5. Check the estimated churn probability.

## Dataset

The project uses the Telco Customer Churn dataset, which contains customer information, subscribed services, contract details, billing information, and churn status.

## Project Objective

The objective is to understand customer churn patterns and build a machine learning model that can help identify customers who may be at risk of leaving a service provider.
