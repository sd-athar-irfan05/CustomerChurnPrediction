import pandas as pd

# Load the dataset

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display first 5 rows

print(df.head())

# Check number of rows and columns

print("Dataset Shape:")
print(df.shape)

# Check column names

print("\nColumn Names:")
print(df.columns)

# Check data types and non-null values

print("\nDataset Information:")
df.info()

# Check statistical summary

print("\nStatistical Summary:")
print(df.describe())


# Check missing values in each column

print("\nMissing Values:")
print(df.isnull().sum())

# Check total missing values

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Check duplicate rows

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Check dataset data types

print("\nData Types:")
print(df.dtypes)


print("\nTotalCharges Data Type:")
print(df["TotalCharges"].dtype)

print("\nTotalCharges Sample Values:")
print(df["TotalCharges"].head(10).tolist())

# Convert TotalCharges from string to numeric

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check the updated data type

print("\nUpdated TotalCharges Data Type:")
print(df["TotalCharges"].dtype)

# Check missing values created during conversion

print("\nMissing TotalCharges After Conversion:")
print(df["TotalCharges"].isnull().sum())


# Display rows where TotalCharges is missing

missing_rows = df[df["TotalCharges"].isnull()]

print("\nRows with Missing TotalCharges:")
print(missing_rows[["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

# Count customers with zero tenure

print("\nCustomers with Zero Tenure:")
print((df["tenure"] == 0).sum())


# Replace missing TotalCharges with 0

df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Verify missing values

print("\nMissing Values After Cleaning:")
print(df["TotalCharges"].isnull().sum())

# Check the cleaned rows

print("\nCleaned TotalCharges for Zero-Tenure Customers:")
print(df.loc[df["tenure"] == 0, ["tenure", "TotalCharges"]])


# Check duplicate rows

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Check duplicate customer IDs

print("\nDuplicate Customer IDs:")
print(df["customerID"].duplicated().sum())

# Check total rows and columns

print("\nDataset Shape After Cleaning:")
print(df.shape)


# Identify numerical columns

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

print("\nNumerical Columns:")
print(numerical_columns.tolist())

# Identify categorical columns

categorical_columns = df.select_dtypes(include=["object"]).columns

print("\nCategorical Columns:")
print(categorical_columns.tolist())

# Check number of columns in each group

print("\nNumber of Numerical Columns:", len(numerical_columns))
print("Number of Categorical Columns:", len(categorical_columns))


# Count customers in each churn category

print("\nChurn Count:")
print(df["Churn"].value_counts())

# Calculate churn percentage

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True).mul(100).round(2))


# Count customers by contract type and churn status

print("\nContract-wise Churn Count:")
print(pd.crosstab(df["Contract"], df["Churn"]))

# Calculate churn percentage within each contract type

print("\nContract-wise Churn Percentage:")
print(
pd.crosstab(
df["Contract"],
df["Churn"],
normalize="index"
).mul(100).round(2)
)


# Calculate average tenure for each churn category

print("\nAverage Tenure by Churn Status:")
print(df.groupby("Churn")["tenure"].mean().round(2))

# Calculate minimum and maximum tenure by churn status

print("\nTenure Statistics by Churn Status:")
print(
df.groupby("Churn")["tenure"]
.agg(["min", "max", "mean", "median"])
.round(2)
)


# Analyze monthly charges by churn status

print("\nMonthly Charges by Churn Status:")

print(
df.groupby("Churn")["MonthlyCharges"]
.agg(["mean", "median", "min", "max"])
.round(2)
)


# Count churn by internet service

print("\nInternet Service-wise Churn Count:")
print(pd.crosstab(df["InternetService"], df["Churn"]))

# Calculate churn percentage within each internet service category

print("\nInternet Service-wise Churn Percentage:")
print(
pd.crosstab(
df["InternetService"],
df["Churn"],
normalize="index"
).mul(100).round(2)
)


# Count churn by payment method

print("\nPayment Method-wise Churn Count:")
print(pd.crosstab(df["PaymentMethod"], df["Churn"]))

# Calculate churn percentage for each payment method

print("\nPayment Method-wise Churn Percentage:")
print(
pd.crosstab(
df["PaymentMethod"],
df["Churn"],
normalize="index"
).mul(100).round(2)
)


# Create tenure groups

df["TenureGroup"] = pd.cut(
df["tenure"],
bins=[-1, 12, 24, 48, 72],
labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
)

# Count churn by tenure group

print("\nTenure Group-wise Churn Count:")
print(pd.crosstab(df["TenureGroup"], df["Churn"]))

# Calculate churn percentage within each tenure group

print("\nTenure Group-wise Churn Percentage:")
print(
pd.crosstab(
df["TenureGroup"],
df["Churn"],
normalize="index"
).mul(100).round(2)
)

import matplotlib.pyplot as plt

# Count customers by churn status

churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"])

# Create bar chart

plt.figure(figsize=(7, 5))
plt.bar(churn_counts.index, churn_counts.values)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

# Display count labels above the bars

for i, value in enumerate(churn_counts.values):
    plt.text(i, value, str(value), ha="center", va="bottom")

plt.tight_layout()
plt.show()


# Create a contract-wise churn count table

contract_churn = pd.crosstab(df["Contract"], df["Churn"])

# Plot grouped bar chart

contract_churn.plot(kind="bar", figsize=(9, 5))

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn Status")
plt.tight_layout()
plt.show()


# Create tenure-wise churn count table

tenure_churn = pd.crosstab(df["TenureGroup"], df["Churn"])

# Plot grouped bar chart

tenure_churn.plot(kind="bar", figsize=(9, 5))

plt.title("Customer Churn by Tenure Group")
plt.xlabel("Tenure Group")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn Status")
plt.tight_layout()
plt.show()


# Separate monthly charges by churn status

no_churn = df[df["Churn"] == "No"]["MonthlyCharges"]
yes_churn = df[df["Churn"] == "Yes"]["MonthlyCharges"]

# Create box plot

plt.figure(figsize=(8, 5))
plt.boxplot(
[no_churn, yes_churn],
tick_labels=["No Churn", "Churn"],
showmeans=True
)

plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn Status")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()


# Calculate churn percentage by contract type

contract_churn_rate = (
pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
)

# Select churn percentage

churn_rate = contract_churn_rate["Yes"]

# Create bar chart

plt.figure(figsize=(8, 5))
plt.bar(churn_rate.index, churn_rate.values)

plt.title("Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, 100)

# Display percentage labels

for i, value in enumerate(churn_rate.values):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.show()


# Count churn by gender

print("\nGender-wise Churn Count:")
print(pd.crosstab(df["gender"], df["Churn"]))

# Calculate churn percentage by gender

gender_churn_rate = pd.crosstab(
df["gender"],
df["Churn"],
normalize="index"
).mul(100).round(2)

print("\nGender-wise Churn Percentage:")
print(gender_churn_rate)

# Plot churn percentage

gender_churn_rate["Yes"].plot(
kind="bar",
figsize=(7, 5)
)

plt.title("Churn Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


print("\n========== EDA SUMMARY ==========")

# Overall churn rate

overall_churn_rate = (df["Churn"] == "Yes").mean() * 100
print(f"Overall Churn Rate: {overall_churn_rate:.2f}%")

# Contract with the highest churn rate

contract_rates = pd.crosstab(
df["Contract"], df["Churn"], normalize="index"
) * 100

print(
"Highest Churn Contract:",
contract_rates["Yes"].idxmax()
)

# Internet service with the highest churn rate

internet_rates = pd.crosstab(
df["InternetService"], df["Churn"], normalize="index"
) * 100

print(
"Highest Churn Internet Service:",
internet_rates["Yes"].idxmax()
)

# Payment method with the highest churn rate

payment_rates = pd.crosstab(
df["PaymentMethod"], df["Churn"], normalize="index"
) * 100

print(
"Highest Churn Payment Method:",
payment_rates["Yes"].idxmax()
)

# Tenure group with the highest churn rate

tenure_rates = pd.crosstab(
df["TenureGroup"], df["Churn"], normalize="index"
) * 100

print(
"Highest Churn Tenure Group:",
tenure_rates["Yes"].idxmax()
)

print("=================================")


# Separate features and target

X = df.drop(columns=["Churn", "customerID", "TenureGroup"])
y = df["Churn"]

# Display feature information

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nTarget Categories:")
print(y.unique())

print("\nFeature Columns:")
print(X.columns.tolist())


# Identify categorical columns in X

categorical_columns = X.select_dtypes(include=["object"]).columns

print("\nCategorical Columns in X:")
print(categorical_columns.tolist())

print("\nNumber of Categorical Columns:")
print(len(categorical_columns))

# Identify numerical columns in X

numerical_columns = X.select_dtypes(
include=["int64", "float64"]
).columns

print("\nNumerical Columns in X:")
print(numerical_columns.tolist())

print("\nNumber of Numerical Columns:")
print(len(numerical_columns))


# Convert categorical features into numerical features

X_encoded = pd.get_dummies(
X,
columns=categorical_columns,
drop_first=True,
dtype=int
)

# Check the encoded dataset

print("\nOriginal Features Shape:")
print(X.shape)

print("\nEncoded Features Shape:")
print(X_encoded.shape)

print("\nFirst 5 Rows of Encoded Data:")
print(X_encoded.head())

print("\nData Types After Encoding:")
print(X_encoded.dtypes.value_counts())


# Convert target labels into numerical values

y_encoded = y.map({"No": 0, "Yes": 1})

# Check original and encoded target values

print("\nOriginal Target Values:")
print(y.value_counts())

print("\nEncoded Target Values:")
print(y_encoded.value_counts())

# Check for missing values after encoding

print("\nMissing Values in Encoded Target:")
print(y_encoded.isnull().sum())

# Display first 10 encoded labels

print("\nFirst 10 Encoded Target Values:")
print(y_encoded.head(10).tolist())


from sklearn.model_selection import train_test_split

# Split features and target into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
X_encoded,
y_encoded,
test_size=0.20,
random_state=42,
stratify=y_encoded
)

# Check the shapes

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# Check churn distribution in training and testing sets

print("\nTraining Target Distribution:")
print(y_train.value_counts(normalize=True).mul(100).round(2))

print("\nTesting Target Distribution:")
print(y_test.value_counts(normalize=True).mul(100).round(2))


from sklearn.linear_model import LogisticRegression

# Create the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model using training data
model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")


# Predict churn for test data
y_pred = model.predict(X_test)

# Display first 10 actual and predicted values
print("\nFirst 10 Actual Values:")
print(y_test.head(10).values)

print("\nFirst 10 Predicted Values:")
print(y_pred[:10])

from sklearn.metrics import accuracy_score

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Confusion Matrix - Logistic Regression")

plt.show()

from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(
    y_test,
    y_pred,
    target_names=["No Churn", "Churn"]
)

print("\nClassification Report:")
print(report)


from sklearn.ensemble import RandomForestClassifier

# Create the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# Train the model
rf_model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")


from sklearn.metrics import accuracy_score

# Make predictions using Random Forest
y_pred_rf = rf_model.predict(X_test)

# Calculate Random Forest accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:", round(accuracy_rf * 100, 2), "%")

# Display first 10 actual and predicted values
print("\nFirst 10 Actual Values:")
print(y_test.head(10).values)

print("\nFirst 10 Random Forest Predictions:")
print(y_pred_rf[:10])


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create Random Forest confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("\nRandom Forest Confusion Matrix:")
print(cm_rf)

# Visualize confusion matrix
plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Confusion Matrix - Random Forest")

plt.show()


from sklearn.metrics import classification_report

# Generate Random Forest classification report
report_rf = classification_report(
    y_test,
    y_pred_rf,
    target_names=["No Churn", "Churn"]
)

print("\nRandom Forest Classification Report:")
print(report_rf)


import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate Logistic Regression metrics
lr_accuracy = accuracy_score(y_test, y_pred)
lr_precision = precision_score(y_test, y_pred)
lr_recall = recall_score(y_test, y_pred)
lr_f1 = f1_score(y_test, y_pred)

# Calculate Random Forest metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

# Create comparison table
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [lr_accuracy, rf_accuracy],
    "Precision": [lr_precision, rf_precision],
    "Recall": [lr_recall, rf_recall],
    "F1-Score": [lr_f1, rf_f1]
})

print("\nModel Comparison:")
print(comparison.round(4).to_string(index=False))


import joblib

# Save the Logistic Regression model
joblib.dump(model, "logistic_regression_model.pkl")

print("\nLogistic Regression Model Saved Successfully!")