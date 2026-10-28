import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# 2. Load Trained Model
# --------------------------------------------------

model = joblib.load("logistic_regression_model.pkl")

# --------------------------------------------------
# 3. Sidebar
# --------------------------------------------------

st.sidebar.title("About Project")

st.sidebar.write(
    "Customer Churn Prediction is a Machine Learning project "
    "that predicts whether a customer is likely to leave a company."
)

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Test Accuracy: 80.41%")

st.sidebar.markdown("---")

st.sidebar.info(
    "Enter customer details and click the prediction button "
    "to get the result."
)

# --------------------------------------------------
# 4. Main Heading
# --------------------------------------------------

st.title("Customer Churn Prediction")

st.write(
    "This application predicts whether a customer is likely "
    "to leave the company's services."
)

st.markdown("---")

# --------------------------------------------------
# 5. Customer Details Input
# --------------------------------------------------

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Is the customer a Senior Citizen?",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Does the customer have a Partner?",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Does the customer have Dependents?",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "Yes", "No"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col2:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=840.0
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# --------------------------------------------------
# 6. Create Customer DataFrame
# --------------------------------------------------

customer_data = {
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "PaperlessBilling": paperless_billing,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "PaymentMethod": payment_method
}

input_df = pd.DataFrame([customer_data])

# --------------------------------------------------
# 7. Customer Input Summary
# --------------------------------------------------

st.markdown("---")
st.subheader("Customer Input Summary")

st.dataframe(
    input_df,
    use_container_width=True
)

# --------------------------------------------------
# 8. Prepare Input Data for Model
# --------------------------------------------------

df_reference = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

df_reference["TotalCharges"] = pd.to_numeric(
    df_reference["TotalCharges"],
    errors="coerce"
)

df_reference["TotalCharges"] = (
    df_reference["TotalCharges"].fillna(0)
)

X_reference = df_reference.drop(
    columns=["Churn", "customerID"]
)

categorical_cols = X_reference.select_dtypes(
    include="object"
).columns.tolist()

X_reference_encoded = pd.get_dummies(
    X_reference,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

input_encoded = pd.get_dummies(
    input_df,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

# Match the exact feature columns used by the trained model
input_encoded = input_encoded.reindex(
    columns=X_reference_encoded.columns,
    fill_value=0
)

# --------------------------------------------------
# 9. Prediction Button
# --------------------------------------------------

st.markdown("---")

predict_button = st.button(
    "Predict Customer Churn",
    type="primary"
)

if predict_button:

    prediction = model.predict(input_encoded)

    churn_probability = model.predict_proba(
        input_encoded
    )[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error(
            "Prediction: Customer is likely to Churn."
        )
    else:
        st.success(
            "Prediction: Customer is likely to Stay."
        )

    st.metric(
        label="Churn Probability",
        value=f"{churn_probability * 100:.2f}%"
    )

    st.caption(
        "This probability is the model's estimate, "
        "not a guarantee of the customer's future behaviour."
    )
