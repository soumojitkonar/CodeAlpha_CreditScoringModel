import streamlit as st
import pandas as pd
import joblib


# =========================================================
# LOAD MODEL AND PREPROCESSOR
# =========================================================

model = joblib.load("models/credit_scoring_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")


# =========================================================
# HUMAN-READABLE FEATURE NAMES
# =========================================================

feature_name_mapping = {
    "A11": "Less than 0 DM",
    "A12": "0 to 200 DM",
    "A13": "200 DM or more",
    "A14": "No checking account",

    "A30": "No credits taken / all credits paid",
    "A31": "All credits at this bank paid",
    "A32": "Existing credits paid properly",
    "A33": "Previous payment delays",
    "A34": "Critical account / other credits",

    "A40": "New car",
    "A41": "Used car",
    "A42": "Furniture / Equipment",
    "A43": "Radio / Television",
    "A44": "Domestic appliances",
    "A45": "Repairs",
    "A46": "Education",
    "A47": "Vacation",
    "A48": "Retraining",
    "A49": "Business",
    "A410": "Other",

    "A61": "Less than 100 DM savings",
    "A62": "100 to 499 DM savings",
    "A63": "500 to 999 DM savings",
    "A64": "1000 DM or more savings",
    "A65": "Unknown / No savings account",

    "A71": "Unemployed",
    "A72": "Less than 1 year employment",
    "A73": "1 to 4 years employment",
    "A74": "4 to 7 years employment",
    "A75": "7 years or more employment",

    "A91": "Male - Divorced / Separated",
    "A92": "Female - Divorced / Married",
    "A93": "Male - Single",
    "A94": "Male - Married / Widowed",
    "A95": "Female - Single",

    "A101": "No other debtor",
    "A102": "Co-applicant",
    "A103": "Guarantor",

    "A121": "Real Estate",
    "A122": "Building Society Savings / Life Insurance",
    "A123": "Car or Other Property",
    "A124": "Unknown / No Property",

    "A141": "Bank installment plan",
    "A142": "Store installment plan",
    "A143": "No other installment plan",

    "A151": "Rent",
    "A152": "Own",
    "A153": "For Free",

    "A171": "Unemployed / Unskilled Non-resident",
    "A172": "Unskilled Resident",
    "A173": "Skilled Employee / Official",
    "A174": "Management / Self-employed / Highly Qualified",

    "A191": "No telephone",
    "A192": "Registered telephone",

    "A201": "Foreign Worker: Yes",
    "A202": "Foreign Worker: No"
}


# =========================================================
# CONVERT TECHNICAL FEATURE NAMES TO READABLE NAMES
# =========================================================

def make_feature_readable(feature):

    numerical_names = {
        "num__duration_months": "Credit Duration",
        "num__credit_amount": "Credit Amount",
        "num__installment_rate": "Installment Rate",
        "num__residence_since": "Years at Current Residence",
        "num__age": "Age",
        "num__existing_credits": "Number of Existing Credits",
        "num__dependents": "Number of Dependents"
    }

    if feature in numerical_names:
        return numerical_names[feature]

    feature_groups = {
        "checking_account": "Checking Account",
        "credit_history": "Credit History",
        "purpose": "Credit Purpose",
        "savings_account": "Savings Account",
        "employment_duration": "Employment Duration",
        "personal_status_sex": "Personal Status & Sex",
        "other_debtors": "Other Debtors",
        "property": "Property",
        "other_installment_plans": "Other Installment Plans",
        "housing": "Housing",
        "job": "Job",
        "telephone": "Telephone",
        "foreign_worker": "Foreign Worker"
    }

    for column, display_name in feature_groups.items():

        prefix = f"cat__{column}_"

        if feature.startswith(prefix):

            code = feature.replace(prefix, "")

            description = feature_name_mapping.get(
                code,
                code
            )

            return f"{display_name}: {description}"

    return feature


# =========================================================
# FEATURE CONTRIBUTION FUNCTION
# =========================================================

def get_feature_contributions(input_df):

    transformed_input = preprocessor.transform(input_df)

    feature_names = preprocessor.get_feature_names_out()

    coefficients = model.coef_[0]

    if hasattr(transformed_input, "toarray"):
        input_values = transformed_input.toarray()[0]
    else:
        input_values = transformed_input[0]

    contributions = input_values * coefficients

    contribution_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    })

    contribution_df["Absolute Contribution"] = (
        contribution_df["Contribution"].abs()
    )

    contribution_df = contribution_df.sort_values(
        "Absolute Contribution",
        ascending=False
    )

    return contribution_df


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("💳 Credit Scoring System")

st.write(
    "Enter the customer's financial information "
    "to estimate credit risk using a Machine Learning model."
)

st.divider()


# =========================================================
# HUMAN-READABLE INPUT OPTIONS
# =========================================================

checking_options = {
    "Less than 0 DM": "A11",
    "0 to 200 DM": "A12",
    "200 DM or more": "A13",
    "No checking account": "A14"
}

credit_history_options = {
    "No credits taken / all credits paid": "A30",
    "All credits at this bank paid": "A31",
    "Existing credits paid properly": "A32",
    "Delay in paying credits in the past": "A33",
    "Critical account / other credits": "A34"
}

purpose_options = {
    "New car": "A40",
    "Used car": "A41",
    "Furniture / Equipment": "A42",
    "Radio / Television": "A43",
    "Domestic appliances": "A44",
    "Repairs": "A45",
    "Education": "A46",
    "Vacation": "A47",
    "Retraining": "A48",
    "Business": "A49",
    "Other": "A410"
}

savings_options = {
    "Less than 100 DM": "A61",
    "100 to 499 DM": "A62",
    "500 to 999 DM": "A63",
    "1000 DM or more": "A64",
    "Unknown / No savings account": "A65"
}

employment_options = {
    "Unemployed": "A71",
    "Less than 1 year": "A72",
    "1 to 4 years": "A73",
    "4 to 7 years": "A74",
    "7 years or more": "A75"
}

personal_options = {
    "Male - Divorced / Separated": "A91",
    "Female - Divorced / Separated / Married": "A92",
    "Male - Single": "A93",
    "Male - Married / Widowed": "A94",
    "Female - Single": "A95"
}

debtors_options = {
    "None": "A101",
    "Co-applicant": "A102",
    "Guarantor": "A103"
}

property_options = {
    "Real Estate": "A121",
    "Building Society Savings / Life Insurance": "A122",
    "Car or Other Property": "A123",
    "Unknown / No Property": "A124"
}

installment_options = {
    "Bank": "A141",
    "Stores": "A142",
    "None": "A143"
}

housing_options = {
    "Rent": "A151",
    "Own": "A152",
    "For Free": "A153"
}

job_options = {
    "Unemployed / Unskilled Non-resident": "A171",
    "Unskilled Resident": "A172",
    "Skilled Employee / Official": "A173",
    "Management / Self-employed / Highly Qualified": "A174"
}

telephone_options = {
    "None": "A191",
    "Yes - Registered": "A192"
}

foreign_worker_options = {
    "Yes": "A201",
    "No": "A202"
}


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.header("👤 Customer Information")

col1, col2 = st.columns(2)


# =========================================================
# LEFT COLUMN
# =========================================================

with col1:

    checking_label = st.selectbox(
        "Checking Account",
        list(checking_options.keys())
    )

    duration_months = st.number_input(
        "Credit Duration (Months)",
        min_value=1,
        max_value=72,
        value=12
    )

    credit_history_label = st.selectbox(
        "Credit History",
        list(credit_history_options.keys())
    )

    purpose_label = st.selectbox(
        "Purpose",
        list(purpose_options.keys())
    )

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=0,
        value=2500
    )

    savings_label = st.selectbox(
        "Savings Account",
        list(savings_options.keys())
    )

    employment_label = st.selectbox(
        "Employment Duration",
        list(employment_options.keys())
    )

    installment_rate = st.slider(
        "Installment Rate",
        min_value=1,
        max_value=4,
        value=2
    )

    personal_label = st.selectbox(
        "Personal Status & Sex",
        list(personal_options.keys())
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with col2:

    debtors_label = st.selectbox(
        "Other Debtors",
        list(debtors_options.keys())
    )

    residence_since = st.slider(
        "Years at Current Residence",
        min_value=1,
        max_value=4,
        value=2
    )

    property_label = st.selectbox(
        "Property",
        list(property_options.keys())
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    installment_plan_label = st.selectbox(
        "Other Installment Plans",
        list(installment_options.keys())
    )

    housing_label = st.selectbox(
        "Housing",
        list(housing_options.keys())
    )

    existing_credits = st.number_input(
        "Number of Existing Credits",
        min_value=1,
        max_value=4,
        value=1
    )

    job_label = st.selectbox(
        "Job",
        list(job_options.keys())
    )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=1,
        max_value=2,
        value=1
    )

    telephone_label = st.selectbox(
        "Telephone",
        list(telephone_options.keys())
    )

    foreign_worker_label = st.selectbox(
        "Foreign Worker",
        list(foreign_worker_options.keys())
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

if st.button(
    "🔍 Check Credit Risk",
    use_container_width=True
):

    # =====================================================
    # CONVERT UI VALUES TO DATASET CODES
    # =====================================================

    input_data = {

        "checking_account":
            checking_options[checking_label],

        "duration_months":
            duration_months,

        "credit_history":
            credit_history_options[credit_history_label],

        "purpose":
            purpose_options[purpose_label],

        "credit_amount":
            credit_amount,

        "savings_account":
            savings_options[savings_label],

        "employment_duration":
            employment_options[employment_label],

        "installment_rate":
            installment_rate,

        "personal_status_sex":
            personal_options[personal_label],

        "other_debtors":
            debtors_options[debtors_label],

        "residence_since":
            residence_since,

        "property":
            property_options[property_label],

        "age":
            age,

        "other_installment_plans":
            installment_options[installment_plan_label],

        "housing":
            housing_options[housing_label],

        "existing_credits":
            existing_credits,

        "job":
            job_options[job_label],

        "dependents":
            dependents,

        "telephone":
            telephone_options[telephone_label],

        "foreign_worker":
            foreign_worker_options[foreign_worker_label]
    }


    # =====================================================
    # CREATE DATAFRAME
    # =====================================================

    input_df = pd.DataFrame([input_data])


    # =====================================================
    # PREPROCESSING
    # =====================================================

    processed_data = preprocessor.transform(
        input_df
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = model.predict(
        processed_data
    )[0]

    probability = model.predict_proba(
        processed_data
    )[0][1]


    # =====================================================
    # FEATURE CONTRIBUTIONS
    # =====================================================

    contribution_df = get_feature_contributions(
        input_df
    )


    # =====================================================
    # CREDIT ASSESSMENT
    # =====================================================

    st.divider()

    st.header("📊 Credit Assessment")

    result_col1, result_col2, result_col3 = st.columns(3)


    # =====================================================
    # CREDIT STATUS
    # =====================================================

    with result_col1:

        if prediction == 0:

            st.success(
                "✅ Good Credit"
            )

        else:

            st.error(
                "⚠️ Bad Credit"
            )


    # =====================================================
    # RISK LEVEL
    # =====================================================

    with result_col2:

        if prediction == 0:

            st.info(
                "🟢 Risk Level: Low"
            )

        else:

            st.warning(
                "🔴 Risk Level: High"
            )


    # =====================================================
    # BAD CREDIT PROBABILITY
    # =====================================================

    with result_col3:

        st.metric(
            "Bad Credit Probability",
            f"{probability * 100:.2f}%"
        )


    # =====================================================
    # PROBABILITY VISUALIZATION
    # =====================================================

    st.subheader("📈 Risk Probability")

    st.progress(
        min(float(probability), 1.0)
    )

    st.write(
        f"Estimated probability of bad credit: "
        f"**{probability * 100:.2f}%**"
    )


    # =====================================================
    # EXPLAINABLE AI
    # =====================================================

    st.divider()

    st.header("🔎 Prediction Explanation")

    st.write(
        "The following features had the largest "
        "influence on this model's prediction for "
        "the submitted information."
    )


    # =====================================================
    # TOP FEATURES
    # =====================================================

    top_features = contribution_df.head(6).copy()

    top_features["Feature"] = top_features[
        "Feature"
    ].apply(
        make_feature_readable
    )

    top_features["Direction"] = top_features[
        "Contribution"
    ].apply(
        lambda x:
        "Bad Credit"
        if x > 0
        else "Good Credit"
    )

    top_features["Contribution"] = top_features[
        "Contribution"
    ].round(4)


    st.dataframe(
        top_features[
            [
                "Feature",
                "Contribution",
                "Direction"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # POSITIVE AND NEGATIVE CONTRIBUTORS
    # =====================================================

    positive_features = contribution_df[
        contribution_df["Contribution"] > 0
    ].head(3)


    negative_features = contribution_df[
        contribution_df["Contribution"] < 0
    ].head(3)


    exp_col1, exp_col2 = st.columns(2)


    # =====================================================
    # FACTORS TOWARD BAD CREDIT
    # =====================================================

    with exp_col1:

        st.subheader(
            "🔴 Factors toward Bad Credit"
        )

        if len(positive_features) > 0:

            for _, row in positive_features.iterrows():

                readable_name = make_feature_readable(
                    row["Feature"]
                )

                st.write(
                    f"• {readable_name} "
                    f"({row['Contribution']:.3f})"
                )

        else:

            st.write(
                "No strong positive contributors."
            )


    # =====================================================
    # FACTORS TOWARD GOOD CREDIT
    # =====================================================

    with exp_col2:

        st.subheader(
            "🟢 Factors toward Good Credit"
        )

        if len(negative_features) > 0:

            for _, row in negative_features.iterrows():

                readable_name = make_feature_readable(
                    row["Feature"]
                )

                st.write(
                    f"• {readable_name} "
                    f"({abs(row['Contribution']):.3f})"
                )

        else:

            st.write(
                "No strong negative contributors."
            )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.divider()

    st.caption(
        "⚠️ This application is developed for educational "
        "and demonstration purposes as part of a Machine "
        "Learning internship project. The prediction should "
        "not be used as a real-world financial decision."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CodeAlpha Machine Learning Internship | "
    "Credit Scoring Model"
)