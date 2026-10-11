import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------------------------
# Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Insurance Customer Response Predictor",
    page_icon="🚗",
    layout="wide"
)

THRESHOLD = 0.22485247334177014
PROJECT_DIR = Path(__file__).resolve().parent

# -------------------------------------------------
# Load trained model
# -------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(PROJECT_DIR / "insurance_response_model_no_gender.pkl")

model = load_model()

@st.cache_data
def load_test_predictions():
    return pd.read_csv(PROJECT_DIR / "final_test_predictions.csv")

# -------------------------------------------------
# Page
# -------------------------------------------------

st.title("🚗 Insurance Customer Response Predictor")

st.write(
    """
    Predict the likelihood that a customer will respond positively
    to a vehicle insurance cross-sell campaign.
    """
)

st.divider()

st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=20,
        max_value=85,
        value=40
    )

    driving_license = st.selectbox(
        "Driving License",
        ["Yes", "No"]
    )

    previously_insured = st.selectbox(
        "Previously Insured",
        ["No", "Yes"]
    )

    vehicle_age = st.selectbox(
        "Vehicle Age",
        ["< 1 Year", "1-2 Year", "> 2 Years"]
    )

with col2:

    vehicle_damage = st.selectbox(
        "Vehicle Previously Damaged",
        ["Yes", "No"]
    )

    annual_premium = st.number_input(
        "Annual Premium",
        min_value=2630.0,
        value=30000.0
    )

    vintage = st.number_input(
        "Customer Vintage (Days)",
        min_value=10,
        max_value=299,
        value=150
    )

    region_code = st.number_input(
        "Region Code",
        min_value=0,
        max_value=52,
        value=28
    )

    policy_channel = st.number_input(
        "Policy Sales Channel",
        min_value=1,
        max_value=163,
        value=152
    )

# -------------------------------------------------
# Prediction
# -------------------------------------------------

st.divider()

if st.button(
    "Predict Customer Response",
    type="primary",
    width="stretch"
):

    customer = pd.DataFrame({
        "Age": [age],
        "Driving_License": [
            1 if driving_license == "Yes" else 0
        ],
        "Region_Code": [float(region_code)],
        "Previously_Insured": [
            1 if previously_insured == "Yes" else 0
        ],
        "Vehicle_Age": [vehicle_age],
        "Vehicle_Damage": [vehicle_damage],
        "Annual_Premium": [annual_premium],
        "Policy_Sales_Channel": [float(policy_channel)],
        "Vintage": [vintage]
    })

    probability = model.predict_proba(customer)[0, 1]
    prediction = int(probability >= THRESHOLD)

    # --------------------------------------------------
    # Final Prediction
    # --------------------------------------------------

    st.divider()
    st.subheader("Final Prediction")

    if probability >= THRESHOLD:
        st.success("LIKELY TO RESPOND")
        st.markdown(
            f"""
            The model estimates a **{probability * 100:.1f}% response probability**,
            which is **above the {THRESHOLD * 100:.1f}% operating threshold**.
            Therefore, this customer is classified as a **potential responder**
            and may be prioritized for the insurance cross-sell campaign.
            """
        )
    else:
        st.info("LESS LIKELY TO RESPOND")
        st.markdown(
            f"""
            The model estimates a **{probability * 100:.1f}% response probability**,
            which is **below the {THRESHOLD * 100:.1f}% operating threshold**.
            Therefore, this customer is classified as a **lower-priority potential
            responder** for the insurance cross-sell campaign.
            """
        )

    # --------------------------------------------------
    # Prediction Reasoning
    # --------------------------------------------------

    st.subheader("Prediction Reasoning")

    margin = (probability - THRESHOLD) * 100

    if margin >= 0:
        st.write(
            f"The estimated probability is **{margin:.1f} percentage points "
            f"above** the campaign threshold."
        )
    else:
        st.write(
            f"The estimated probability is **{abs(margin):.1f} percentage points "
            f"below** the campaign threshold."
        )

    st.write(
        """
        The model makes this assessment using the customer's insurance,
        vehicle, demographic, regional, and policy-channel information.
        Across the final model, the strongest predictors were:
        """
    )

    st.markdown("""- **Previously Insured**
- **Vehicle Damage**
- **Policy Sales Channel**
- **Age**
- **Vehicle Age**""")

    st.caption(
        "These are overall model-level importance results. They indicate which "
        "variables the model generally relies on most, but they do not prove that "
        "a particular characteristic caused this individual prediction."
    )

    # --------------------------------------------------
    # Prediction Result
    # --------------------------------------------------

    st.divider()
    st.subheader("Customer Response Assessment")

    probability_pct = probability * 100
    threshold_pct = THRESHOLD * 100
    margin_pct = (probability - THRESHOLD) * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Estimated Response",
            f"{probability_pct:.1f}%"
        )

    with col2:
        st.metric(
            "Operating Threshold",
            f"{threshold_pct:.1f}%"
        )

    with col3:
        st.metric(
            "Threshold Margin",
            f"{margin_pct:+.1f} pp"
        )

    st.progress(
        min(max(float(probability), 0.0), 1.0)
    )

    if prediction == 1:

        st.success(
            "HIGHER CAMPAIGN PRIORITY"
        )

        st.write(
            "The model-estimated response probability is above "
            "the validation-selected operating threshold. "
            "This customer may be considered for cross-sell targeting."
        )

    else:

        st.info(
            "LOWER CAMPAIGN PRIORITY"
        )

        st.write(
            "The model-estimated response probability is below "
            "the validation-selected operating threshold."
        )

    st.caption(
        "This output represents a statistical decision-support score "
        "and should not be interpreted as certainty that the customer "
        "will or will not purchase insurance."
    )

    # --------------------------------------------------
    # Model Insights
    # --------------------------------------------------

    st.divider()
    st.subheader("Model Insights")

    st.write(
        """
        The following variables had the strongest contribution to
        the model's predictive performance based on permutation
        importance analysis.
        """
    )

    importance_data = pd.DataFrame({
        "Feature": [
            "Previously Insured",
            "Vehicle Damage",
            "Policy Sales Channel",
            "Age",
            "Vehicle Age",
            "Region Code",
            "Annual Premium",
            "Vintage",
            "Driving License"
        ],
        "Importance": [
            0.110383,
            0.075694,
            0.068484,
            0.067505,
            0.040468,
            0.018410,
            0.002018,
            0.000493,
            0.000318
        ]
    })

    st.bar_chart(
        importance_data,
        x="Feature",
        y="Importance",
        horizontal=True
    )

    st.caption(
        "Permutation importance measures how much model performance "
        "decreases when a feature is randomly disrupted. Higher values "
        "indicate greater predictive reliance. These values do not "
        "represent causal effects."
    )

    # --------------------------------------------------
    # Business Decision Support
    # --------------------------------------------------

    st.divider()
    st.subheader("Business Decision Support")

    if prediction == 1:
        st.write(
            """
            **Suggested operational use:** This customer falls above the
            selected campaign threshold and could be included in a
            prioritized cross-sell contact list.

            Marketing teams could use this score together with campaign
            capacity, contact cost, customer preferences, and business
            rules when deciding whom to contact.
            """
        )
    else:
        st.write(
            """
            **Suggested operational use:** This customer falls below the
            selected campaign threshold and may receive lower priority
            when campaign resources are limited.

            The score should support—not replace—business judgment and
            other customer-management considerations.
            """
        )

    st.warning(
        "The model identifies statistical patterns and predictive "
        "associations. It does not establish that any individual "
        "customer characteristic causes a response."
    )

    with st.expander("View customer data"):
        st.dataframe(
            customer,
            width="stretch"
        )

st.divider()

st.caption(
    "Decision-support prototype for insurance cross-sell targeting. "
    "Predictions represent model estimates and should support, "
    "not replace, business judgment."
)

cohort_predictions = load_test_predictions()
cohort_probabilities = cohort_predictions["Response_Probability"]
expected_responses = int(round(cohort_probabilities.sum()))
expected_response_rate = cohort_probabilities.mean()
above_threshold = cohort_probabilities >= THRESHOLD

st.divider()
st.subheader("Overall Test-Set Outlook")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Customers Assessed", f"{len(cohort_probabilities):,}")

with col2:
    st.metric("Estimated Responses", f"About {expected_responses:,}")

with col3:
    st.metric("Estimated Response Rate", f"{expected_response_rate:.1%}")

st.caption(
    f"{int(above_threshold.sum()):,} customers "
    f"({above_threshold.mean():.1%}) are above the campaign threshold. "
    "This is a model estimate for the unlabeled test set, not an observed "
    "response rate; actual outcomes are not available."
)