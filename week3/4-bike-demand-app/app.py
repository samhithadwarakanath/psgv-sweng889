import streamlit as st

from src.llm_service import LLMServiceError, get_explanation
from src.predict import MAX_TEMPERATURE, MIN_TEMPERATURE, predict_demand

st.set_page_config(page_title="Bike Demand Predictor", page_icon="🚲")
st.title("🚲 Hourly Bike Rental Demand Predictor")
st.write("Predict hourly bike rental demand and get an AI-generated explanation.")

with st.form("prediction_form"):
    date = st.date_input("Date")
    time = st.time_input("Time")
    temperature = st.number_input(
        "Temperature (°C)", min_value=-50.0, max_value=60.0, value=20.0, step=0.5
    )
    is_holiday = st.toggle("Holiday?", value=False)
    submitted = st.form_submit_button("Predict Demand")

if submitted:
    errors = []
    # date_input/time_input always return a value in this form's default configuration,
    # but we check defensively in case that ever changes.
    if date is None:
        errors.append("Please select a date.")
    if time is None:
        errors.append("Please select a time.")
    if temperature is None:
        errors.append("Please enter a temperature.")
    elif not (MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE):
        errors.append(
            f"Temperature must be between {MIN_TEMPERATURE}°C and {MAX_TEMPERATURE}°C "
            "(the range the model was trained on)."
        )

    if errors:
        for error in errors:
            st.error(error)
    else:
        try:
            prediction = predict_demand(date, time, temperature, is_holiday)
        except Exception as exc:
            st.error(f"Could not generate a prediction: {exc}")
        else:
            st.success(f"Predicted demand: **{prediction} bikes**")

            with st.spinner("Generating AI explanation..."):
                try:
                    explanation = get_explanation(
                        prediction, date, time, temperature, is_holiday
                    )
                except LLMServiceError as exc:
                    st.warning(f"AI explanation is unavailable: {exc}")
                else:
                    st.subheader("AI Explanation")
                    st.write(explanation)
