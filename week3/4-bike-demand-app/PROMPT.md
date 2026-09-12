# AI-Assisted Development Prompts

This file documents the main AI-assisted prompts used to build the bike-demand
prediction model, following the 11-stage AI-Assisted Machine Learning Lifecycle
demo (Google Colab). Each stage below reflects one prompt/interaction with the
AI assistant, run one stage at a time as required by the exercise.

Dataset: Seoul Bike Sharing Demand (UCI Machine Learning Repository)
https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

---

## Stage 1: Data Collection

**Prompt used:**
"Implement the code to load the Seoul Bike Sharing Demand dataset into a
pandas DataFrame from this source: [UCI dataset link]. Show the first 5 rows,
the DataFrame shape, and the column names. Keep the code simple. Only
implement this stage."

**What I used:**
Used the generated approach as-is, downloading the dataset directly from
UCI's public zip URL and extracting the CSV in-memory, rather than manually
uploading the file to Colab. Confirmed the dataset loaded correctly: 8760
rows, 14 columns.

---

## Stage 2: Problem Definition and Model Requirements

**Prompt used:**
"Analyze this dataset for a machine learning regression problem. Identify
the target variable, the useful input features, the data types, possible
data leakage, and how we should split the data. Also suggest simple success
criteria for the model. Do not train a model yet. Keep the explanation
short."

**What I used:**
Used the AI's analysis to confirm:
- Target variable: `Rented Bike Count` (regression problem)
- Key features: Hour, weather variables, Seasons, Holiday, Functioning Day
- Noted `Dew point temperature(°C)` as highly correlated with Temperature
  (potential multicollinearity, not leakage)
- Adopted a chronological (not random) train/test split, since the data is
  time-series-like
- Adopted MAE and R² as the primary success criteria

---

## Stage 3: Data Exploration

**Prompt used:**
"Implement a small exploratory analysis for this DataFrame. Show summary
statistics, missing values, duplicate rows, the target distribution, and a
few simple relationships with Rented Bike Count. Use only a few useful
plots. Add short comments explaining what each check tells us. Only
implement this stage."

**What I used:**
Used the generated code as-is. Confirmed no missing values and no duplicate
rows. The generated plots showed:
- A right-skewed distribution of Rented Bike Count (mean 704.6, median 504.5)
- Clear commuting-hour demand peaks (a smaller peak around 7-8 AM, a larger
  peak around 6 PM)
- A positive relationship between temperature and demand, with almost no
  high-demand hours below ~5°C

---

## Stage 4: Data Preparation

**Prompt used:**
"Implement the code to clean and transform this DataFrame for modeling.
Handle the Date column, missing values if any, duplicate rows if any, and
categorical columns. Separate features X from target y. Use simple
functions and add a short comment explaining the goal of each function.
Only implement this stage."

**What I used:**
Used the generated functions (clean_data, process_date, encode_categoricals,
split_features_target) as-is. Encoded Holiday and Functioning Day as binary
values and one-hot encoded Seasons. Kept Date as a datetime column at this
stage rather than dropping it immediately, so it could be used to engineer
time-based features in the next stage.

---

## Stage 5: Feature Engineering

**Prompt used:**
"Implement simple feature engineering for this bike demand problem. Create
useful features from Date and Hour, such as year, month, day of week,
weekend, and time-of-day information. Keep only features that have a clear
reason to help prediction. Briefly explain why each new feature may help.
Only implement this stage."

**What I used:**
Used the generated features as-is: `month`, `day_of_week`, `is_weekend`,
`is_peak_hour` (7-8 AM and 5-7 PM), and `is_night` (midnight-5 AM). Each
feature was directly justified by patterns observed during Stage 3
exploration (e.g., the hourly demand plot motivated `is_peak_hour` and
`is_night`). Dropped the raw Date column after extracting these features.
Final feature count: 19.

---

## Stage 6: Model Training

**Prompt used:**
"Choose one simple scikit regression model for this dataset and explain in
2 or 3 sentences why it is a good choice for this demo. Then implement the
train/test split and model training code. Keep the code short and readable.
Do not evaluate the model yet. Only implement this stage."

**What I used:**
Accepted the AI's recommendation of GradientBoostingRegressor, justified by
its ability to model nonlinear relationships and feature interactions
without requiring feature scaling. Used a chronological 80/20 train/test
split (7008 train / 1752 test rows) to avoid leaking future information
into training, consistent with the Stage 2 decision.

---

## Stage 7: Model Evaluation

**Prompt used:**
"Evaluate the trained regression model. Use MAE, RMSE, and R-squared.
Briefly explain why each metric is useful. Implement the code to calculate
the metrics and create one simple plot comparing actual and predicted
values. Then help me interpret the results in plain English. Only
implement this stage."

**What I used:**
Used the generated evaluation code as-is. Initial results: MAE 256.00,
RMSE 356.18, R² 0.6627. Reviewed the actual-vs-predicted scatter plot with
the AI's help and identified that the model systematically under-predicts
at higher demand levels (1000+ bikes), and occasionally predicts negative
values, which are not physically meaningful for a count of bike rentals.

---

## Stage 8: Model Improvement

**Prompt used:**
"Suggest one simple way to improve the current model, such as trying a
stronger regression algorithm or tuning a small number of parameters.
Explain the reason, implement the change, and compare the new evaluation
metrics with the previous model. Keep it simple. Only implement this
stage."

**What I used:**
Accepted the AI's two-part suggestion: (1) tune hyperparameters
(n_estimators=200, learning_rate=0.1, max_depth=7) and (2) clip negative
predictions to zero, since bike demand cannot be negative. This improved
results meaningfully: MAE dropped from 256.00 to 196.95, RMSE dropped from
356.18 to 279.51, and R² improved from 0.6627 to 0.7923.

---

## Stage 9: Deployment and Prediction

**Prompt used:**
"Implement the code to save and download the trained model from Google
Colab. Save the trained model (e.g. regression.pkl) and data
pre-processing steps (e.g. scaler.pkl). Then show how to load the saved
model and make one prediction using sample input data. Explain briefly how
this model could be integrated into another application, for example
through a Python API. Keep the example simple. Only implement this stage."

**What I used:**
Used the generated save/load/predict code, saving the model as
`bike_demand_model.joblib` (no separate scaler was needed since
Gradient Boosting does not require feature scaling). While testing a
sample prediction, I noticed the model predicted 192 bikes for a row where
`Functioning Day = 0` (system not operating), even though actual demand
was 0. This was not something the AI flagged on its own — I identified it
by inspecting the sample input and output rather than assuming the
prediction was correct. Decided the final application will assume the
bike-share system is always operating (Functioning Day = "Yes"), which
avoids this edge case for the assignment's scope.

---

## Stage 10: Monitoring

**Prompt used:**
"Explain what we need to monitor after deploying this bike demand model.
Cover prediction quality, data drift, feature/data quality, errors,
latency, and retraining. For each item, say what to observe and why it
matters. Do not implement a full monitoring platform. If useful, provide
only a very small Python example for logging predictions."

**What I used:**
Used the AI's explanation of monitoring concerns (prediction quality
drift, feature drift, malformed input, latency, and periodic retraining)
as reference material for this documentation. Included the small optional
logging code snippet as an example but did not run or integrate it, since
it was outside the scope of the assignment's application requirements.

---

## Stage 11: Lifecycle Review

**Prompt used:**
"Summarize the machine learning lifecycle we just completed: data
collection, requirements, exploration, preparation, feature engineering,
training, evaluation, improvement, deployment, and monitoring. For each
stage, give one sentence explaining how AI helped us. Keep the summary
concise and suitable for software engineers."

**What I used:**
Used the generated summary to connect the full lifecycle into a coherent
narrative, and used it as the basis for the model-development portion of
this PROMPT.md file.

---

## Notes

- The exact prompts above were paraphrased slightly to fit this
  documentation format; the intent and scope of each prompt matches what
  was actually run, one stage at a time, in Google Colab.
- Prompts for the Streamlit application itself (UI, LLM integration,
  failure handling) are documented separately below, once that part of the
  assignment is implemented.


# Claude Prompts

**What I used:**
1.
Go through the files and Build a simple Streamlit application in this folder that predicts hourly bike rental demand and adds an LLM-generated explanation. Requirements:

  src/predict.py: loads models/bike_demand_model.joblib (a GradientBoostingRegressor trained on 19 features: Hour, Temperature(°C), Humidity(%), Wind speed (m/s), Visibility (10m), Dew point temperature(°C), Solar Radiation (MJ/m2), Rainfall(mm), Snowfall (cm), Holiday, Functioning Day, Seasons_Spring, Seasons_Summer, Seasons_Winter, month, day_of_week, is_weekend, is_peak_hour, is_night — in that exact order). Provide a predict_demand(date, time, temperature, is_holiday) function that derives all 19 features from just those 4 inputs (using fixed reasonable defaults for Humidity=60, Wind speed=2.0, Visibility=1500, Dew point temperature=Temperature-5, Solar Radiation=0.5, Rainfall=0, Snowfall=0, Functioning Day=1 always), clips negative predictions to 0, and returns the predicted integer bike count.
  src/llm_service.py: sends a prompt to a local Ollama service (URL and model name from environment variables OLLAMA_URL and OLLAMA_MODEL, defaulting to http://localhost:11434 and llama3.2:1b) asking it to generate a short plain-language explanation and operational recommendation based on the predicted demand number and the input conditions. Must handle connection failures or timeouts gracefully by raising a clear custom exception rather than crashing.
  app.py: a Streamlit UI with a date picker, time picker, temperature number input, and holiday yes/no toggle. On submit, calls predict_demand, displays the prediction, then calls the LLM service and displays its response. If the LLM call fails, the UI must still show the ML prediction with a clear message that the AI explanation is unavailable. If required input is missing or invalid, show a clear error message instead of crashing.
  Keep UI code, prediction logic, and LLM logic in separate files (no business logic in app.py).
  Add a requirements.txt with streamlit, joblib, scikit-learn, pandas, numpy, requests.
  Add a Dockerfile and docker-compose.yml that runs this app alongside an Ollama service, similar in style to week3/1-local-llm/docker-compose.yml in this repo (Ollama on port 11434, this app on Streamlit's default port 8501).

  Do not over-engineer this — keep it as simple as possible while meeting these requirements.


2.
Add a startup step to docker-compose.yml (or an entrypoint script) that automatically pulls the llama3.2:1b model into the Ollama container on startup, similar to how week3/1-local-llm/app.py waits for Ollama and pulls the model, so the LLM works on first docker compose up --build without a manual docker exec step.

3.
i think we should shorten the explanation, since it takes a long time and it got cut off, Predictor
  Predict hourly bike rental demand and get an AI-generated explanation.

  Date

  2026
  /
  09
  /
  13

  09/13/2026
  Time

  �
  19
  :
  00
  �
  Temperature (°C)

  24.00

  Holiday?

  Predicted demand: 1832 bikes

  AI Explanation
  The high demand for bike rentals might be due to the pleasant weather on 2026-09-13, making people more inclined to cycle instead of driving. The model's prediction of 1832 rentals is likely a rough estimate of the actual number of people who might rent bikes during this hour, as some people might have already booked their rentals or not have realized they can rent bikes that day. To optimize operations, the bike-share operator could consider promoting their bikes in online ride-sharing platforms, such

  4.
  Add input validation to app.py: temperature must be between -20°C and 40°C (the range the model was trained on), matching realistic Seoul weather. If the user enters a value outside this range, show a clear error message instead of running the prediction. Also validate that date and time are provided (Streamlit's date_input/time_input may already guarantee this, but confirm and handle the missing case defensively).