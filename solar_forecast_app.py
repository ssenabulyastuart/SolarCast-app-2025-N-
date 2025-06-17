import tensorflow as tf  # Add this if not already present

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ✅ MUST BE FIRST Streamlit command
st.set_page_config(page_title="Solar Output Predictor", layout="wide")

# ===============================
# 🔐 Password Authentication
# ===============================
PASSWORD = "SolarCast-2025"
st.sidebar.title("🔒 Login Required")
user_password = st.sidebar.text_input("Enter Access Password", type="password")
if user_password != PASSWORD:
    st.sidebar.warning("Invalid password.")
    st.stop()

# ===============================
# 🎨 App UI and Title
# ===============================
st.title("🔆 SOLARCAST: NEXT 12_HOUR PREDICTOR")
st.markdown("Upload your CSV file to forecast the next 12 hours of solar output.")
st.markdown("<h4 style='text-align: center; color: gray;'>FYP by <b>Stuart Ssenabulya</b> and <b>Juliet Tusabe</b></h4>", unsafe_allow_html=True)

# ===============================
# 📁 File Uploads and Date Inputs
# ===============================
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("📁 Upload Solar Data (CSV)", type=["csv"])
with col2:
    true_values_file = st.file_uploader("📂 Upload Actual 12-Hour Output (Optional)", type=["csv"])

start_date = st.date_input("📅 Dataset Start Date", pd.to_datetime("2021-05-01"))
end_date = st.date_input("📅 Dataset End Date (used for filtering only)", pd.to_datetime("2024-07-30"))
prediction_date = st.date_input("📅 Prediction Date (6 AM to 6 PM forecast)", pd.to_datetime("2024-07-31"))

# ===============================
# 🔧 Paths and Loaders
# ===============================
MODEL_PATH = "optimized_model3.h5"
SCALER_PATH = "scaler_model3.pkl"
WINDOW_SIZE = 24

@st.cache_resource
def load_model_and_scaler():
    if not os.path.exists("final_model"):
        st.error("❌ Model folder not found. Ensure 'final_model/' is in the app directory.")
        st.stop()
    if not os.path.exists("scaler_model3.pkl"):
        st.error("❌ Scaler not found. Ensure 'scaler_model3.pkl' is in this directory.")
        st.stop()

    model = tf.keras.models.load_model("final_model")  # Load from folder
    scaler = joblib.load("scaler_model3.pkl")
    return model, scaler


model, scaler = load_model_and_scaler()

# ===============================
# 🧹 Preprocessing
# ===============================
def preprocess(df, scaler, start_date):
    try:
        if not all(col in df.columns for col in ['date', 'solar output']):
            st.error("CSV must contain 'date' and 'solar output' columns.")
            return None, None, None

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date', 'solar output'], inplace=True)
        df = df[df['date'] >= pd.to_datetime(start_date)].copy()

        if not df['date'].is_monotonic_increasing:
            st.warning("⚠️ 'date' column is not sorted. Results may be inaccurate.")

        df.rename(columns={'date': 'ds', 'solar output': 'y'}, inplace=True)
        df = df.set_index('ds')
        df = df.between_time('06:00', '18:00')
        df.reset_index(inplace=True)

        values = df['y'].values.reshape(-1, 1)
        scaled = scaler.transform(values)

        if len(scaled) < WINDOW_SIZE:
            st.error("⛔ Not enough rows (need at least 24 records after filtering).")
            return None, None, None

        last_24 = scaled[-WINDOW_SIZE:]
        X_input = last_24.reshape((1, WINDOW_SIZE, 1))

        return X_input, df, values[-12:] if len(values) >= 12 else values

    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        return None, None, None

# ===============================
# 🔮 Prediction
# ===============================
def predict_next_12_hours(model, scaler, X_input):
    preds_scaled = []
    current_input = X_input.copy()

    for _ in range(12):
        next_pred = model.predict(current_input, verbose=0)
        preds_scaled.append(next_pred[0][0])
        current_input = np.append(current_input[:, 1:, :], [[[next_pred[0][0]]]], axis=1)

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_unscaled = scaler.inverse_transform(preds_scaled)
    return preds_unscaled.flatten()

# ===============================
# 📊 Evaluation
# ===============================
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nrmse = rmse / (max(y_true) - min(y_true))
    return rmse, mae, r2, nrmse

def evaluate_segments(y_true, y_pred):
    return (
        mean_absolute_error(y_true[:4], y_pred[:4]),
        mean_absolute_error(y_true[4:8], y_pred[4:8]),
        mean_absolute_error(y_true[8:12], y_pred[8:12])
    )

# ===============================
# 🚀 App Main Logic
# ===============================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    X_input, df_cleaned, y_true_last12 = preprocess(df, scaler, start_date)

    if X_input is not None:
        y_pred_12 = predict_next_12_hours(model, scaler, X_input)

        # Build 6AM–6PM forecast timestamps for the prediction date
        future_dates = pd.date_range(
            start=pd.to_datetime(prediction_date.strftime("%Y-%m-%d") + " 06:00"),
            periods=12, freq='h'
        )

        forecast_df = pd.DataFrame({
            'Hourly Timestamp': future_dates,
            'Predicted Solar Output(MW)': y_pred_12
        })

        st.success("✅ Forecast complete!")
        st.subheader("12-Hour Solar Output Forecast")
        st.dataframe(forecast_df)

        # ========== 📈 Forecast Plot and Evaluation ==========
        st.subheader("Forecast Visualization")
        plt.figure(figsize=(10, 4))
        plt.plot(future_dates, y_pred_12, marker='o', color='orange', label='Predicted')

        y_true_uploaded = None
        if true_values_file is not None:
            try:
                actual_df = pd.read_csv(true_values_file)

                if 'solar output' not in actual_df.columns:
                    st.warning("⚠️ 'solar output' column missing in actuals file.")
                else:
                    y_true_uploaded = actual_df['solar output'].dropna().values[:12]

                    if len(y_true_uploaded) < 12:
                        st.warning("⚠️ Less than 12 actual values provided.")
                    else:
                        plt.plot(future_dates, y_true_uploaded, marker='x', color='blue', label='Actual (from uploaded file)')

            except Exception as e:
                st.warning("⚠️ Could not load actuals file.")
                st.text(str(e))

        plt.title("LSTM Solar Output Forecast (July 31,2024)")
        plt.xlabel("Timestamp")
        plt.ylabel("Solar Output")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

        # 📊 Evaluation block if actuals available
        if y_true_uploaded is not None and len(y_true_uploaded) == 12:
            rmse, mae, r2, nrmse = evaluate(y_true_uploaded, y_pred_12)
            mae1, mae2, mae3 = evaluate_segments(y_true_uploaded, y_pred_12)

            st.subheader("Model Evaluation (12-hour horizon)")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"MAE: {mae:.4f}")
            st.write(f"R² Score: {r2:.4f}")
            st.write(f"nRMSE: {nrmse:.4f}")

            st.markdown("#### MAE by Prediction Horizon:")
            st.write(f"- 1–4 hours: **{mae1:.4f}**")
            st.write(f"- 5–8 hours: **{mae2:.4f}**")
            st.write(f"- 9–12 hours: **{mae3:.4f}**")
# ===============================
# 🎨 Custom CSS Styling
# ===============================
custom_css = """
<style>
    /* Background color or gradient */
    body {
        background: linear-gradient(to right, #f0f2f6, #ffffff);
    }

    /* Title Styling */
    .stApp h1 {
        font-family: 'Arial Black', sans-serif;
        color: #ff9900;
        text-align: center;
    }

    /* Subtitle and headers */
    .stApp h4, .stApp h2, .stApp h3 {
        color: #333333;
        text-align: center;
    }

    /* Dataframe & metrics */
    .stDataFrame {
        border: 2px solid #ddd;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Password input section */
    section[data-testid="stSidebar"] {
        background-color: #f6f6f9;
    }

    /* Improve button visibility */
    button {
        background-color: #ff9900 !important;
        color: white !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.sidebar.image("logo.png", use_container_width=True)




        



