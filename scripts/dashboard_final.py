import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from lib.db import get_engine, get_cities_with_recent_data, load_city_data
from lib.models import train_and_forecast, train_and_validate
from lib.metrics import evaluate_forecast, classify_model_quality
from lib.config import TRAIN_CUTOFF
from lib.charts import plot_history, plot_forecast, plot_monthly_breakdown, plot_validation

warnings.filterwarnings('ignore')

st.set_page_config(page_title="India AQI Forecast", page_icon="🏭", layout="wide")
st.title("🏭 India AQI Forecasting Dashboard")
st.markdown("**Interactive forecasts for Indian cities (2015-2030)**")

engine = get_engine()

@st.cache_data
def get_cities():
    return get_cities_with_recent_data(engine)

cities = get_cities()

st.sidebar.header("📊 Controls")
selected_city = st.sidebar.selectbox("Select City", cities)

@st.cache_data
def get_city_data(city):
    return load_city_data(engine, city)

df = get_city_data(selected_city)

col1, col2, col3, col4 = st.columns(4)
col1.metric("City", selected_city)
col2.metric("Data Points", f"{len(df):,}")
col3.metric("Date Range", f"{df['ds'].min().date()} to {df['ds'].max().date()}")
col4.metric("Latest AQI", f"{df['y'].iloc[-1]:.0f}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📈 Historical", "🔮 Forecast", "✅ Validation"])

with tab1:
    st.header(f"Historical Data: {selected_city}")

    df_copy = df.copy()
    df_copy['year'] = df_copy['ds'].dt.year
    yearly_stats = df_copy.groupby('year')['y'].agg(['mean', 'min', 'max']).round(1)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_history(ax, df_copy, selected_city)
        st.pyplot(fig)

    with col2:
        st.subheader("Yearly Statistics")
        st.dataframe(yearly_stats)

with tab2:
    st.header(f"Forecast to 2030: {selected_city}")

    with st.spinner('Training Prophet model...'):
        model, forecast = train_and_forecast(df)

        pred_2030 = forecast[forecast['ds'].dt.year == 2030]['yhat'].mean()
        current_avg = df[df['ds'] >= '2024-01-01']['y'].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Current (2024)", f"{current_avg:.1f}")
        col2.metric("2030 Prediction", f"{pred_2030:.1f}")
        trend = "📉 Improving" if pred_2030 < current_avg else "📈 Worsening"
        col3.metric("Trend", trend)

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_forecast(ax, df, forecast, selected_city)
        st.pyplot(fig)

        st.subheader("2030 Monthly Forecast")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        plot_monthly_breakdown(ax2, forecast, 2030)
        st.pyplot(fig2)

with tab3:
    st.header("Model Validation")

    train = df[df['ds'] < TRAIN_CUTOFF]
    test = df[df['ds'] >= TRAIN_CUTOFF]

    if len(test) > 30:
        with st.spinner('Validating...'):
            _, _, results = train_and_validate(train, test)
            metrics = evaluate_forecast(results)
            mape = metrics['mape']

            col1, col2, col3 = st.columns(3)
            col1.metric("MAPE", f"{mape:.1f}%")
            col2.metric("RMSE", f"{metrics['rmse']:.1f}")
            col3.metric("Test Period", "2023-2024")

            quality, _ = classify_model_quality(mape)
            st.info(
                f"**Model Quality: {quality}**\n\n"
                f"MAPE of {mape:.1f}% means predictions are within "
                f"±{mape:.0f}% of actual values"
            )

            fig, ax = plt.subplots(figsize=(10, 5))
            plot_validation(ax, results, mape, selected_city)
            st.pyplot(fig)
    else:
        st.warning("Insufficient 2023-2024 data for validation")

st.markdown("---")
st.markdown("Built with **Streamlit + Prophet** | Data: India Air Quality (2015-2024)")
