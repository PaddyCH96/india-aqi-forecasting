import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from lib.db import get_engine, get_cities_with_data_summary, load_city_data
from lib.models import train_and_forecast, train_and_validate
from lib.metrics import evaluate_forecast, classify_model_quality
from lib.config import TRAIN_CUTOFF
from lib.charts import (
    plot_history,
    plot_forecast,
    plot_monthly_breakdown,
    plot_validation,
    plot_scatter_accuracy,
    plot_comparison_barh,
)

warnings.filterwarnings('ignore')

st.set_page_config(page_title="India AQI Forecast", page_icon="🏭", layout="wide")
st.title("🏭 India AQI Forecasting Dashboard")
st.markdown("**Interactive forecasts for Indian cities (2015-2030)**")

engine = get_engine()

@st.cache_data
def get_cities():
    return get_cities_with_data_summary(engine)

cities_df = get_cities()
city_list = cities_df['city'].tolist()

st.sidebar.header("📊 Controls")
st.sidebar.write(f"Found {len(city_list)} cities with data")
st.sidebar.caption("Min 1000 days required")

selected_city = st.sidebar.selectbox(
    "🌆 Select City",
    city_list,
    index=0 if 'Hyderabad' not in city_list else city_list.index('Hyderabad')
)

city_info = cities_df[cities_df['city'] == selected_city].iloc[0]
st.sidebar.write("**Data available:**")
st.sidebar.write(f"• Days: {city_info['days']:,}")
st.sidebar.write(f"• From: {city_info['start_date']}")
st.sidebar.write(f"• To: {city_info['end_date']}")
st.sidebar.markdown("---")
st.sidebar.markdown("**About:** AQI forecasting using Prophet")

@st.cache_data
def get_city_data(city):
    return load_city_data(engine, city)

df = get_city_data(selected_city)

st.markdown(f"## Currently viewing: **{selected_city}**")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Date Range", f"{df['ds'].min().date()} to {df['ds'].max().date()}")
col3.metric("Average AQI", f"{df['y'].mean():.0f}")
col4.metric("Latest AQI", f"{df['y'].iloc[-1]:.0f}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Historical", "🔮 2030 Forecast", "✅ Validation", "📊 Comparison"])

with tab1:
    st.header(f"Historical AQI Data: {selected_city}")

    df_copy = df.copy()
    df_copy['year'] = df_copy['ds'].dt.year
    yearly_stats = df_copy.groupby('year')['y'].agg(['mean', 'min', 'max', 'count']).round(1)
    yearly_stats.columns = ['Mean', 'Min', 'Max', 'Days']

    col1, col2 = st.columns([3, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_history(ax, df_copy, selected_city, show_covid=True)
        st.pyplot(fig)

        st.info("""
        **Chart Guide:**
        - Blue line: Daily AQI measurements
        - Red dots: Yearly averages
        - Gray shading: COVID-19 lockdown period (reduced traffic/industrial activity)
        - Orange dashed: Moderate threshold (100)
        - Red dashed: Poor threshold (200)
        """)

    with col2:
        st.subheader("Yearly Statistics")
        st.dataframe(yearly_stats, use_container_width=True)

        if len(yearly_stats) > 1:
            first_year_avg = yearly_stats.iloc[0]['Mean']
            last_year_avg = yearly_stats.iloc[-1]['Mean']
            change = last_year_avg - first_year_avg
            trend = "improving 📉" if change < 0 else "worsening 📈"
            st.metric("Overall Trend", f"{change:+.1f} points", trend)

with tab2:
    st.header(f"Forecast to 2030: {selected_city}")

    with st.spinner(f'Training Prophet model for {selected_city}...'):
        model, forecast = train_and_forecast(df)

        pred_2030 = forecast[forecast['ds'].dt.year == 2030]['yhat'].mean()
        pred_2030_lower = forecast[forecast['ds'].dt.year == 2030]['yhat_lower'].mean()
        pred_2030_upper = forecast[forecast['ds'].dt.year == 2030]['yhat_upper'].mean()

        current_2024 = (
            df[df['ds'] >= '2024-01-01']['y'].mean()
            if len(df[df['ds'] >= '2024-01-01']) > 0
            else df['y'].tail(365).mean()
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current (2024)", f"{current_2024:.1f}")
        col2.metric("2030 Estimate", f"{pred_2030:.1f}")
        col3.metric("Range (95% CI)", f"{pred_2030_lower:.0f} - {pred_2030_upper:.0f}")

        change = pred_2030 - current_2024
        trend_icon = "📉 Improving" if change < 0 else "📈 Worsening"
        col4.metric("Trend", trend_icon, f"{abs(change):.1f} points")

        fig, ax = plt.subplots(figsize=(14, 7))
        plot_forecast(ax, df, forecast, selected_city)
        st.pyplot(fig)

        st.subheader("2030 Monthly Breakdown")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        plot_monthly_breakdown(ax2, forecast, 2030)
        st.pyplot(fig2)

with tab3:
    st.header(f"Model Validation: {selected_city}")

    train = df[df['ds'] < TRAIN_CUTOFF]
    test = df[df['ds'] >= TRAIN_CUTOFF]

    if len(test) > 30:
        with st.spinner('Running validation...'):
            _, _, results = train_and_validate(train, test)
            metrics = evaluate_forecast(results)
            mape = metrics['mape']

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAPE", f"{mape:.1f}%")
            col2.metric("RMSE", f"{metrics['rmse']:.1f}")
            col3.metric("MAE", f"{metrics['mae']:.1f}")
            col4.metric("Test Days", f"{len(test)}")

            quality, color = classify_model_quality(mape)
            st.markdown(
                f"<h3 style='color: {color}'>Model Quality: {quality}</h3>",
                unsafe_allow_html=True,
            )

            st.info(f"""
            **Interpretation:**
            - **MAPE of {mape:.1f}%** means predictions are typically within ±{mape:.0f}% of actual values
            - For a prediction of 100 AQI, expect actual value between **{100 * (1 - mape / 100):.0f} and {100 * (1 + mape / 100):.0f}**
            - Model trained on {len(train)} days, tested on {len(test)} days (2023-2024)
            """)

            fig, ax = plt.subplots(figsize=(12, 6))
            plot_validation(ax, results, mape, selected_city)
            st.pyplot(fig)

            fig2, ax2 = plt.subplots(figsize=(8, 8))
            plot_scatter_accuracy(ax2, results, mape)
            st.pyplot(fig2)
    else:
        st.warning(f"⚠️ Insufficient 2023-2024 data for {selected_city} validation")
        st.info("Need at least 30 days of 2023-2024 data to validate model performance.")

with tab4:
    st.header("Multi-City Comparison")
    st.markdown("Compare forecast accuracy across all cities:")

    comparison_data = []
    for city in city_list[:10]:
        city_df = get_city_data(city)
        train_c = city_df[city_df['ds'] < TRAIN_CUTOFF]
        test_c = city_df[city_df['ds'] >= TRAIN_CUTOFF]

        if len(test_c) > 30:
            try:
                _, _, res_c = train_and_validate(train_c, test_c)
                met_c = evaluate_forecast(res_c)
                mape_c = met_c['mape']
                comparison_data.append({
                    'City': city,
                    'MAPE (%)': round(mape_c, 1),
                    '2024 Avg': round(
                        city_df[city_df['ds'] >= '2024-01-01']['y'].mean(), 1
                    ),
                    'Data Quality': 'High' if mape_c < 20 else 'Medium' if mape_c < 30 else 'Low',
                })
            except Exception:
                pass

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data).sort_values('MAPE (%)')
        st.dataframe(comp_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_comparison_barh(ax, comp_df)
        st.pyplot(fig)

        st.info(
            f"**{selected_city} ranking:** "
            f"Currently viewing one of {len(comp_df)} cities with validation data"
        )
    else:
        st.info("Run validation on individual cities to see comparison")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Built with Streamlit + Prophet | India AQI Forecasting Project | 2015-2030</small>
</div>
""", unsafe_allow_html=True)
