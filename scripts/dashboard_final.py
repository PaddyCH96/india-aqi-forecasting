import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="India AQI Forecast", page_icon="🏭", layout="wide")

st.title("🏭 India AQI Forecasting Dashboard")
st.markdown("**Interactive forecasts for Indian cities (2015-2030)**")

# Database connection
engine = create_engine('postgresql://postgres@localhost:5432/india_air_quality')

@st.cache_data
def get_cities():
    cities_df = pd.read_sql("""
        SELECT DISTINCT city FROM city_day 
        WHERE aqi IS NOT NULL AND date >= '2024-01-01'
        ORDER BY city
    """, engine)
    return cities_df['city'].tolist()

cities = get_cities()

st.sidebar.header("📊 Controls")
selected_city = st.sidebar.selectbox("Select City", cities)

@st.cache_data
def get_city_data(city):
    df = pd.read_sql(f"""
        SELECT date as ds, aqi as y
        FROM city_day
        WHERE city = '{city}' AND aqi IS NOT NULL
        ORDER BY date
    """, engine)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

df = get_city_data(selected_city)

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("City", selected_city)
col2.metric("Data Points", f"{len(df):,}")
col3.metric("Date Range", f"{df['ds'].min().date()} to {df['ds'].max().date()}")
col4.metric("Latest AQI", f"{df['y'].iloc[-1]:.0f}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📈 Historical", "🔮 Forecast", "✅ Validation"])

with tab1:
    st.header(f"Historical Data: {selected_city}")
    
    # Calculate yearly stats
    df['year'] = df['ds'].dt.year
    yearly_stats = df.groupby('year')['y'].agg(['mean', 'min', 'max']).round(1)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot daily data
        ax.plot(df['ds'], df['y'], color='steelblue', alpha=0.4, linewidth=0.5, label='Daily AQI')
        
        # Plot yearly averages as scatter points
        yearly_dates = [pd.Timestamp(f"{year}-06-15") for year in yearly_stats.index]
        ax.scatter(yearly_dates, yearly_stats['mean'], color='crimson', s=60, zorder=5, label='Yearly Average')
        ax.plot(yearly_dates, yearly_stats['mean'], color='crimson', linewidth=2, alpha=0.7)
        
        ax.set_title(f'{selected_city} AQI History (2015-2024)')
        ax.set_xlabel('Year')
        ax.set_ylabel('AQI')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Yearly Statistics")
        st.dataframe(yearly_stats)

with tab2:
    st.header(f"Forecast to 2030: {selected_city}")
    
    with st.spinner('Training Prophet model...'):
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, 
                       daily_seasonality=False, changepoint_prior_scale=0.05)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=365*6, freq='D')
        forecast = model.predict(future)
        
        pred_2030 = forecast[forecast['ds'].dt.year == 2030]['yhat'].mean()
        current_avg = df[df['ds'] >= '2024-01-01']['y'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current (2024)", f"{current_avg:.1f}")
        col2.metric("2030 Prediction", f"{pred_2030:.1f}")
        trend = "📉 Improving" if pred_2030 < current_avg else "📈 Worsening"
        col3.metric("Trend", trend)
        
        # Plot forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Historical
        ax.plot(df['ds'], df['y'], color='steelblue', alpha=0.5, linewidth=1, label='Historical')
        
        # Forecast
        future_mask = forecast['ds'] > df['ds'].max()
        ax.plot(forecast['ds'][future_mask], forecast['yhat'][future_mask], 
                color='crimson', linewidth=2, label='Forecast')
        ax.fill_between(forecast['ds'][future_mask], 
                        forecast['yhat_lower'][future_mask], 
                        forecast['yhat_upper'][future_mask],
                        color='crimson', alpha=0.15, label='95% Confidence')
        
        ax.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Moderate (100)')
        ax.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='Poor (200)')
        
        ax.set_title(f'{selected_city} Forecast (2015-2030)')
        ax.set_xlabel('Year')
        ax.set_ylabel('AQI')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Monthly breakdown
        st.subheader("2030 Monthly Forecast")
        pred_2030_df = forecast[forecast['ds'].dt.year == 2030].copy()
        pred_2030_df['month'] = pred_2030_df['ds'].dt.month
        
        monthly_avg = pred_2030_df.groupby('month')['yhat'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        bars = ax2.bar(range(12), [monthly_avg.get(i+1, 0) for i in range(12)], 
                       color='steelblue', alpha=0.7)
        ax2.set_xticks(range(12))
        ax2.set_xticklabels(months)
        ax2.set_title('Average AQI by Month (2030)')
        ax2.set_ylabel('AQI')
        ax2.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig2)

with tab3:
    st.header("Model Validation")
    
    train = df[df['ds'] < '2023-01-01']
    test = df[df['ds'] >= '2023-01-01']
    
    if len(test) > 30:
        with st.spinner('Validating...'):
            model_val = Prophet(yearly_seasonality=True, weekly_seasonality=False, 
                               daily_seasonality=False, changepoint_prior_scale=0.05)
            model_val.fit(train)
            
            future_val = model_val.make_future_dataframe(periods=len(test), freq='D')
            forecast_val = model_val.predict(future_val)
            
            predictions = forecast_val[forecast_val['ds'].isin(test['ds'])][['ds', 'yhat']]
            results = test.merge(predictions, on='ds')
            
            mape = np.mean(np.abs((results['y'] - results['yhat']) / results['y'])) * 100
            rmse = np.sqrt(np.mean((results['y'] - results['yhat'])**2))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MAPE", f"{mape:.1f}%")
            col2.metric("RMSE", f"{rmse:.1f}")
            col3.metric("Test Period", "2023-2024")
            
            quality = 'Reliable ✅' if mape < 20 else 'Moderate ⚠️' if mape < 30 else 'Unreliable ❌'
            st.info(f"**Model Quality: {quality}**\n\nMAPE of {mape:.1f}% means predictions are within ±{mape:.0f}% of actual values")
            
            # Validation plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(results['ds'], results['y'], color='steelblue', linewidth=2, label='Actual')
            ax.plot(results['ds'], results['yhat'], color='crimson', linewidth=2, label='Predicted')
            ax.set_title(f'Validation: Actual vs Predicted (MAPE: {mape:.1f}%)')
            ax.set_xlabel('Date')
            ax.set_ylabel('AQI')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    else:
        st.warning("Insufficient 2023-2024 data for validation")

st.markdown("---")
st.markdown("Built with **Streamlit + Prophet** | Data: India Air Quality (2015-2024)")
