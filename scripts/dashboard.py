import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sqlalchemy import create_engine
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="India AQI Forecast Dashboard",
    page_icon="🏭",
    layout="wide"
)

# Title
st.title("🏭 India AQI Forecasting Dashboard")
st.markdown("**Interactive forecasts for Indian cities (2015-2030)**")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    engine = create_engine('postgresql://postgres@localhost:5432/india_air_quality')
    
    # Get city list
    cities = pd.read_sql("""
        SELECT DISTINCT city 
        FROM city_day 
        WHERE aqi IS NOT NULL AND date >= '2024-01-01'
        ORDER BY city
    """, engine)['city'].tolist()
    
    return cities, engine

cities, engine = load_data()

# Sidebar
st.sidebar.header("📊 Controls")
selected_city = st.sidebar.selectbox("Select City", cities)

# Load city data
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

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Historical Data", "🔮 Forecast to 2030", "📊 Model Validation"])

with tab1:
    st.header(f"Historical AQI Data: {selected_city}")
    
    # Yearly averages
    df['year'] = df['ds'].dt.year
    yearly_avg = df.groupby('year')['y'].mean().reset_index()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['ds'], y=df['y'],
            mode='lines',
            name='Daily AQI',
            line=dict(color='steelblue', width=1),
            opacity=0.6
        ))
        fig.add_trace(go.Scatter(
            x=yearly_avg['year'].astype(str) + '-06-15',
            y=yearly_avg['y'],
            mode='lines+markers',
            name='Yearly Average',
            line=dict(color='crimson', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title=f'{selected_city} AQI History (2015-2024)',
            xaxis_title='Year',
            yaxis_title='AQI',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Yearly Statistics")
        stats = df.groupby('year')['y'].agg(['mean', 'min', 'max', 'std']).round(1)
        stats.columns = ['Mean', 'Min', 'Max', 'Std']
        st.dataframe(stats, use_container_width=True)

with tab2:
    st.header(f"Forecast to 2030: {selected_city}")
    
    with st.spinner(f'Training Prophet model for {selected_city}...'):
        # Train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        model.fit(df)
        
        # Forecast
        future = model.make_future_dataframe(periods=365*6, freq='D')
        forecast = model.predict(future)
        
        # Extract 2030 prediction
        pred_2030 = forecast[forecast['ds'].dt.year == 2030]['yhat'].mean()
        current_avg = df[df['ds'] >= '2024-01-01']['y'].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current AQI (2024 avg)", f"{current_avg:.1f}")
        col2.metric("2030 Prediction", f"{pred_2030:.1f}")
        col3.metric("Trend", "Improving 📉" if pred_2030 < current_avg else "Worsening 📈")
        
        # Plot
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=df['ds'], y=df['y'],
            mode='lines',
            name='Historical',
            line=dict(color='steelblue', width=1),
            opacity=0.6
        ))
        
        # Forecast
        future_mask = forecast['ds'] > df['ds'].max()
        fig.add_trace(go.Scatter(
            x=forecast['ds'][future_mask],
            y=forecast['yhat'][future_mask],
            mode='lines',
            name='Forecast',
            line=dict(color='crimson', width=2)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'][future_mask].tolist() + forecast['ds'][future_mask].tolist()[::-1],
            y=forecast['yhat_upper'][future_mask].tolist() + forecast['yhat_lower'][future_mask].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence'
        ))
        
        # Reference lines
        fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Moderate")
        fig.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Poor")
        fig.add_vline(x=pd.Timestamp('2025-01-01'), line_dash="dot", line_color="green", annotation_text="Forecast Start")
        
        fig.update_layout(
            title=f'{selected_city} AQI Forecast (2025-2030)',
            xaxis_title='Year',
            yaxis_title='AQI',
            height=600,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly 2030 breakdown
        st.subheader("2030 Monthly Forecast")
        pred_2030_monthly = forecast[forecast['ds'].dt.year == 2030]
        pred_2030_monthly['month'] = pred_2030_monthly['ds'].dt.strftime('%B')
        monthly_avg = pred_2030_monthly.groupby('month')['yhat'].mean().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=monthly_avg.index,
            y=monthly_avg.values,
            marker_color='steelblue'
        ))
        fig2.update_layout(
            title='Average AQI by Month (2030)',
            xaxis_title='Month',
            yaxis_title='AQI',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("Model Validation")
    
    # Calculate MAPE for this city
    train = df[df['ds'] < '2023-01-01']
    test = df[df['ds'] >= '2023-01-01']
    
    if len(test) > 30:
        with st.spinner('Validating model...'):
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
            
            # Validation plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['ds'], y=results['y'],
                mode='lines',
                name='Actual',
                line=dict(color='steelblue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=results['ds'], y=results['yhat'],
                mode='lines',
                name='Predicted',
                line=dict(color='crimson', width=2)
            ))
            fig.update_layout(
                title=f'Model Validation: Actual vs Predicted (MAPE: {mape:.1f}%)',
                xaxis_title='Date',
                yaxis_title='AQI',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"""
            **Interpretation:**
            - MAPE of {mape:.1f}% means predictions are typically within ±{mape:.0f}% of actual values
            - For a prediction of 100 AQI, expect actual value between {100*(1-mape/100):.0f} and {100*(1+mape/100):.0f}
            - {'Reliable' if mape < 20 else 'Moderate' if mape < 30 else 'Unreliable'} forecast quality
            """)
    else:
        st.warning("Insufficient 2023-2024 data for validation")

# Footer
st.markdown("---")
st.markdown("""
**About this dashboard:**
- Built with Streamlit + Plotly + Prophet
- Data: India Air Quality (2015-2024)
- Model: Facebook Prophet with yearly seasonality
- Validation: Train 2015-2022, Test 2023-2024
""")

st.sidebar.markdown("---")
st.sidebar.info("""
**How to use:**
1. Select a city from the dropdown
2. View historical trends
3. See 2030 forecast with confidence intervals
4. Check model validation metrics
""")
