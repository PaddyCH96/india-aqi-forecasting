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
    """Get all cities with sufficient data"""
    cities_df = pd.read_sql("""
        SELECT city, COUNT(*) as days, 
               MIN(date) as start_date, 
               MAX(date) as end_date
        FROM city_day 
        WHERE aqi IS NOT NULL
        GROUP BY city
        HAVING COUNT(*) >= 1000
        ORDER BY COUNT(*) DESC
    """, engine)
    return cities_df

# Get cities
cities_df = get_cities()
city_list = cities_df['city'].tolist()

st.sidebar.header("📊 Controls")

# Debug info
st.sidebar.write(f"Found {len(city_list)} cities with data")
st.sidebar.caption("Min 1000 days required")

# City selector
selected_city = st.sidebar.selectbox(
    "🌆 Select City",
    city_list,
    index=0 if 'Hyderabad' not in city_list else city_list.index('Hyderabad')
)

# Show city info
city_info = cities_df[cities_df['city'] == selected_city].iloc[0]
st.sidebar.write(f"**Data available:**")
st.sidebar.write(f"• Days: {city_info['days']:,}")
st.sidebar.write(f"• From: {city_info['start_date']}")
st.sidebar.write(f"• To: {city_info['end_date']}")

st.sidebar.markdown("---")
st.sidebar.markdown("**About:** AQI forecasting using Prophet")

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

# Title for selected city
st.markdown(f"## Currently viewing: **{selected_city}**")

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Date Range", f"{df['ds'].min().date()} to {df['ds'].max().date()}")
col3.metric("Average AQI", f"{df['y'].mean():.0f}")
col4.metric("Latest AQI", f"{df['y'].iloc[-1]:.0f}")

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Historical", "🔮 2030 Forecast", "✅ Validation", "📊 Comparison"])

with tab1:
    st.header(f"Historical AQI Data: {selected_city}")
    
    # Calculate yearly stats
    df['year'] = df['ds'].dt.year
    yearly_stats = df.groupby('year')['y'].agg(['mean', 'min', 'max', 'count']).round(1)
    yearly_stats.columns = ['Mean', 'Min', 'Max', 'Days']
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot daily data
        ax.plot(df['ds'], df['y'], color='steelblue', alpha=0.3, linewidth=0.5, label='Daily AQI')
        
        # Plot yearly averages
        years = yearly_stats.index
        ax.plot([pd.Timestamp(f"{y}-06-15") for y in years], yearly_stats['Mean'], 
                color='crimson', linewidth=3, marker='o', markersize=8, 
                label='Yearly Average', zorder=5)
        
        # Add COVID period shading
        ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-30'), 
                   alpha=0.1, color='gray', label='COVID Period')
        
        ax.set_title(f'{selected_city} AQI History (2015-2024)', fontsize=14)
        ax.set_xlabel('Year')
        ax.set_ylabel('AQI')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add reference lines
        ax.axhline(y=100, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(y=200, color='red', linestyle='--', alpha=0.5)
        
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
        
        # Show trend
        if len(yearly_stats) > 1:
            first_year_avg = yearly_stats.iloc[0]['Mean']
            last_year_avg = yearly_stats.iloc[-1]['Mean']
            change = last_year_avg - first_year_avg
            trend = "improving 📉" if change < 0 else "worsening 📈"
            st.metric("Overall Trend", f"{change:+.1f} points", trend)

with tab2:
    st.header(f"Forecast to 2030: {selected_city}")
    
    with st.spinner(f'Training Prophet model for {selected_city}...'):
        # Prepare data
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        model.fit(df)
        
        # Generate forecast
        future = model.make_future_dataframe(periods=365*6, freq='D')
        forecast = model.predict(future)
        
        # Get 2030 prediction
        pred_2030 = forecast[forecast['ds'].dt.year == 2030]['yhat'].mean()
        pred_2030_lower = forecast[forecast['ds'].dt.year == 2030]['yhat_lower'].mean()
        pred_2030_upper = forecast[forecast['ds'].dt.year == 2030]['yhat_upper'].mean()
        
        # Current average
        current_2024 = df[df['ds'] >= '2024-01-01']['y'].mean() if len(df[df['ds'] >= '2024-01-01']) > 0 else df['y'].tail(365).mean()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current (2024)", f"{current_2024:.1f}")
        col2.metric("2030 Estimate", f"{pred_2030:.1f}")
        col3.metric("Range (95% CI)", f"{pred_2030_lower:.0f} - {pred_2030_upper:.0f}")
        
        change = pred_2030 - current_2024
        trend_icon = "📉 Improving" if change < 0 else "📈 Worsening"
        col4.metric("Trend", trend_icon, f"{abs(change):.1f} points")
        
        # Main forecast plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Historical
        ax.plot(df['ds'], df['y'], color='steelblue', alpha=0.4, linewidth=1, label='Historical (2015-2024)')
        
        # Forecast period
        future_mask = forecast['ds'] > df['ds'].max()
        ax.plot(forecast['ds'][future_mask], forecast['yhat'][future_mask], 
                color='crimson', linewidth=2.5, label='Forecast (2025-2030)')
        
        # Confidence interval
        ax.fill_between(forecast['ds'][future_mask], 
                        forecast['yhat_lower'][future_mask], 
                        forecast['yhat_upper'][future_mask],
                        color='crimson', alpha=0.2, label='95% Confidence Interval')
        
        # Reference lines
        ax.axhline(y=100, color='orange', linestyle='--', alpha=0.7, label='Moderate (100)')
        ax.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='Poor (200)')
        ax.axvline(x=pd.Timestamp('2025-01-01'), color='green', linestyle=':', alpha=0.5, label='Forecast Start')
        
        ax.set_title(f'{selected_city}: AQI Forecast 2025-2030', fontsize=16, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('AQI', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Set x-axis limits to show full forecast
        ax.set_xlim(df['ds'].min(), forecast['ds'].max())
        
        st.pyplot(fig)
        
        # Monthly breakdown for 2030
        st.subheader("2030 Monthly Breakdown")
        pred_2030_monthly = forecast[forecast['ds'].dt.year == 2030].copy()
        pred_2030_monthly['month'] = pred_2030_monthly['ds'].dt.month
        
        monthly_avg = pred_2030_monthly.groupby('month')['yhat'].mean()
        monthly_lower = pred_2030_monthly.groupby('month')['yhat_lower'].mean()
        monthly_upper = pred_2030_monthly.groupby('month')['yhat_upper'].mean()
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        x_pos = range(12)
        
        # Plot bars
        bars = ax2.bar(x_pos, [monthly_avg.get(i+1, 0) for i in range(12)], 
                       color='steelblue', alpha=0.7, edgecolor='black')
        
        # Add error bars for confidence interval
        ax2.errorbar(x_pos, [monthly_avg.get(i+1, 0) for i in range(12)],
                    yerr=[[monthly_avg.get(i+1,0) - monthly_lower.get(i+1,0) for i in range(12)],
                          [monthly_upper.get(i+1,0) - monthly_avg.get(i+1,0) for i in range(12)]],
                    fmt='none', color='black', alpha=0.5)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(months)
        ax2.set_title('Predicted Monthly AQI for 2030', fontsize=14)
        ax2.set_ylabel('AQI')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add reference line
        ax2.axhline(y=100, color='orange', linestyle='--', alpha=0.5)
        
        st.pyplot(fig2)

with tab3:
    st.header(f"Model Validation: {selected_city}")
    
    # Check if we have 2023-2024 data
    train = df[df['ds'] < '2023-01-01']
    test = df[df['ds'] >= '2023-01-01']
    
    if len(test) > 30:
        with st.spinner('Running validation...'):
            # Train on pre-2023
            model_val = Prophet(yearly_seasonality=True, weekly_seasonality=False, 
                               daily_seasonality=False, changepoint_prior_scale=0.05)
            model_val.fit(train)
            
            # Predict 2023-2024
            future_val = model_val.make_future_dataframe(periods=len(test), freq='D')
            forecast_val = model_val.predict(future_val)
            
            predictions = forecast_val[forecast_val['ds'].isin(test['ds'])][['ds', 'yhat']]
            results = test.merge(predictions, on='ds')
            
            # Calculate metrics
            mape = np.mean(np.abs((results['y'] - results['yhat']) / results['y'])) * 100
            rmse = np.sqrt(np.mean((results['y'] - results['yhat'])**2))
            mae = np.mean(np.abs(results['y'] - results['yhat']))
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAPE", f"{mape:.1f}%")
            col2.metric("RMSE", f"{rmse:.1f}")
            col3.metric("MAE", f"{mae:.1f}")
            col4.metric("Test Days", f"{len(test)}")
            
            # Quality assessment
            if mape < 15:
                quality = "Excellent ✅"
                color = "green"
            elif mape < 20:
                quality = "Good ✅"
                color = "blue"
            elif mape < 30:
                quality = "Moderate ⚠️"
                color = "orange"
            else:
                quality = "Poor ❌"
                color = "red"
            
            st.markdown(f"<h3 style='color: {color}'>Model Quality: {quality}</h3>", unsafe_allow_html=True)
            
            st.info(f"""
            **Interpretation:**
            - **MAPE of {mape:.1f}%** means predictions are typically within ±{mape:.0f}% of actual values
            - For a prediction of 100 AQI, expect actual value between **{100*(1-mape/100):.0f} and {100*(1+mape/100):.0f}**
            - Model trained on {len(train)} days, tested on {len(test)} days (2023-2024)
            """)
            
            # Validation plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results['ds'], results['y'], color='steelblue', linewidth=2, 
                   label='Actual AQI (2023-2024)', marker='o', markersize=3)
            ax.plot(results['ds'], results['yhat'], color='crimson', linewidth=2, 
                   label=f'Predicted (MAPE: {mape:.1f}%)', alpha=0.8)
            
            # Shade prediction error
            ax.fill_between(results['ds'], results['yhat']*0.8, results['yhat']*1.2, 
                           alpha=0.1, color='red', label='±20% Error Band')
            
            ax.set_title(f'Model Validation: Actual vs Predicted for {selected_city}', fontsize=14)
            ax.set_xlabel('Date')
            ax.set_ylabel('AQI')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Scatter plot
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.scatter(results['y'], results['yhat'], alpha=0.5, edgecolors='black', linewidth=0.5)
            
            # Perfect prediction line
            min_val = min(results['y'].min(), results['yhat'].min())
            max_val = max(results['y'].max(), results['yhat'].max())
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            ax2.set_xlabel('Actual AQI')
            ax2.set_ylabel('Predicted AQI')
            ax2.set_title(f'Prediction Accuracy (MAPE: {mape:.1f}%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            
    else:
        st.warning(f"⚠️ Insufficient 2023-2024 data for {selected_city} validation")
        st.info("Need at least 30 days of 2023-2024 data to validate model performance.")

with tab4:
    st.header("Multi-City Comparison")
    
    st.markdown("Compare forecast accuracy across all cities:")
    
    # Show summary table
    comparison_data = []
    for city in city_list[:10]:  # Top 10 by data volume
        city_df = get_city_data(city)
        train_c = city_df[city_df['ds'] < '2023-01-01']
        test_c = city_df[city_df['ds'] >= '2023-01-01']
        
        if len(test_c) > 30:
            # Quick validation
            try:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, 
                           daily_seasonality=False)
                m.fit(train_c)
                future_c = m.make_future_dataframe(periods=len(test_c), freq='D')
                forecast_c = m.predict(future_c)
                pred_c = forecast_c[forecast_c['ds'].isin(test_c['ds'])][['ds', 'yhat']]
                res_c = test_c.merge(pred_c, on='ds')
                mape_c = np.mean(np.abs((res_c['y'] - res_c['yhat']) / res_c['y'])) * 100
                
                comparison_data.append({
                    'City': city,
                    'MAPE (%)': round(mape_c, 1),
                    '2024 Avg': round(city_df[city_df['ds'] >= '2024-01-01']['y'].mean(), 1),
                    'Data Quality': 'High' if mape_c < 20 else 'Medium' if mape_c < 30 else 'Low'
                })
            except:
                pass
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data).sort_values('MAPE (%)')
        st.dataframe(comp_df, use_container_width=True)
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(comp_df['City'], comp_df['MAPE (%)'], 
                       color=['green' if x < 20 else 'orange' if x < 30 else 'red' for x in comp_df['MAPE (%)']])
        ax.set_xlabel('MAPE (%) - Lower is Better')
        ax.set_title('Forecast Accuracy Comparison Across Cities')
        ax.axvline(x=20, color='orange', linestyle='--', alpha=0.5, label='Good Threshold')
        ax.legend()
        st.pyplot(fig)
        
        st.info(f"**{selected_city} ranking:** Currently viewing one of {len(comp_df)} cities with validation data")
    else:
        st.info("Run validation on individual cities to see comparison")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Built with Streamlit + Prophet | India AQI Forecasting Project | 2015-2030</small>
</div>
""", unsafe_allow_html=True)
