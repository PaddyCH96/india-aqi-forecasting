import pandas as pd
import numpy as np
from prophet import Prophet
from sqlalchemy import create_engine
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Connect to PostgreSQL
print("Connecting to database...")
engine = create_engine('postgresql://postgres@localhost:5432/india_air_quality')

# Get eligible cities (>=1000 days, data through 2024)
print("Fetching city list...")
cities_df = pd.read_sql("""
    SELECT city, COUNT(*) as total_days, MAX(date) as end_date
    FROM city_day 
    WHERE aqi IS NOT NULL
    GROUP BY city 
    HAVING COUNT(*) >= 1000 AND MAX(date) >= '2024-01-01'
    ORDER BY total_days DESC
""", engine)

eligible = cities_df['city'].tolist()
print(f"\nFound {len(eligible)} eligible cities: {eligible}\n")

def forecast_city(city_name):
    """Train model and return metrics for a city"""
    try:
        # Load data
        df = pd.read_sql(f"""
            SELECT date as ds, aqi as y
            FROM city_day
            WHERE city = '{city_name}' AND aqi IS NOT NULL
            ORDER BY date
        """, engine)
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Split: Train 2015-2022, Test 2023-2024
        train = df[df['ds'] < '2023-01-01']
        test = df[df['ds'] >= '2023-01-01']
        
        if len(test) < 30:
            return None
            
        # Train Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(train)
        
        # Validate
        future = model.make_future_dataframe(periods=len(test), freq='D')
        forecast = model.predict(future)
        predictions = forecast[forecast['ds'].isin(test['ds'])][['ds', 'yhat']]
        results = test.merge(predictions, on='ds')
        
        # Calculate MAPE
        mape = np.mean(np.abs((results['y'] - results['yhat']) / results['y'])) * 100
        
        # 2030 forecast
        future_2030 = model.make_future_dataframe(periods=365*6, freq='D')
        forecast_2030 = model.predict(future_2030)
        pred_2030 = forecast_2030[forecast_2030['ds'].dt.year == 2030]['yhat'].mean()
        
        # Recent average
        recent = df[df['ds'] >= '2024-01-01']['y'].mean()
        
        return {
            'city': city_name,
            'mape': round(mape, 2),
            'avg_aqi_2024': round(recent, 1),
            'pred_aqi_2030': round(pred_2030, 1),
            'trend': 'improving' if pred_2030 < recent else 'worsening'
        }
        
    except Exception as e:
        print(f"\nError on {city_name}: {e}")
        return None

# Process all cities
print("Training models for all cities (this takes 2-3 minutes)...\n")
results = []

for city in tqdm(eligible, desc="Processing"):
    result = forecast_city(city)
    if result:
        results.append(result)

# Create results DataFrame
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('mape')

# Display results
print("\n" + "="*70)
print("MULTI-CITY FORECAST RESULTS")
print("="*70)
print(f"\nTotal cities processed: {len(df_results)}")
print(f"Average MAPE: {df_results['mape'].mean():.1f}%")
print(f"Best: {df_results['mape'].min():.1f}% ({df_results.loc[df_results['mape'].idxmin(), 'city']})")
print(f"Worst: {df_results['mape'].max():.1f}% ({df_results.loc[df_results['mape'].idxmax(), 'city']})")

print("\n" + "-"*70)
print("CITY RANKINGS (by forecast accuracy):")
print("-"*70)
print(df_results[['city', 'mape', 'avg_aqi_2024', 'pred_aqi_2030', 'trend']].to_string(index=False))

# Best and worst
print("\n" + "="*70)
print("🏆 MOST FORECASTABLE (MAPE < 15%):")
best = df_results[df_results['mape'] < 15]
if len(best) > 0:
    for _, row in best.iterrows():
        print(f"  • {row['city']}: {row['mape']:.1f}% MAPE")
else:
    print("  No cities with MAPE < 15%")

print("\n⚠️  LEAST FORECASTABLE (MAPE > 25%):")
worst = df_results[df_results['mape'] > 25].sort_values('mape', ascending=False)
if len(worst) > 0:
    for _, row in worst.head(3).iterrows():
        print(f"  • {row['city']}: {row['mape']:.1f}% MAPE")
else:
    print("  No cities with MAPE > 25%")

# Trend summary
improving = len(df_results[df_results['trend'] == 'improving'])
worsening = len(df_results[df_results['trend'] == 'worsening'])
print(f"\n📈 TREND SUMMARY:")
print(f"  Improving by 2030: {improving} cities ({improving/len(df_results)*100:.0f}%)")
print(f"  Worsening by 2030: {worsening} cities ({worsening/len(df_results)*100:.0f}%)")

# Save results
df_results.to_csv('outputs/multi_city_results.csv', index=False)
print("\n" + "="*70)
print("✅ Saved: outputs/multi_city_results.csv")
print("="*70)
