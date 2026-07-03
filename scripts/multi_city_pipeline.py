import pandas as pd
import numpy as np
from tqdm import tqdm
from lib.logging import setup_logger
from lib.db import get_engine, get_eligible_cities, load_city_data
from lib.models import train_and_forecast, train_and_validate
from lib.metrics import evaluate_forecast

logger = setup_logger("multi-city-pipeline")

engine = get_engine()

logger.info("Fetching eligible cities...")
cities_df = get_eligible_cities(engine)
eligible = cities_df['city'].tolist()
logger.info(f"Found {len(eligible)} eligible cities: {eligible}")


def forecast_city(city_name: str):
    try:
        df = load_city_data(engine, city_name)

        train = df[df['ds'] < '2023-01-01']
        test = df[df['ds'] >= '2023-01-01']

        if len(test) < 30:
            return None

        _, _, results = train_and_validate(train, test)
        metrics = evaluate_forecast(results)
        mape = metrics['mape']

        _, forecast_2030 = train_and_forecast(df)
        pred_2030 = forecast_2030[forecast_2030['ds'].dt.year == 2030]['yhat'].mean()
        recent = df[df['ds'] >= '2024-01-01']['y'].mean()

        return {
            'city': city_name,
            'mape': round(mape, 2),
            'avg_aqi_2024': round(recent, 1),
            'pred_aqi_2030': round(pred_2030, 1),
            'trend': 'improving' if pred_2030 < recent else 'worsening'
        }

    except Exception as e:
        logger.error(f"Error on {city_name}: {e}")
        return None


logger.info("Training models for all cities (this takes 2-3 minutes)...\n")
results = []

for city in tqdm(eligible, desc="Processing"):
    result = forecast_city(city)
    if result:
        results.append(result)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('mape')

print("\n" + "=" * 70)
print("MULTI-CITY FORECAST RESULTS")
print("=" * 70)
print(f"\nTotal cities processed: {len(df_results)}")
print(f"Average MAPE: {df_results['mape'].mean():.1f}%")
print(f"Best: {df_results['mape'].min():.1f}% ({df_results.loc[df_results['mape'].idxmin(), 'city']})")
print(f"Worst: {df_results['mape'].max():.1f}% ({df_results.loc[df_results['mape'].idxmax(), 'city']})")

print("\n" + "-" * 70)
print("CITY RANKINGS (by forecast accuracy):")
print("-" * 70)
print(df_results[['city', 'mape', 'avg_aqi_2024', 'pred_aqi_2030', 'trend']].to_string(index=False))

print("\n" + "=" * 70)
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

improving = len(df_results[df_results['trend'] == 'improving'])
worsening = len(df_results[df_results['trend'] == 'worsening'])
print(f"\n📈 TREND SUMMARY:")
total = len(df_results)
print(f"  Improving by 2030: {improving} cities ({improving / total * 100:.0f}%)" if total > 0 else "  No data")
print(f"  Worsening by 2030: {worsening} cities ({worsening / total * 100:.0f}%)" if total > 0 else "  No data")

df_results.to_csv('outputs/multi_city_results.csv', index=False)
print("\n" + "=" * 70)
print("✅ Saved: outputs/multi_city_results.csv")
print("=" * 70)
