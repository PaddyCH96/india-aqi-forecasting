#!/usr/bin/env python3
import pandas as pd
import requests
import time
from sqlalchemy import create_engine

CITIES = {
    'Hyderabad': {'lat': 17.385, 'lon': 78.4867},
    'Delhi': {'lat': 28.6139, 'lon': 77.209},
    'Mumbai': {'lat': 19.076, 'lon': 72.8777},
    'Bengaluru': {'lat': 12.9716, 'lon': 77.5946},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707},
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639}
}

def main():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    all_data = []
    
    print("=== Fetching AQI 2020-2024 ===\n")
    
    for city, coords in CITIES.items():
        params = {
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'start_date': '2020-07-01',
            'end_date': '2024-12-31',
            'daily': ['pm10', 'pm2_5'],
            'timezone': 'Asia/Kolkata'
        }
        
        print(f"Fetching {city}...")
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'daily' in data and data['daily']:
                df = pd.DataFrame({
                    'date': pd.to_datetime(data['daily']['time']),
                    'pm25': data['daily']['pm2_5'],
                    'pm10': data['daily']['pm10'],
                    'city': city
                })
                
                # Calculate AQI
                def calc_aqi(pm25):
                    if pd.isna(pm25): return None
                    if pm25 <= 30: return pm25 * 1.67
                    elif pm25 <= 60: return 50 + (pm25-30) * 1.67
                    elif pm25 <= 90: return 100 + (pm25-60) * 3.33
                    elif pm25 <= 120: return 200 + (pm25-90) * 3.33
                    elif pm25 <= 250: return 300 + (pm25-120) * 0.77
                    else: return min(400 + (pm25-250) * 0.4, 500)
                
                df['aqi'] = df['pm25'].apply(calc_aqi).round(0)
                all_data.append(df)
                print(f"  ✓ {len(df)} days ({df['date'].min().date()} to {df['date'].max().date()})")
            else:
                print(f"  ⚠️  No data returned")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
        
        time.sleep(1.5)
    
    if not all_data:
        print("\n✗ No data fetched")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n=== Summary ===")
    print(f"Total: {len(combined)} rows")
    print(f"Range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"Cities: {combined['city'].nunique()}")
    
    # Save
    output = 'data/processed/aqi_2020_2024_fetched.csv'
    combined.to_csv(output, index=False)
    print(f"\n✓ Saved: {output}")
    
    # Database
    print("\n=== Database ===")
    try:
        engine = create_engine('postgresql://postgres@localhost:5432/india_air_quality')
        with engine.connect() as conn:
            existing = conn.execute("SELECT COUNT(*) FROM city_day WHERE date >= '2020-07-01'").scalar()
            print(f"Existing rows: {existing}")
            
            if existing == 0:
                combined.rename(columns={'pm25': 'pm2_5'}).to_sql('city_day', engine, if_exists='append', index=False)
                print(f"✓ Inserted {len(combined)} rows")
            else:
                print("⚠️  Data exists, skipping")
    except Exception as e:
        print(f"✗ DB error: {e}")

if __name__ == '__main__':
    main()
