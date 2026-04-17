# India Air Quality Forecasting & Urban Risk Analysis

## Problem
Urban air quality in India is highly volatile and poorly forecasted at a city level.  
This creates downstream risks for public health, real estate valuation, and policy planning.

## Objective
Build a data-driven pipeline to:
- Analyze historical AQI trends across Indian cities
- Forecast future air quality using time-series modeling
- Surface insights relevant to urban planning and risk assessment

## What This Project Does
This project implements an end-to-end analytics workflow:

1. Data ingestion and preprocessing of AQI datasets  
2. Exploratory analysis of seasonal and city-level trends  
3. Time-series forecasting using Prophet  
4. Visualization of long-term AQI trajectories (up to 2030)  
5. Comparative analysis across cities  

## Key Insights (Sample)
- Clear seasonal AQI spikes across most Tier-1 cities  
- Long-term upward AQI trend in high-growth urban zones  
- Significant variance across cities → localized policy required  

## Tech Stack
- Python (pandas, matplotlib)
- Prophet (time-series forecasting)
- SQL (analytical queries)
- Jupyter Notebooks (exploration)

## Project Structure
-data/ # processed datasets
notebooks/ # analysis and experimentation
scripts/ # production-ready scripts
sql/ # analytical queries


## How to Run
```bash
pip install -r requirements.txt
python scripts/dashboard_fixed.py

Limitations
Historical data limited (majority pre-2020)
Forecast accuracy constrained by lack of external regressors
No real-time data integration
Future Improvements
Integrate live AQI APIs
Add weather + traffic regressors
Build a city-level risk scoring system
Deploy as an API/dashboard
Why This Matters

This project demonstrates how air quality data can be transformed into:

Urban risk indicators
Policy-relevant insights
Decision support tools for real estate and infrastructure planning

- Hyderabad shows a consistent upward AQI trend post-2018, indicating increasing environmental risk in high-growth urban corridors

---

### Step 13 — commit it

```bash id="0n7lwe"
git add README.md
git commit -m "Add consulting-focused project README"