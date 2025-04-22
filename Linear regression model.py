import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ----------------- CONFIGURATION -----------------
N_WEEKS_FORECAST = 4
MIN_HISTORY = 4
FILE_DEMAND = "demand_data.xlsx"
FILE_SUPPLY = "supply_data.xlsx"
FILE_COVARIATES = "covariate_data.xlsx"
# -------------------------------------------------

def load_and_preprocess():
    demand = pd.read_excel(FILE_DEMAND)
    supply = pd.read_excel(FILE_SUPPLY)
    covariates = pd.read_excel(FILE_COVARIATES)

    demand['date'] = pd.to_datetime(demand['date'])
    supply['date'] = pd.to_datetime(supply['date'])

    demand['week'] = demand['date'].dt.to_period('W').apply(lambda r: r.start_time)
    supply['week'] = supply['date'].dt.to_period('W').apply(lambda r: r.start_time)

    weekly_demand = demand.groupby(['week', 'location', 'food_type'])['quantity'].sum().reset_index()
    weekly_supply = supply.groupby(['week', 'location', 'food_type'])['quantity'].sum().reset_index()

    return weekly_demand, weekly_supply, covariates

def create_time_series_matrix(df, label):
    matrix = df.pivot_table(index='week', columns=['location', 'food_type'], values='quantity', aggfunc='sum').fillna(0)
    matrix.columns = pd.MultiIndex.from_tuples(matrix.columns)
    return matrix


def forecast_demand_from_model(demand_matrix, supply_matrix):
    forecast_output = []

    for col in demand_matrix.columns:
        demand_series = demand_matrix[col]

        # Use supply from the same column (if available)
        supply_series = supply_matrix[col] if col in supply_matrix.columns else pd.Series(0, index=demand_series.index)

        valid_demand = demand_series[demand_series > 0]
        if len(valid_demand) < MIN_HISTORY:
            continue

        # Train only on demand
        X = np.arange(len(valid_demand)).reshape(-1, 1)
        y = valid_demand.values
        model = LinearRegression()
        model.fit(X, y)

        last_week = valid_demand.index[-1]

        for i in range(1, N_WEEKS_FORECAST + 1):
            future_index = len(valid_demand) + i - 1
            forecast_val = model.predict([[future_index]])[0]
            forecast_val = max(forecast_val, 0)

            future_week = last_week + timedelta(weeks=i)
            week_str = future_week.strftime('%Y-%W')

            # ⚠️ Fix: use actual supply from that forecasted week
            supply_val = 0
            if future_week in supply_series.index:
                supply_val = supply_series.loc[future_week]
            elif supply_series.index.max() < future_week:
                supply_val = supply_series.loc[supply_series.index.max()]
            else:
                supply_val = 0

            status = "Demand" if forecast_val > supply_val else "Surplus"

            forecast_output.append({
                "week": week_str,
                "food_type": col[1],
                "location": col[0],
                "status": status,
                "forecast_quantity": round(forecast_val, 2),
                "supply_checked": round(supply_val, 2) 
            })

    return forecast_output

def save_trained_models(demand_matrix):
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)

    for col in demand_matrix.columns:
        demand_series = demand_matrix[col]
        valid_demand = demand_series[demand_series > 0]

        if len(valid_demand) < MIN_HISTORY:
            continue

        X = np.arange(len(valid_demand)).reshape(-1, 1)
        y = valid_demand.values

        model = LinearRegression()
        model.fit(X, y)

        loc, food = col
        filename = f"{model_dir}/{loc}_{food}_demand_model.pkl"
        joblib.dump(model, filename)
        print(f"Saved model for {food} at {loc} -> {filename}")

def main():
    weekly_demand, weekly_supply, covariates = load_and_preprocess()

    demand_ts = create_time_series_matrix(weekly_demand, 'Demand')
    supply_ts = create_time_series_matrix(weekly_supply, 'Supply')

    forecast = forecast_demand_from_model(demand_ts, supply_ts)

    print("\n======= DEMAND FORECAST OUTPUT =======\n")
    for f in forecast:
        print(f"WEEK: {f['week']} | ITEM: {f['food_type']} | LOCATION: {f['location']} | STATUS: {f['status']} | FORECAST: {f['forecast_quantity']}")

if __name__ == "__main__":
    main()
