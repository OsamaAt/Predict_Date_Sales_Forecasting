import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

st.title("Universal Sales Forecasting App")
st.markdown("Upload any sales dataset and the app will automatically clean, group, prepare, train, and forecast.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

if uploaded_file is not None:

    # ----------- READ FILE -----------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, header=0)

    st.write("Detected Columns:", df.columns.tolist())

    columns = df.columns.tolist()

    # User selects date + sales
    date_column = st.selectbox("Select Date Column", columns)
    sales_column = st.selectbox("Select Sales Column", columns)

    # Optional grouping
    grouping_columns = st.multiselect(
        "Optional: Select grouping columns (e.g., store, category, item)",
        [c for c in columns if c not in [date_column, sales_column]]
    )

    # ----------- CLEAN DATE -----------
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])

    df = df.sort_values(date_column)

    # ----------- GROUPING IF SELECTED -----------
    if len(grouping_columns) > 0:
        st.info(f"Grouping by Date + {grouping_columns}")
        temp = df.groupby([date_column] + grouping_columns)[sales_column].sum().reset_index()
        temp = temp.groupby(date_column)[sales_column].sum().reset_index()
    else:
        temp = df[[date_column, sales_column]].copy()
        temp = temp.groupby(date_column)[sales_column].sum().reset_index()

    temp = temp.drop_duplicates(subset=[date_column])

    # ----------- PREPARE DAILY SERIES -----------
    temp = temp.rename(columns={date_column: "date", sales_column: "sales"})
    temp = temp.set_index("date")

    # Give choice if user wants resample
    do_resample = st.checkbox("Resample to Daily (recommended for forecasting)", value=True)

    if do_resample:
        temp = temp.resample("D").ffill()

    df = temp.reset_index()

    # ----------- FEATURE ENGINEERING -----------
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # LAGS
    for lag in [1,2,3,4,5,7,14,30]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    # MOVING AVERAGES
    df["ma_7"] = df["sales"].rolling(7).mean()
    df["ma_30"] = df["sales"].rolling(30).mean()

    df = df.dropna()

    # ----------- TRAIN TEST SPLIT -----------
    split_idx = int(len(df) * 0.8)
    X_train = df.iloc[:split_idx].drop(["sales", "date"], axis=1)
    y_train = df.iloc[:split_idx]["sales"]
    X_test = df.iloc[split_idx:].drop(["sales", "date"], axis=1)
    y_test = df.iloc[split_idx:]["sales"]
    date_test = df.iloc[split_idx:]["date"]

    # ----------- MODEL SELECTION -----------
    model_choice = st.selectbox("Choose model", ["XGBoost", "Random Forest"])

    if model_choice == "XGBoost":
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            random_state=42
        )

    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    st.metric("R-squared", f"{r2:.4f}")
    st.metric("RMSE", f"{rmse}")
    st.metric("MAE", f"{mae:.2f}")

    # ----------- PLOT -----------
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(date_test, y_test, label="Actual", linewidth=2)
    ax.plot(date_test, y_pred, label="Predicted", linewidth=2, linestyle="--")
    ax.set_title("Actual vs Predicted Sales")
    ax.legend()
    st.pyplot(fig)

    # ----------- FUTURE FORECAST (recursive) -----------
    st.subheader("Forecast Future Sales")
    n_days = st.number_input("Days to forecast", min_value=1, max_value=60, value=7)

    future_dates = pd.date_range(start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=n_days)

    last_row = df.iloc[-1:].copy()
    future_preds = []

    for future_date in future_dates:
        new_row = last_row.copy()

        new_row["year"] = future_date.year
        new_row["month"] = future_date.month
        new_row["day"] = future_date.day
        new_row["dayofweek"] = future_date.weekday()
        new_row["weekofyear"] = future_date.isocalendar()[1]
        new_row["is_weekend"] = int(future_date.weekday() in [5, 6])

        # Update lags
        for lag in [1,2,3,4,5,7,14,30]:
            if len(future_preds) >= lag:
                new_row[f"lag_{lag}"] = future_preds[-lag]
            else:
                new_row[f"lag_{lag}"] = last_row["sales"].iloc[0]

        # Update moving averages
        new_row["ma_7"] = last_row["sales"].iloc[-1]
        new_row["ma_30"] = last_row["sales"].iloc[-1]

        pred = model.predict(new_row.drop(["sales", "date"], axis=1))[0]

        future_preds.append(pred)
        last_row["sales"] = pred  # update for next iteration

    st.write("Forecast:", future_preds)
    
    st.subheader("Residuals Distribution")
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots()
    ax2.hist(residuals, bins=30, color='skyblue', edgecolor='black')
    ax2.set_title("Residuals Histogram")
    ax2.set_xlabel("Error")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)
    
