import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler

st.title("Universal Sales Forecasting App")
st.markdown("Upload any sales dataset and the app will automatically clean, group, prepare, train, and forecast.")

# ---------------------- File Upload ----------------------
uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

if uploaded_file is not None:

    # ----------- READ FILE -----------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, header=0)

    st.write("Detected Columns:", df.columns.tolist())
    columns = df.columns.tolist()

    # Select columns
    date_column = st.selectbox("Select Date Column", columns)
    sales_column = st.selectbox("Select Sales Column", columns)

    grouping_columns = st.multiselect(
        "Optional: Select grouping columns (store, category, item...)",
        [c for c in columns if c not in [date_column, sales_column]]
    )

    # ----------- CLEAN DATE -----------
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    df = df.sort_values(date_column)

    # ----------- GROUPING -----------
    if len(grouping_columns) > 0:
        st.info(f"Grouping by Date + {grouping_columns}")
        temp = df.groupby([date_column] + grouping_columns)[sales_column].sum().reset_index()
        temp = temp.groupby(date_column)[sales_column].sum().reset_index()
    else:
        temp = df[[date_column, sales_column]].groupby(date_column).sum().reset_index()

    temp = temp.rename(columns={date_column: "date", sales_column: "sales"})
    temp = temp.drop_duplicates(subset=["date"])
    temp = temp.set_index("date")

    # ----------- RESAMPLE TO DAILY -----------
    if st.checkbox("Resample to Daily (recommended)", value=True):
        temp = temp.resample("D").ffill()

    df = temp.reset_index()

    # ---------------- FEATURE ENGINEERING ----------------
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
    df["ma_9"] = df["sales"].rolling(9).mean()
    df["ma_14"] = df["sales"].rolling(14).mean()
    df["ma_21"] = df["sales"].rolling(21).mean()
    df["ma_25"] = df["sales"].rolling(25).mean()
    df["ma_28"] = df["sales"].rolling(28).mean()
    df["ma_30"] = df["sales"].rolling(30).mean()

    df = df.dropna()

    # ---------------- TRAIN / TEST SPLIT ----------------
    split_idx = int(len(df) * 0.8)

    X = df.drop(["sales", "date"], axis=1)
    y = df["sales"]
    dates_test = df["date"].iloc[split_idx:]

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    # ---------------- SCALING ----------------
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    # ---------------- DATASET ----------------
    class SalesDataset(Dataset):
        def __init__(self, X, y, window=30):
            self.X = X
            self.y = y
            self.window = window

        def __len__(self):
            return len(self.X) - self.window

        def __getitem__(self, idx):
            X_seq = self.X[idx : idx + self.window]
            y_val = self.y[idx + self.window]
            return torch.FloatTensor(X_seq), torch.FloatTensor(y_val)

    window = 30
    train_dataset = SalesDataset(X_train_scaled, y_train_scaled, window)
    test_dataset = SalesDataset(X_test_scaled, y_test_scaled, window)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ---------------- MODEL ----------------
    class LSTM_Model(nn.Module):
        def __init__(self, input_size, hidden=64, layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True)
            self.ReLU=nn.ReLU()
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.ReLU(out)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    model = LSTM_Model(input_size=X_train.shape[1])

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # ---------------- TRAINING ----------------
    num_epochs = 80
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            st.write(f"Epoch {epoch}, Loss = {total_loss:.5f}")

    # ---------------- EVALUATION ----------------
    model.eval()
    preds_scaled = []

    with torch.no_grad():
        for xb, _ in test_loader:
            p = model(xb).numpy()
            preds_scaled.extend(p)

    preds = scaler_y.inverse_transform(np.array(preds_scaled))

    y_test_synced = y_test.values[window:]
    dates_synced = dates_test.values[window:]

    # ---------------- PLOT ----------------
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(dates_synced, y_test_synced, label="Actual", linewidth=2)
    ax.plot(dates_synced, preds, label="Predicted", linestyle="--", linewidth=2)
    ax.legend()
    ax.set_title("LSTM Forecast")
    ax.grid(True)
    st.pyplot(fig)

# ---------------- FORECAST FUTURE ----------------
    st.subheader("Forecast Future Sales")
    n_days = st.number_input("Days to forecast", min_value=1, max_value=60, value=7)

    future_dates = pd.date_range(start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=n_days)

    # آخر 30 يوم (window) لتغذية LSTM
    last_window = X_train_scaled[-window:].copy()
    last_sales = df["sales"].iloc[-window:].tolist()

    future_preds = []

    for future_date in future_dates:

        #  LSTM  Must Be(1, window, features)
        X_input = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_scaled = model(X_input).item()

        pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        future_preds.append(pred)

        # -------- Update the window feature --------
        new_row = {}

        new_row["year"] = future_date.year
        new_row["month"] = future_date.month
        new_row["day"] = future_date.day
        new_row["dayofweek"] = future_date.dayofweek
        new_row["weekofyear"] = future_date.isocalendar().week
        new_row["is_weekend"] = int(future_date.dayofweek in [5, 6])

        # Update The Lags
        all_sales = last_sales + future_preds
        for lag in [1,2,3,4,5,7,14,30]:
            new_row[f"lag_{lag}"] = all_sales[-lag] if len(all_sales) >= lag else all_sales[0]

        # Update The Means
        for ma in [7,9,14,21,25,28,30]:
            new_row[f"ma_{ma}"] = np.mean(all_sales[-ma:]) if len(all_sales) >= ma else np.mean(all_sales)

        new_row_df = pd.DataFrame([new_row])
        new_row_scaled = scaler_X.transform(new_row_df)

        # Update The Window
        last_window = np.vstack([last_window[1:], new_row_scaled])
        last_sales.append(pred)

    st.write("Forecast:", future_preds)
