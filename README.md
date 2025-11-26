# 🧠 Universal Sales Forecasting App  
An end-to-end **Sales Forecasting System** built with **PyTorch LSTM**, **Streamlit**, and **Scikit-learn**.  
This app automatically cleans, processes, engineers features, trains a deep learning model, and generates accurate future sales forecasts for any dataset.

---

## 🚀 Features

### ✔ 1. Automatic Data Processing
- Upload CSV or Excel (any structure)
- Automatic date parsing & missing value handling  
- Optional grouping by store / category / item  
- Resampling to daily frequency  

### ✔ 2. Intelligent Feature Engineering
The app generates over **25 time-series features**, including:
- Year, month, day, weekday  
- Week of year  
- Weekend indicator  
- Lag features (1,2,3,4,5,7,14,30 days)  
- Moving averages (7, 9, 14, 21, 25, 28, 30 days)

### ✔ 3. LSTM Deep Learning Model
- Multi-layer **PyTorch LSTM**  
- Window size: 30 days  
- Optimizer: Adam  
- Loss: MSE  
- 80 epochs training  
- Batch processing with DataLoader  

### ✔ 4. Model Evaluation
- Visual comparison between **actual vs predicted** sales  
- Auto-synchronized dates  
- MinMaxScaler for stable learning  

### ✔ 5. Future Forecasting  
Predict future sales for **7–60 days** using recursive multi-step forecasting.  
All future features (lags, MA, date-based features) update dynamically during prediction.

---

## 🖥 Demo (Streamlit)
The app runs fully inside Streamlit with:
- Live training logs  
- Interactive plotting  
- Upload + configure + train + forecast  
- Clean UI for business users  

---


---

## 📸 Model Workflow Diagram

**Dataset → Cleaning → Grouping → Feature Engineering → Scaling → Windowing → LSTM → Forecast**

---

## 🧩 Technologies Used

- **Python**
- **PyTorch**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**

---

## ⚙ How to Run Locally

### 1. Install requirements

pip install -r requirements.txt 

### 2. Run the Streamlit App

streamlit run app.py

### 3. Open in browser

http://localhost:8501

#📊 Example Use Cases

Store sales forecasting

Market demand prediction

Inventory planning

Restaurant daily sales

E-commerce analytics

Multi-store time-series analysis

##🧑‍💻 Author

Osama Al Attar
