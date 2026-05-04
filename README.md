# 🛡️ Credit Card Fraud Detection using Machine Learning

## 📌 Overview

This project is a Machine Learning-based system designed to detect fraudulent credit card transactions.
It analyzes transaction data and predicts whether a transaction is **fraudulent or normal** in real-time.

The application is built using
**Python + Streamlit + XGBoost**, providing a simple and interactive user interface.

---

## 🎯 Objectives

* Detect fraudulent transactions accurately
* Handle imbalanced datasets effectively
* Provide real-time prediction through a web interface
* Demonstrate practical use of Machine Learning in finance

---

## 📊 Dataset Information

The dataset contains credit card transaction details, including:

* **Time** – seconds elapsed from first transaction
* **Amount** – transaction amount
* **V1 to V28** – anonymized features
* **Class** – target variable (0 = Normal, 1 = Fraud)

👉 Note:
The features **V1–V28** are generated using **Principal Component Analysis (PCA)** to protect sensitive user data.

---

## ⚙️ Technologies Used

* Python
* Streamlit
* XGBoost
* Pandas
* NumPy
* Scikit-learn

---

## 🧠 Machine Learning Approach

* Data preprocessing and scaling
* Handling imbalanced data using SMOTE
* Model training using XGBoost classifier
* Evaluation using precision, recall, and confusion matrix

---

## 🏗️ Project Structure

```
├── app.py                         # Streamlit frontend
├── credit_card_fraud_pipeline.py # Backend prediction pipeline
├── model.pkl                      # Trained ML model
├── scaler.pkl                     # Data scaler
├── model_columns.pkl              # Feature columns
├── creditcard.csv                 # Dataset
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 🚀 How to Run the Project

1. Clone the repository:

```
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the application:

```
streamlit run app.py
```

---

## 💡 How It Works

1. User enters transaction details
2. Data is preprocessed and scaled
3. Model predicts fraud probability
4. If probability > 0.3 → Fraud
5. Else → Normal transaction

---

## 📈 Output

* Fraud / Normal classification
* Probability score
* Visual alert for suspicious transactions

---

## ⚠️ Limitations

* Uses static dataset
* Not connected to real-time banking systems
* Limited feature inputs in demo

---

## 🔮 Future Improvements

* Integration with real-time transaction systems
* Use of deep learning models
* Improved feature engineering
* Deployment on cloud platforms

---

## 👨‍💻 Author

* Prakrathi K shetty

---

## ⭐ Conclusion

This project demonstrates how Machine Learning can be used to detect fraudulent transactions efficiently.
It provides a foundation for building real-world financial security systems.

---
