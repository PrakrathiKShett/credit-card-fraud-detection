import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import traceback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_mock_data(filename="creditcard.csv", num_samples=20000):
    """
    Generates a mock dataset resembling the Kaggle Credit Card Fraud dataset.
    """
    print(f"[INFO] '{filename}' not found. Generating mock dataset with {num_samples} samples...")
    time_col = np.random.uniform(0, 172800, num_samples)
    amount_col = np.random.exponential(scale=50, size=num_samples)
    v_cols = {f'V{i}': np.random.randn(num_samples) for i in range(1, 29)}
    class_col = np.random.choice([0, 1], size=num_samples, p=[0.985, 0.015])
    
    fraud_indices = np.where(class_col == 1)[0]
    v_cols['V1'][fraud_indices] -= 5
    v_cols['V2'][fraud_indices] += 4
    v_cols['V3'][fraud_indices] -= 5
    v_cols['V14'][fraud_indices] -= 6

    data = {'Time': time_col}
    data.update(v_cols)
    data['Amount'] = amount_col
    data['Class'] = class_col
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"[INFO] Mock dataset saved as '{filename}'.\n")
    return df

def feature_engineering(df, is_train=True, scaler=None):
    """
    Applies feature engineering. Extracts time features and scales amount.
    """
    try:
        df = df.copy()
        
        # Explain V1-V28 (Only on train/EDA to avoid spamming logs on inference)
        if is_train:
            print("[INFO] Note on Dataset: V1-V28 are principal components obtained with PCA.")
            print("       They are already scaled and anonymized to protect user identities and sensitive features.")
            
        # 1. Feature: transaction_hour
        if 'Time' in df.columns:
            df['transaction_hour'] = (df['Time'] // 3600) % 24
        elif 'transaction_hour' not in df.columns:
             df['transaction_hour'] = 12 # Default fallback
            
        # 2. Feature: is_night_transaction (Assuming 0 to 6 is night)
        df['is_night_transaction'] = df['transaction_hour'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
        
        # 3. Feature: amount_zscore 
        # Z-score helps find amount anomalies globally
        if is_train:
            amount_mean = df['Amount'].mean()
            amount_std = df['Amount'].std()
            df['amount_zscore'] = (df['Amount'] - amount_mean) / (amount_std + 1e-6)
            # Store mean and std in scaler object dynamically later, here we just do simple standard scaling
        else:
            # During inference, we rely on the StandardScaler for 'Amount' overall.
            # Z-score can just be approximated or we can just use StandardScaler. Let's just create the column for continuity.
            pass
            
        # 4. Optional: Simulate User_ID metrics
        if 'User_ID' in df.columns:
            if is_train:
                user_avg = df.groupby('User_ID')['Amount'].mean().to_dict()
                df['user_avg_amount'] = df['User_ID'].map(user_avg).fillna(df['Amount'].mean())
            else:
                # If User_ID is supplied in prediction, assume average is the transaction if brand new
                df['user_avg_amount'] = df['Amount'] 
            df['amount_to_avg_ratio'] = df['Amount'] / (df['user_avg_amount'] + 1e-6)

        return df
    except Exception as e:
        print(f"[ERROR] feature_engineering failed: {e}")
        return None

def train_and_evaluate():
    print("="*50)
    print(" END-TO-END CREDIT CARD FRAUD PIPELINE ")
    print("="*50)
    
    # 1. Load Data
    dataset_file = 'creditcard.csv'
    if not os.path.exists(dataset_file):
        df = generate_mock_data(dataset_file)
    else:
        df = pd.read_csv(dataset_file)
        
    print(f"\n[EDA] Dataset Shape: {df.shape}")
    print(f"[EDA] Class Distribution:\n{df['Class'].value_counts(normalize=True)*100}\n")
    
    # Optional Simulation of User ID
    df['User_ID'] = np.random.randint(1, max(2, len(df)//10), df.shape[0])
    
    # 2. Feature Engineering
    df = feature_engineering(df, is_train=True)
    
    # For z-score we use StandardScaler officially
    df.drop(columns=['amount_zscore'], inplace=True, errors='ignore') 
    
    X = df.drop(columns=['Class', 'Time', 'User_ID'])
    y = df['Class']
    
    # 3. Train-Test Split & Scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    scaler = StandardScaler()
    cols_to_scale = ['Amount', 'user_avg_amount', 'amount_to_avg_ratio', 'transaction_hour']
    # Ensure they exist (if User_ID wasn't used)
    cols_to_scale = [c for c in cols_to_scale if c in X_train.columns]
    
    X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # 4. Handle Imbalance
    print("\n[INFO] Handling Imbalance using SMOTE...")
    print("       Without SMOTE, models might overfit to the majority class (Normal) getting 99% accuracy but 0% Recall.")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE: Normal: {sum(y_train_smote==0)}, Fraud: {sum(y_train_smote==1)}\n")
    
    # 5. Models (Comparing with class_weight='balanced')
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=10, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': XGBClassifier(scale_pos_weight=(sum(y_train==0)/sum(y_train==1)), random_state=RANDOM_STATE, n_estimators=50, eval_metric='logloss')
    }
    
    results = {}
    best_f1 = -1
    best_model_name = ""
    best_model = None
    
    print("="*20 + " MODEL EVALUATION " + "="*20)
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_smote, y_train_smote)
        
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
        
        # Default Threshold 0.5
        y_pred = (y_prob >= 0.5).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_test, y_prob)
        
        print(f"[{name}] Threshold 0.5 -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc_score:.4f}")
        
        results[name] = {'Precision': precision, 'Recall': recall, 'F1': f1, 'AUC': auc_score}
        
        # Select best model based on combination of Recall and F1 (Harmonic preference to strong recall)
        score = recall * 0.6 + f1 * 0.4
        if score > best_f1:
            best_f1 = score
            best_model_name = name
            best_model = model

    # 6. Threshold Tuning (IMPORTANT) on the BEST model
    print(f"\n[INFO] Threshold Tuning on Best Model ({best_model_name})...")
    y_prob_best = best_model.predict_proba(X_test)[:, 1]
    
    thresholds_to_test = [0.2, 0.3, 0.5, 0.7]
    for thresh in thresholds_to_test:
        y_pred_t = (y_prob_best >= thresh).astype(int)
        r_t = recall_score(y_test, y_pred_t, zero_division=0)
        p_t = precision_score(y_test, y_pred_t, zero_division=0)
        print(f"       Threshold {thresh:.2f} -> Recall: {r_t:.4f} | Precision: {p_t:.4f}")

    df_results = pd.DataFrame(results).T
    print("\n=== Model Comparison Table ===")
    print(df_results)
    
    # 7. Visualizations
    
    # Precision-Recall Curve (Crucial for Imbalanced Data)
    pt, rt, _ = precision_recall_curve(y_test, y_prob_best)
    plt.figure(figsize=(8,6))
    plt.plot(rt, pt, label=f"{best_model_name} PR Curve (AUC={auc(rt, pt):.3f})", color='purple')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Trade-off')
    plt.legend()
    plt.savefig('precision_recall_curve.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={results[best_model_name]['AUC']:.3f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('roc_curve.png')
    plt.close()
    
    # Confusion Matrix (At 0.3 threshold for higher recall)
    tuned_thresh = 0.3
    y_pred_tuned = (y_prob_best >= tuned_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_tuned)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title(f'Confusion Matrix (Threshold {tuned_thresh})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        plt.figure(figsize=(10, 8))
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        plt.barh(range(len(indices)), importances[indices][::-1], align='center')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices][::-1])
        plt.xlabel('Relative Importance')
        plt.title('Top 15 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("\n[INFO] Feature Importance Note: Higher values mean the feature more heavily splits fraud vs normal.")

    # 8. Save Pipeline
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    # Save column structure for prediction mapping
    joblib.dump(list(X.columns), 'model_columns.pkl')
    print(f"\n[INFO] best_model.pkl, scaler.pkl, and model_columns.pkl saved successfully.")
    
    print_viva_explanations()

def predict_transaction(input_dict):
    """
    Real-Time Prediction Function
    Handles graceful error catching and returns prediction probability.
    """
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        columns = joblib.load('model_columns.pkl')
        
        # Convert to DataFrame
        df_in = pd.DataFrame([input_dict])
        
        # Missing values handling
        df_in = df_in.fillna(0)
        
        # Apply Feature Engineering
        df_in = feature_engineering(df_in, is_train=False)
        
        # Ensure all required columns exist
        for col in columns:
            if col not in df_in.columns:
                df_in[col] = 0.0 # Graceful fallback
                
        # Reorder to match model
        X_infer = df_in[columns]

        # 🔥 ADD THIS LINE (IMPORTANT FIX)
        X_infer = X_infer.astype(float)
        
        # Scale
        cols_to_scale = ['Amount', 'user_avg_amount', 'amount_to_avg_ratio', 'transaction_hour']
        cols_to_scale = [c for c in cols_to_scale if c in X_infer.columns]
        
        X_infer.loc[:, cols_to_scale] = scaler.transform(X_infer[cols_to_scale])
        
        probability = model.predict_proba(X_infer)[0, 1]
        
        # Using tuned threshold of 0.3 for maximum recall safety
        label = "FRAUD" if probability >= 0.3 else "NORMAL"
        
        return {"probability": float(probability), "label": label, "status": "success"}
        
    except Exception as e:
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}

def print_viva_explanations():
    """
    Prints the required theoretical explanations for Viva/Business Context.
    """
    print("\n" + "="*50)
    print(" VIVA & BUSINESS EXPLANATIONS ")
    print("="*50)
    print("1. Business Impact:")
    print("   Banks process millions of transactions. False Negatives (missing a fraud) directly cost the bank money and damage customer trust.")
    print("   False Positives (blocking a normal purchase) annoy customers but don't result in direct theft.")
    print("   Therefore, we prioritize RECALL to catch as much fraud as possible, even at the expense of a drop in Precision.")
    
    print("\n2. Threshold Tuning:")
    print("   By default, models use a 0.5 threshold. However, for fraud, a 0.5 threshold might be too strict.")
    print("   Lowering the threshold to 0.3 or 0.2 massively increases Recall (catches more fraud) by classifying more 'suspicious' transactions as fraud.")
    
    print("\n3. Setup Limitations:")
    print("   - V1-V28 are PCA transformed. We cannot explain exactly what they represent in real life.")
    print("   - Our User_ID and aggregated 'historical' metrics are simulated and might behave differently with real historical tracking systems.")
    print("="*50 + "\n")

if __name__ == "__main__":
    train_and_evaluate()
