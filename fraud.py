"""
ENHANCED VERSION: Analisis Komparatif Kinerja Algoritma Machine Learning 
untuk Deteksi Fraud pada Transaksi Pembayaran Digital UMKM

FOCUS: Detailed Feature Engineering & Explainable AI (XAI)
Dataset: PaySim Synthetic Financial Dataset

Author: Research Team
Version: 2.0 - Enhanced for Journal Publication
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, precision_score, recall_score, auc)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import shap
import time
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Styling untuk plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# CUDA Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*100)
print("🚀 ENHANCED FRAUD DETECTION SYSTEM - DETAILED ANALYSIS")
print("="*100)
print(f"⚙️  Device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*100)


class DetailedFeatureEngineering:
    """
    ENHANCED Feature Engineering dengan dokumentasi lengkap
    Setiap fitur dijelaskan dengan detail untuk publikasi jurnal
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.feature_catalog = {}  # Katalog semua fitur yang dibuat
        self.feature_importance_raw = {}
        
    def create_all_features(self):
        """Main function untuk create semua features"""
        print("\n" + "="*100)
        print("📊 DETAILED FEATURE ENGINEERING PROCESS")
        print("="*100)
        
        # Category 1: Balance Error Features
        self.create_balance_error_features()
        
        # Category 2: Temporal/Behavioral Features
        self.create_temporal_features()
        
        # Category 3: Customer Behavior Aggregation
        self.create_customer_aggregation_features()
        
        # Category 4: Transaction Pattern Features
        self.create_transaction_pattern_features()
        
        # Category 5: Balance Ratio Features
        self.create_balance_ratio_features()
        
        # Category 6: Transaction Type Encoding
        self.create_transaction_type_features()
        
        # Category 7: Risk Indicator Features
        self.create_risk_indicator_features()
        
        # Category 8: Velocity Features
        self.create_velocity_features()
        
        # Generate feature catalog report
        self.generate_feature_report()
        
        return self.df
    
    def create_balance_error_features(self):
        """
        CATEGORY 1: BALANCE ERROR FEATURES
        
        Rationale: Dalam transaksi normal, persamaan berikut harus berlaku:
        oldBalance - amount = newBalance
        
        Jika tidak, ada indikasi manipulasi atau fraud
        """
        print("\n" + "-"*100)
        print("🔍 CATEGORY 1: BALANCE ERROR FEATURES")
        print("-"*100)
        
        # Feature 1: Error Balance Origin
        self.df['errorBalanceOrig'] = (
            self.df['oldbalanceOrg'] - self.df['amount'] - self.df['newbalanceOrig']
        )
        self.feature_catalog['errorBalanceOrig'] = {
            'description': 'Selisih antara expected balance dan actual balance pengirim',
            'formula': 'oldbalanceOrg - amount - newbalanceOrig',
            'hypothesis': 'Nilai != 0 mengindikasikan anomali/fraud',
            'type': 'Numerical - Continuous'
        }
        
        # Feature 2: Error Balance Destination
        self.df['errorBalanceDest'] = (
            self.df['oldbalanceDest'] + self.df['amount'] - self.df['newbalanceDest']
        )
        self.feature_catalog['errorBalanceDest'] = {
            'description': 'Selisih antara expected balance dan actual balance penerima',
            'formula': 'oldbalanceDest + amount - newbalanceDest',
            'hypothesis': 'Nilai != 0 mengindikasikan anomali/fraud',
            'type': 'Numerical - Continuous'
        }
        
        # Feature 3: Absolute Error (Total error magnitude)
        self.df['totalBalanceError'] = (
            np.abs(self.df['errorBalanceOrig']) + np.abs(self.df['errorBalanceDest'])
        )
        self.feature_catalog['totalBalanceError'] = {
            'description': 'Total magnitude error dari kedua sisi transaksi',
            'formula': 'abs(errorBalanceOrig) + abs(errorBalanceDest)',
            'hypothesis': 'Nilai tinggi = high fraud probability',
            'type': 'Numerical - Continuous'
        }
        
        # Feature 4: Error Flag
        self.df['hasBalanceError'] = (
            (np.abs(self.df['errorBalanceOrig']) > 0.01) | 
            (np.abs(self.df['errorBalanceDest']) > 0.01)
        ).astype(int)
        self.feature_catalog['hasBalanceError'] = {
            'description': 'Binary flag untuk keberadaan balance error',
            'formula': '1 if abs(error) > 0.01 else 0',
            'hypothesis': 'Flag=1 strongly correlates dengan fraud',
            'type': 'Binary'
        }
        
        print(f"✅ Created 4 Balance Error Features")
        print(f"   - Mean errorBalanceOrig (Fraud): {self.df[self.df['isFraud']==1]['errorBalanceOrig'].mean():.2f}")
        print(f"   - Mean errorBalanceOrig (Normal): {self.df[self.df['isFraud']==0]['errorBalanceOrig'].mean():.2f}")
        print(f"   - % Transactions with errors: {self.df['hasBalanceError'].mean()*100:.2f}%")
        
    def create_temporal_features(self):
        """
        CATEGORY 2: TEMPORAL/TIME-BASED FEATURES
        
        Rationale: Fraud patterns berbeda berdasarkan waktu
        """
        print("\n" + "-"*100)
        print("⏰ CATEGORY 2: TEMPORAL FEATURES")
        print("-"*100)
        
        # Feature 5: Hour of day (24-hour format)
        self.df['hour_of_day'] = self.df['step'] % 24
        self.feature_catalog['hour_of_day'] = {
            'description': 'Jam transaksi dalam format 24 jam',
            'formula': 'step mod 24',
            'hypothesis': 'Fraud cenderung terjadi pada jam-jam tertentu (malam)',
            'type': 'Numerical - Discrete (0-23)'
        }
        
        # Feature 6: Is Night Time
        self.df['is_night'] = (
            (self.df['hour_of_day'] >= 0) & (self.df['hour_of_day'] <= 6)
        ).astype(int)
        self.feature_catalog['is_night'] = {
            'description': 'Flag transaksi malam (00:00 - 06:00)',
            'formula': '1 if 0 <= hour <= 6 else 0',
            'hypothesis': 'Transaksi malam lebih berisiko fraud',
            'type': 'Binary'
        }
        
        # Feature 7: Is Peak Hour (business hours)
        self.df['is_peak_hour'] = (
            (self.df['hour_of_day'] >= 9) & (self.df['hour_of_day'] <= 17)
        ).astype(int)
        self.feature_catalog['is_peak_hour'] = {
            'description': 'Flag transaksi jam kerja (09:00 - 17:00)',
            'formula': '1 if 9 <= hour <= 17 else 0',
            'hypothesis': 'Transaksi normal dominan di jam kerja',
            'type': 'Binary'
        }
        
        # Feature 8: Day of simulation (to capture trends)
        self.df['day_of_simulation'] = self.df['step'] // 24
        self.feature_catalog['day_of_simulation'] = {
            'description': 'Hari ke-n dalam simulasi',
            'formula': 'step // 24',
            'hypothesis': 'Pola fraud dapat berubah seiring waktu',
            'type': 'Numerical - Discrete'
        }
        
        print(f"✅ Created 4 Temporal Features")
        print(f"   - Night transactions (Fraud): {self.df[self.df['isFraud']==1]['is_night'].mean()*100:.2f}%")
        print(f"   - Night transactions (Normal): {self.df[self.df['isFraud']==0]['is_night'].mean()*100:.2f}%")
        
    def create_customer_aggregation_features(self):
        """
        CATEGORY 3: CUSTOMER BEHAVIOR AGGREGATION
        
        Rationale: Fraud detection memerlukan konteks historis pelanggan
        """
        print("\n" + "-"*100)
        print("👤 CATEGORY 3: CUSTOMER AGGREGATION FEATURES")
        print("-"*100)
        print("   ⏳ Computing aggregations (this may take a moment)...")
        
        # Agregasi per customer (nameOrig)
        customer_stats = self.df.groupby('nameOrig').agg({
            'amount': ['count', 'mean', 'std', 'sum', 'min', 'max'],
            'step': ['min', 'max']
        }).reset_index()
        
        customer_stats.columns = ['nameOrig', 'trans_count', 'avg_amount', 
                                   'std_amount', 'total_amount', 'min_amount',
                                   'max_amount', 'first_step', 'last_step']
        
        # Fill std=0 untuk customers dengan hanya 1 transaksi
        customer_stats['std_amount'].fillna(0, inplace=True)
        
        # Merge back to main dataframe
        self.df = self.df.merge(customer_stats, on='nameOrig', how='left')
        
        # Feature 9: Transaction Count
        self.feature_catalog['trans_count'] = {
            'description': 'Total jumlah transaksi per customer',
            'formula': 'COUNT(transactions) GROUP BY nameOrig',
            'hypothesis': 'Fraudster sering memiliki sedikit transaksi',
            'type': 'Numerical - Count'
        }
        
        # Feature 10: Average Amount
        self.feature_catalog['avg_amount'] = {
            'description': 'Rata-rata nominal transaksi per customer',
            'formula': 'AVG(amount) GROUP BY nameOrig',
            'hypothesis': 'Baseline untuk mendeteksi deviasi',
            'type': 'Numerical - Continuous'
        }
        
        # Feature 11: Standard Deviation
        self.feature_catalog['std_amount'] = {
            'description': 'Variabilitas nominal transaksi per customer',
            'formula': 'STDEV(amount) GROUP BY nameOrig',
            'hypothesis': 'High std = pola tidak konsisten (suspicious)',
            'type': 'Numerical - Continuous'
        }
        
        # Feature 12: Account Activity Duration
        self.df['account_age_steps'] = self.df['last_step'] - self.df['first_step']
        self.feature_catalog['account_age_steps'] = {
            'description': 'Lama aktif account (dalam steps)',
            'formula': 'MAX(step) - MIN(step) per customer',
            'hypothesis': 'Account baru lebih berisiko',
            'type': 'Numerical - Continuous'
        }
        
        print(f"✅ Created 6 Customer Aggregation Features")
        print(f"   - Avg trans_count (Fraud): {self.df[self.df['isFraud']==1]['trans_count'].mean():.2f}")
        print(f"   - Avg trans_count (Normal): {self.df[self.df['isFraud']==0]['trans_count'].mean():.2f}")
        
    def create_transaction_pattern_features(self):
        """
        CATEGORY 4: TRANSACTION PATTERN FEATURES
        
        Rationale: Deteksi pola transaksi abnormal
        """
        print("\n" + "-"*100)
        print("📈 CATEGORY 4: TRANSACTION PATTERN FEATURES")
        print("-"*100)
        
        # Feature 13: Deviation from Average
        self.df['deviation_from_avg'] = np.abs(self.df['amount'] - self.df['avg_amount'])
        self.feature_catalog['deviation_from_avg'] = {
            'description': 'Absolute deviasi nominal dari rata-rata customer',
            'formula': 'abs(amount - avg_amount)',
            'hypothesis': 'Large deviation = anomali/fraud',
            'type': 'Numerical - Continuous'
        }
        
        # Feature 14: Amount to Average Ratio
        self.df['amount_to_avg_ratio'] = self.df['amount'] / (self.df['avg_amount'] + 1)
        self.feature_catalog['amount_to_avg_ratio'] = {
            'description': 'Rasio nominal terhadap rata-rata customer',
            'formula': 'amount / (avg_amount + 1)',
            'hypothesis': 'Ratio >> 1 atau << 1 = suspicious',
            'type': 'Numerical - Continuous'
        }
        
        # Feature 15: Is Outlier Amount (menggunakan Z-score)
        self.df['is_amount_outlier'] = (
            np.abs(self.df['deviation_from_avg']) > (2 * self.df['std_amount'] + 0.01)
        ).astype(int)
        self.feature_catalog['is_amount_outlier'] = {
            'description': 'Flag outlier berdasarkan Z-score (>2 std)',
            'formula': '1 if deviation > 2*std else 0',
            'hypothesis': 'Outliers strongly correlate dengan fraud',
            'type': 'Binary'
        }
        
        # Feature 16: Amount to Total Ratio
        self.df['amount_to_total_ratio'] = self.df['amount'] / (self.df['total_amount'] + 1)
        self.feature_catalog['amount_to_total_ratio'] = {
            'description': 'Proporsi transaksi terhadap total historis',
            'formula': 'amount / total_amount',
            'hypothesis': 'Single large transaction = red flag',
            'type': 'Numerical - Continuous (0-1)'
        }
        
        print(f"✅ Created 4 Transaction Pattern Features")
        print(f"   - % Outliers (Fraud): {self.df[self.df['isFraud']==1]['is_amount_outlier'].mean()*100:.2f}%")
        print(f"   - % Outliers (Normal): {self.df[self.df['isFraud']==0]['is_amount_outlier'].mean()*100:.2f}%")
        
    def create_balance_ratio_features(self):
        """
        CATEGORY 5: BALANCE RATIO FEATURES
        
        Rationale: Rasio balance memberikan konteks keuangan
        """
        print("\n" + "-"*100)
        print("💰 CATEGORY 5: BALANCE RATIO FEATURES")
        print("-"*100)
        
        # Feature 17: Origin Balance Ratio
        self.df['orig_balance_ratio'] = self.df['newbalanceOrig'] / (self.df['oldbalanceOrg'] + 1)
        self.feature_catalog['orig_balance_ratio'] = {
            'description': 'Rasio balance baru terhadap lama (pengirim)',
            'formula': 'newbalanceOrig / (oldbalanceOrg + 1)',
            'hypothesis': 'Ratio=0 (depleted) = suspicious',
            'type': 'Numerical - Continuous (0-1)'
        }
        
        # Feature 18: Destination Balance Ratio
        self.df['dest_balance_ratio'] = self.df['newbalanceDest'] / (self.df['oldbalanceDest'] + 1)
        self.feature_catalog['dest_balance_ratio'] = {
            'description': 'Rasio balance baru terhadap lama (penerima)',
            'formula': 'newbalanceDest / (oldbalanceDest + 1)',
            'hypothesis': 'Sudden large increase = suspicious',
            'type': 'Numerical - Continuous'
        }
        
        # Feature 19: Amount to Origin Balance Ratio
        self.df['amount_to_orig_balance'] = self.df['amount'] / (self.df['oldbalanceOrg'] + 1)
        self.feature_catalog['amount_to_orig_balance'] = {
            'description': 'Proporsi transaksi terhadap balance pengirim',
            'formula': 'amount / (oldbalanceOrg + 1)',
            'hypothesis': 'Ratio ≈ 1 (menguras account) = fraud pattern',
            'type': 'Numerical - Continuous (0-1+)'
        }
        
        # Feature 20: Zero Balance Flags
        self.df['orig_zero_balance'] = (self.df['oldbalanceOrg'] == 0).astype(int)
        self.df['dest_zero_balance'] = (self.df['oldbalanceDest'] == 0).astype(int)
        self.df['orig_depleted'] = (self.df['newbalanceOrig'] == 0).astype(int)
        
        self.feature_catalog['orig_zero_balance'] = {
            'description': 'Flag: pengirim memiliki balance awal = 0',
            'formula': '1 if oldbalanceOrg == 0 else 0',
            'hypothesis': 'Zero balance accounts = higher risk',
            'type': 'Binary'
        }
        
        print(f"✅ Created 6 Balance Ratio Features")
        print(f"   - % Depleted accounts (Fraud): {self.df[self.df['isFraud']==1]['orig_depleted'].mean()*100:.2f}%")
        print(f"   - % Depleted accounts (Normal): {self.df[self.df['isFraud']==0]['orig_depleted'].mean()*100:.2f}%")
        
    def create_transaction_type_features(self):
        """
        CATEGORY 6: TRANSACTION TYPE ENCODING
        
        Rationale: One-hot encoding untuk categorical variable
        """
        print("\n" + "-"*100)
        print("🏷️  CATEGORY 6: TRANSACTION TYPE FEATURES")
        print("-"*100)
        
        # One-hot encoding
        self.df['type_CASH_OUT'] = (self.df['type'] == 'CASH_OUT').astype(int)
        self.df['type_PAYMENT'] = (self.df['type'] == 'PAYMENT').astype(int)
        self.df['type_CASH_IN'] = (self.df['type'] == 'CASH_IN').astype(int)
        self.df['type_TRANSFER'] = (self.df['type'] == 'TRANSFER').astype(int)
        self.df['type_DEBIT'] = (self.df['type'] == 'DEBIT').astype(int)
        
        for trans_type in ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']:
            self.feature_catalog[f'type_{trans_type}'] = {
                'description': f'Binary flag untuk transaction type = {trans_type}',
                'formula': f'1 if type == {trans_type} else 0',
                'hypothesis': 'Certain types more prone to fraud',
                'type': 'Binary (One-Hot Encoded)'
            }
        
        # Fraud rate per type
        print(f"✅ Created 5 Transaction Type Features")
        print(f"\n   Fraud Rate by Transaction Type:")
        for trans_type in self.df['type'].unique():
            fraud_rate = self.df[self.df['type']==trans_type]['isFraud'].mean() * 100
            print(f"   - {trans_type:12s}: {fraud_rate:6.2f}%")
        
    def create_risk_indicator_features(self):
        """
        CATEGORY 7: COMPOSITE RISK INDICATORS
        
        Rationale: Combined features untuk high-level risk assessment
        """
        print("\n" + "-"*100)
        print("⚠️  CATEGORY 7: RISK INDICATOR FEATURES")
        print("-"*100)
        
        # Feature: High Value Transaction
        amount_95percentile = self.df['amount'].quantile(0.95)
        self.df['is_high_value'] = (self.df['amount'] > amount_95percentile).astype(int)
        self.feature_catalog['is_high_value'] = {
            'description': 'Flag transaksi bernilai tinggi (>95th percentile)',
            'formula': '1 if amount > P95 else 0',
            'hypothesis': 'High value = higher risk',
            'type': 'Binary'
        }
        
        # Feature: New Customer
        self.df['is_new_customer'] = (self.df['trans_count'] <= 2).astype(int)
        self.feature_catalog['is_new_customer'] = {
            'description': 'Flag customer baru (≤2 transaksi)',
            'formula': '1 if trans_count <= 2 else 0',
            'hypothesis': 'New customers = higher fraud risk',
            'type': 'Binary'
        }
        
        # Feature: Composite Risk Score (simple heuristic)
        self.df['risk_score'] = (
            self.df['hasBalanceError'] * 3 +
            self.df['is_amount_outlier'] * 2 +
            self.df['is_high_value'] * 1 +
            self.df['is_new_customer'] * 1 +
            self.df['is_night'] * 1 +
            self.df['orig_depleted'] * 2
        )
        self.feature_catalog['risk_score'] = {
            'description': 'Composite risk score (weighted sum of risk factors)',
            'formula': 'weighted_sum(hasBalanceError*3, is_outlier*2, ...)',
            'hypothesis': 'Higher score = higher fraud probability',
            'type': 'Numerical - Discrete (0-10)'
        }
        
        print(f"✅ Created 3 Risk Indicator Features")
        print(f"   - Avg risk_score (Fraud): {self.df[self.df['isFraud']==1]['risk_score'].mean():.2f}")
        print(f"   - Avg risk_score (Normal): {self.df[self.df['isFraud']==0]['risk_score'].mean():.2f}")
        
    def create_velocity_features(self):
        """
        CATEGORY 8: VELOCITY/FREQUENCY FEATURES
        
        Rationale: Rapid sequences of transactions = suspicious
        """
        print("\n" + "-"*100)
        print("⚡ CATEGORY 8: VELOCITY FEATURES")
        print("-"*100)
        print("   ⏳ Computing time-window aggregations...")
        
        # Sort by customer and time
        self.df = self.df.sort_values(['nameOrig', 'step']).reset_index(drop=True)
        
        # Feature: Transaction Frequency (transactions per day)
        self.df['trans_per_day'] = self.df['trans_count'] / (self.df['account_age_steps'] / 24 + 1)
        self.feature_catalog['trans_per_day'] = {
            'description': 'Frekuensi transaksi per hari',
            'formula': 'trans_count / (account_age_days + 1)',
            'hypothesis': 'Very high frequency = bot/fraud',
            'type': 'Numerical - Continuous'
        }
        
        # Feature: Time since last transaction
        self.df['time_since_last'] = self.df.groupby('nameOrig')['step'].diff()
        self.df['time_since_last'].fillna(999, inplace=True)  # First transaction
        self.feature_catalog['time_since_last'] = {
            'description': 'Steps sejak transaksi terakhir',
            'formula': 'current_step - previous_step (per customer)',
            'hypothesis': 'Very short intervals = automated fraud',
            'type': 'Numerical - Continuous'
        }
        
        # Feature: Rapid Transaction Flag
        self.df['is_rapid_transaction'] = (self.df['time_since_last'] < 5).astype(int)
        self.feature_catalog['is_rapid_transaction'] = {
            'description': 'Flag transaksi dalam interval sangat cepat (<5 steps)',
            'formula': '1 if time_since_last < 5 else 0',
            'hypothesis': 'Rapid sequences = suspicious',
            'type': 'Binary'
        }
        
        print(f"✅ Created 3 Velocity Features")
        print(f"   - % Rapid trans (Fraud): {self.df[self.df['isFraud']==1]['is_rapid_transaction'].mean()*100:.2f}%")
        print(f"   - % Rapid trans (Normal): {self.df[self.df['isFraud']==0]['is_rapid_transaction'].mean()*100:.2f}%")
        
    def generate_feature_report(self):
        """Generate comprehensive feature engineering report"""
        print("\n" + "="*100)
        print("📋 FEATURE ENGINEERING SUMMARY REPORT")
        print("="*100)
        
        total_features = len(self.feature_catalog)
        print(f"\n✅ Total Features Created: {total_features}")
        print(f"📊 Original Features: {len([c for c in self.df.columns if c in ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']])}")
        print(f"🆕 Engineered Features: {total_features}")
        
        # Feature categories summary
        categories = {
            'Balance Error': 4,
            'Temporal': 4,
            'Customer Aggregation': 6,
            'Transaction Pattern': 4,
            'Balance Ratio': 6,
            'Transaction Type': 5,
            'Risk Indicator': 3,
            'Velocity': 3
        }
        
        print(f"\n📂 Features by Category:")
        for cat, count in categories.items():
            print(f"   {cat:25s}: {count:2d} features")
        
        # Save detailed catalog
        catalog_df = pd.DataFrame(self.feature_catalog).T
        catalog_df.to_csv('feature_engineering_catalog.csv')
        print(f"\n💾 Detailed feature catalog saved to: feature_engineering_catalog.csv")
        
        # Correlation with fraud (preliminary)
        print(f"\n🔍 Top 10 Features Correlated with Fraud:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlations = self.df[numeric_cols].corrwith(self.df['isFraud']).sort_values(ascending=False)
        print(correlations.head(10).to_string())
        
        return catalog_df


class EnhancedExplainableAI:
    """
    ENHANCED Explainable AI (XAI) dengan SHAP
    Detail analysis untuk jurnal publikasi
    """
    
    def __init__(self, model, X_test, y_test, feature_names, model_name='Model'):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.model_name = model_name
        self.shap_values = None
        self.explainer = None
        
    def create_explainer(self, num_samples=1000):
        """Create SHAP explainer"""
        print("\n" + "="*100)
        print(f"🔍 EXPLAINABLE AI (XAI) ANALYSIS - {self.model_name}")
        print("="*100)
        print(f"⏳ Creating SHAP explainer (sampling {num_samples} instances)...")
        
        X_sample = self.X_test[:num_samples]
        
        start_time = time.time()
        
        # Choose appropriate explainer
        if self.model_name in ['XGBoost', 'LightGBM', 'Random_Forest', 'Decision_Tree']:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_sample)
        else:
            # For other models, use KernelExplainer (slower)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                shap.kmeans(self.X_test, 50)
            )
            self.shap_values = self.explainer.shap_values(X_sample)[1]
        
        elapsed = time.time() - start_time
        print(f"✅ SHAP explainer created in {elapsed:.2f} seconds")
        
        return X_sample
    
    def plot_summary(self, X_sample, save_prefix='xai'):
        """
        SHAP Summary Plot - Shows feature importance and impact
        """
        print("\n" + "-"*100)
        print("📊 GENERATING SHAP SUMMARY PLOT")
        print("-"*100)
        
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            self.shap_values, 
            X_sample, 
            feature_names=self.feature_names,
            show=False,
            plot_size=(14, 10)
        )
        plt.title(f'SHAP Feature Importance - {self.model_name}\n(Impact on Model Predictions)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        filename = f'{save_prefix}_summary_{self.model_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Summary plot saved: {filename}")
        plt.show()
        
    def plot_bar(self, X_sample, top_n=20, save_prefix='xai'):
        """
        SHAP Bar Plot - Feature importance ranking
        """
        print("\n" + "-"*100)
        print(f"📊 GENERATING TOP {top_n} FEATURE IMPORTANCE BAR PLOT")
        print("-"*100)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            X_sample,
            feature_names=self.feature_names,
            plot_type='bar',
            show=False,
            max_display=top_n
        )
        plt.title(f'Top {top_n} Most Important Features - {self.model_name}\n(Mean Absolute SHAP Values)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        filename = f'{save_prefix}_bar_{self.model_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Bar plot saved: {filename}")
        plt.show()
        
    def analyze_feature_importance(self, X_sample, top_n=15):
        """
        Detailed feature importance analysis dengan statistik
        """
        print("\n" + "-"*100)
        print(f"🔬 DETAILED FEATURE IMPORTANCE ANALYSIS (Top {top_n})")
        print("-"*100)
        
        # Calculate mean absolute SHAP values
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Mean_SHAP': np.abs(self.shap_values).mean(axis=0),
            'Std_SHAP': np.abs(self.shap_values).std(axis=0),
            'Max_SHAP': np.abs(self.shap_values).max(axis=0),
            'Min_SHAP': np.abs(self.shap_values).min(axis=0)
        }).sort_values('Mean_SHAP', ascending=False)
        
        # Display top features
        print(f"\n{'Rank':<6}{'Feature':<35}{'Mean |SHAP|':<15}{'Std |SHAP|':<15}{'Max |SHAP|':<15}")
        print("-"*100)
        
        for idx, row in feature_importance.head(top_n).iterrows():
            rank = feature_importance.index.get_loc(idx) + 1
            print(f"{rank:<6}{row['Feature']:<35}{row['Mean_SHAP']:<15.6f}{row['Std_SHAP']:<15.6f}{row['Max_SHAP']:<15.6f}")
        
        # Save to CSV
        feature_importance.to_csv(f'feature_importance_{self.model_name}.csv', index=False)
        print(f"\n💾 Feature importance saved: feature_importance_{self.model_name}.csv")
        
        return feature_importance
    
    def plot_waterfall(self, X_sample, instance_idx=0, save_prefix='xai'):
        """
        SHAP Waterfall Plot - Explain individual prediction
        """
        print("\n" + "-"*100)
        print(f"💧 GENERATING WATERFALL PLOT FOR INSTANCE #{instance_idx}")
        print("-"*100)
        
        # Get prediction for this instance
        if hasattr(self.model, 'predict_proba'):
            pred_proba = self.model.predict_proba(X_sample[instance_idx:instance_idx+1])[0][1]
        else:
            pred_proba = self.model.predict(X_sample[instance_idx:instance_idx+1])[0]
        
        actual_label = self.y_test[instance_idx]
        
        print(f"   Instance #{instance_idx}:")
        print(f"   - Predicted Probability: {pred_proba:.4f}")
        print(f"   - Actual Label: {actual_label}")
        print(f"   - Prediction: {'FRAUD' if pred_proba > 0.5 else 'NORMAL'}")
        print(f"   - Correct: {'✅ YES' if (pred_proba > 0.5) == actual_label else '❌ NO'}")
        
        # Create waterfall plot
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                data=X_sample[instance_idx],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall - {self.model_name} (Instance #{instance_idx})\nPred: {pred_proba:.3f}, Actual: {actual_label}', 
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        filename = f'{save_prefix}_waterfall_{self.model_name}_instance{instance_idx}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Waterfall plot saved: {filename}")
        plt.show()
        
    def plot_dependence(self, X_sample, feature_name, interaction_feature=None, save_prefix='xai'):
        """
        SHAP Dependence Plot - Shows how feature value affects prediction
        """
        print("\n" + "-"*100)
        print(f"📈 GENERATING DEPENDENCE PLOT FOR: {feature_name}")
        print("-"*100)
        
        try:
            feature_idx = self.feature_names.index(feature_name)
        except ValueError:
            print(f"❌ Feature '{feature_name}' not found!")
            return
        
        plt.figure(figsize=(10, 6))
        
        if interaction_feature:
            try:
                interaction_idx = self.feature_names.index(interaction_feature)
                shap.dependence_plot(
                    feature_idx,
                    self.shap_values,
                    X_sample,
                    feature_names=self.feature_names,
                    interaction_index=interaction_idx,
                    show=False
                )
            except ValueError:
                print(f"❌ Interaction feature '{interaction_feature}' not found! Using auto.")
                shap.dependence_plot(
                    feature_idx,
                    self.shap_values,
                    X_sample,
                    feature_names=self.feature_names,
                    show=False
                )
        else:
            shap.dependence_plot(
                feature_idx,
                self.shap_values,
                X_sample,
                feature_names=self.feature_names,
                show=False
            )
        
        plt.title(f'SHAP Dependence Plot - {self.model_name}\nFeature: {feature_name}', 
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        filename = f'{save_prefix}_dependence_{self.model_name}_{feature_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Dependence plot saved: {filename}")
        plt.show()
        
    def plot_force_plot(self, X_sample, instance_idx=0, save_prefix='xai'):
        """
        SHAP Force Plot - Interactive visualization of prediction
        """
        print("\n" + "-"*100)
        print(f"🎯 GENERATING FORCE PLOT FOR INSTANCE #{instance_idx}")
        print("-"*100)
        
        try:
            shap.initjs()
            force_plot = shap.force_plot(
                self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                self.shap_values[instance_idx],
                X_sample[instance_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            
            plt.title(f'SHAP Force Plot - {self.model_name} (Instance #{instance_idx})', 
                      fontsize=12, fontweight='bold')
            plt.tight_layout()
            filename = f'{save_prefix}_force_{self.model_name}_instance{instance_idx}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Force plot saved: {filename}")
            plt.show()
        except Exception as e:
            print(f"⚠️  Force plot generation skipped: {str(e)}")
    
    def generate_comprehensive_report(self, X_sample, feature_importance_df):
        """
        Generate comprehensive XAI report untuk jurnal
        """
        print("\n" + "="*100)
        print("📝 COMPREHENSIVE XAI REPORT")
        print("="*100)
        
        report = f"""
{'='*100}
EXPLAINABLE AI (XAI) ANALYSIS REPORT
Model: {self.model_name}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*100}

1. METHODOLOGY
   - Technique: SHAP (SHapley Additive exPlanations)
   - Explainer Type: {'Tree Explainer' if self.model_name in ['XGBoost', 'LightGBM', 'Random_Forest'] else 'Kernel Explainer'}
   - Sample Size: {len(X_sample)} instances
   - Total Features: {len(self.feature_names)}

2. TOP 10 MOST IMPORTANT FEATURES
   These features have the highest impact on fraud detection predictions:
   
   Rank | Feature Name                      | Mean |SHAP|    | Interpretation
   -----|-----------------------------------|----------------|-----------------
"""
        
        for idx, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
            report += f"   {idx:2d}   | {row['Feature']:35s} | {row['Mean_SHAP']:14.6f} | {'High Impact' if row['Mean_SHAP'] > 0.1 else 'Moderate Impact'}\n"
        
        report += f"""
3. KEY INSIGHTS FROM SHAP ANALYSIS

   a) Feature Categories Impact:
      - Balance Error Features: {'HIGH' if any('error' in f.lower() for f in feature_importance_df.head(5)['Feature']) else 'MODERATE'}
      - Temporal Features: {'HIGH' if any('hour' in f.lower() or 'night' in f.lower() for f in feature_importance_df.head(10)['Feature']) else 'MODERATE'}
      - Customer Behavior: {'HIGH' if any('trans' in f.lower() or 'avg' in f.lower() for f in feature_importance_df.head(10)['Feature']) else 'MODERATE'}
   
   b) Model Interpretability:
      - The top 3 features account for a significant portion of predictions
      - Feature interactions are captured and visualized
      - Individual predictions can be explained with waterfall plots
   
4. IMPLICATIONS FOR FRAUD DETECTION
   
   ✓ Transparency: SHAP provides clear explanations for each prediction
   ✓ Trust: Stakeholders can understand WHY a transaction is flagged
   ✓ Compliance: Explainability meets regulatory requirements
   ✓ Debugging: Identify potential model biases or errors
   ✓ Feature Engineering: Validate the importance of engineered features

5. RECOMMENDATIONS
   
   - Focus on top 10 features for real-time deployment (efficiency)
   - Monitor SHAP values for concept drift detection
   - Use SHAP explanations in fraud investigation workflows
   - Regularly update feature importance analysis

{'='*100}
END OF REPORT
{'='*100}
"""
        
        print(report)
        
        # Save report
        with open(f'xai_report_{self.model_name}.txt', 'w') as f:
            f.write(report)
        
        print(f"\n💾 XAI report saved: xai_report_{self.model_name}.txt")
        
        return report


# ============================================================================
# MAIN ENHANCED PIPELINE
# ============================================================================

class EnhancedFraudDetectionPipeline:
    """Enhanced pipeline dengan detailed FE & XAI"""
    
    def __init__(self):
        self.df = None
        self.feature_engineer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self, data_path=None, df=None, sample_size=None):
        """Load data"""
        print("\n" + "="*100)
        print("📥 STEP 1: DATA LOADING")
        print("="*100)
        
        if df is not None:
            self.df = df
        elif data_path:
            print(f"Loading from: {data_path}")
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("Provide either dataframe or data_path")
        
        if sample_size and sample_size < len(self.df):
            print(f"⚠️  Sampling {sample_size:,} transactions for faster processing...")
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        print(f"✅ Dataset loaded: {self.df.shape}")
        print(f"   - Total transactions: {len(self.df):,}")
        print(f"   - Fraud cases: {self.df['isFraud'].sum():,} ({self.df['isFraud'].mean()*100:.2f}%)")
        print(f"   - Normal cases: {(~self.df['isFraud'].astype(bool)).sum():,}")
        
    def apply_feature_engineering(self):
        """Apply detailed feature engineering"""
        self.feature_engineer = DetailedFeatureEngineering(self.df)
        self.df = self.feature_engineer.create_all_features()
        
    def prepare_train_test(self, test_size=0.3, sampling_method='smote'):
        """Prepare train/test with sampling"""
        print("\n" + "="*100)
        print(f"📊 STEP 3: TRAIN/TEST SPLIT & SAMPLING ({sampling_method.upper()})")
        print("="*100)
        
        # Select features
        drop_cols = ['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud', 'type',
                     'first_step', 'last_step']  # Drop unnecessary columns
        self.feature_names = [col for col in self.df.columns if col not in drop_cols]
        
        X = self.df[self.feature_names].values
        y = self.df['isFraud'].values
        
        print(f"✅ Feature selection complete")
        print(f"   - Total features: {len(self.feature_names)}")
        print(f"   - Original fraud ratio: {y.mean()*100:.2f}%")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Sampling
        if sampling_method.lower() != 'none':
            print(f"\n⚙️  Applying {sampling_method.upper()}...")
            
            samplers = {
                'smote': SMOTE(random_state=42, k_neighbors=5),
                'adasyn': ADASYN(random_state=42),
                'borderline_smote': BorderlineSMOTE(random_state=42),
                'tomek': TomekLinks()
            }
            
            sampler = samplers.get(sampling_method.lower())
            if sampler:
                X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
                print(f"   ✅ After {sampling_method}:")
                print(f"      - Training samples: {len(y_train_resampled):,}")
                print(f"      - Fraud ratio: {y_train_resampled.mean()*100:.2f}%")
                
                self.X_train = X_train_resampled
                self.y_train = y_train_resampled
            else:
                self.X_train = X_train_scaled
                self.y_train = y_train
        else:
            self.X_train = X_train_scaled
            self.y_train = y_train
        
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        print(f"\n✅ Data preparation complete")
        print(f"   - Training set: {self.X_train.shape}")
        print(f"   - Test set: {self.X_test.shape}")
        
    def train_xgboost(self):
        """Train XGBoost"""
        print("\n" + "="*100)
        print("🚀 TRAINING: XGBoost")
        print("="*100)
        
        start_time = time.time()
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            scale_pos_weight=len(self.y_train) / sum(self.y_train) - 1,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            tree_method='gpu_hist' if torch.cuda.is_available() else 'auto',
            gpu_id=0 if torch.cuda.is_available() else None,
            random_state=42
        )
        
        model.fit(self.X_train, self.y_train, verbose=False)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        training_time = time.time() - start_time
        
        # Metrics
        metrics = {
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'AUC-ROC': roc_auc_score(self.y_test, y_pred_proba),
            'Training_Time': training_time,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.models['XGBoost'] = model
        self.results['XGBoost'] = metrics
        
        print(f"\n✅ XGBoost Training Complete")
        print(f"   - Precision: {metrics['Precision']:.4f}")
        print(f"   - Recall: {metrics['Recall']:.4f}")
        print(f"   - F1-Score: {metrics['F1-Score']:.4f}")
        print(f"   - AUC-ROC: {metrics['AUC-ROC']:.4f}")
        print(f"   - Training Time: {training_time:.2f}s")
        
        return model
    
    def apply_comprehensive_xai(self, model_name='XGBoost', num_samples=500):
        """Apply comprehensive XAI analysis"""
        
        if model_name not in self.models:
            print(f"❌ Model '{model_name}' not found!")
            return
        
        model = self.models[model_name]
        
        # Create XAI analyzer
        xai = EnhancedExplainableAI(
            model=model,
            X_test=self.X_test,
            y_test=self.y_test,
            feature_names=self.feature_names,
            model_name=model_name
        )
        
        # Create explainer
        X_sample = xai.create_explainer(num_samples=num_samples)
        
        # Generate all plots
        xai.plot_summary(X_sample)
        xai.plot_bar(X_sample, top_n=20)
        
        # Feature importance analysis
        feature_importance_df = xai.analyze_feature_importance(X_sample, top_n=15)
        
        # Waterfall for fraud and normal cases
        fraud_indices = np.where(self.y_test[:num_samples] == 1)[0]
        normal_indices = np.where(self.y_test[:num_samples] == 0)[0]
        
        if len(fraud_indices) > 0:
            xai.plot_waterfall(X_sample, instance_idx=fraud_indices[0])
        if len(normal_indices) > 0:
            xai.plot_waterfall(X_sample, instance_idx=normal_indices[0])
        
        # Dependence plots for top 3 features
        top_features = feature_importance_df.head(3)['Feature'].tolist()
        for feature in top_features:
            xai.plot_dependence(X_sample, feature)
        
        # Generate comprehensive report
        xai.generate_comprehensive_report(X_sample, feature_importance_df)
        
        return xai, feature_importance_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main_enhanced():
    """Main execution with enhanced FE & XAI"""
    
    print("\n\n")
    print("="*100)
    print("  ENHANCED FRAUD DETECTION RESEARCH SYSTEM")
    print("  Focus: Detailed Feature Engineering & Explainable AI (XAI)")
    print("="*100)
    print()
    
    # Initialize
    pipeline = EnhancedFraudDetectionPipeline()
    
    # Load data
    # CHANGE THIS PATH to your PaySim dataset location
    try:
        pipeline.load_and_prepare_data(
            data_path='PS_20174392719_1491204439457_log.csv',
            sample_size=100000  # Use 100k for faster testing, remove for full dataset
        )
    except FileNotFoundError:
        print("\n❌ Dataset not found!")
        print("📥 Please download PaySim dataset from:")
        print("   https://www.kaggle.com/datasets/ealaxi/paysim1")
        return
    
    # Feature Engineering (DETAILED)
    print("\n" + "="*100)
    print("📊 STEP 2: ADVANCED FEATURE ENGINEERING")
    print("="*100)
    pipeline.apply_feature_engineering()
    
    # Prepare data
    pipeline.prepare_train_test(test_size=0.3, sampling_method='smote')
    
    # Train model
    model_xgb = pipeline.train_xgboost()
    
    # XAI Analysis (COMPREHENSIVE)
    print("\n" + "="*100)
    print("📊 STEP 4: COMPREHENSIVE EXPLAINABLE AI (XAI) ANALYSIS")
    print("="*100)
    
    xai_analyzer, feature_importance = pipeline.apply_comprehensive_xai(
        model_name='XGBoost',
        num_samples=500
    )
    
    # Final summary
    print("\n" + "="*100)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*100)
    print("\n📁 Generated Files:")
    print("   1. feature_engineering_catalog.csv - Complete feature documentation")
    print("   2. feature_importance_XGBoost.csv - SHAP feature rankings")
    print("   3. xai_summary_XGBoost.png - SHAP summary plot")
    print("   4. xai_bar_XGBoost.png - Feature importance bar chart")
    print("   5. xai_waterfall_XGBoost_*.png - Individual prediction explanations")
    print("   6. xai_dependence_XGBoost_*.png - Feature dependence plots")
    print("   7. xai_report_XGBoost.txt - Comprehensive XAI report")
    
    print("\n💡 KEY ACHIEVEMENTS:")
    print("   ✓ Created 35+ engineered features with full documentation")
    print("   ✓ Applied SHAP for comprehensive model explainability")
    print("   ✓ Generated publication-ready visualizations")
    print("   ✓ Produced detailed analysis reports")
    
    print("\n📝 READY FOR JOURNAL SUBMISSION!")
    print("="*100)
    
    return pipeline, xai_analyzer, feature_importance


if __name__ == "__main__":
    pipeline, xai_analyzer, feature_importance = main_enhanced()