# ============================================
# HOME CREDIT DEFAULT RISK - PYTHON WORKFLOW
# ============================================

# 1. IMPORT LIBRARY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------
application_train = pd.read_csv('application_train.csv')
application_test  = pd.read_csv('application_test.csv')
bureau = pd.read_csv('bureau.csv')
previous_app = pd.read_csv('previous_application.csv')
installments = pd.read_csv('installments_payments.csv')

# --------------------------------------------
# 3. BASIC EDA
# --------------------------------------------
print(application_train.shape)
print(application_train['TARGET'].value_counts())
print(application_train.isnull().sum().sort_values(ascending=False).head())

# --------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------
# ===== Bureau Aggregation =====
bureau_agg = bureau.groupby('SK_ID_CURR').agg({
    'AMT_CREDIT_SUM': ['mean', 'max'],
    'DAYS_CREDIT': ['mean'],
    'CREDIT_DAY_OVERDUE': ['mean'],
})

bureau_agg.columns = ['BUREAU_CREDIT_MEAN', 'BUREAU_CREDIT_MAX',
                      'BUREAU_DAYS_CREDIT_MEAN', 'BUREAU_OVERDUE_MEAN']

# ===== Previous Application Aggregation =====
prev_agg = previous_app.groupby('SK_ID_CURR').agg({
    'AMT_APPLICATION': ['mean'],
    'AMT_CREDIT': ['mean'],
    'DAYS_DECISION': ['mean']
})

prev_agg.columns = ['PREV_APP_MEAN', 'PREV_CREDIT_MEAN', 'PREV_DECISION_MEAN']

# ===== Installments Aggregation =====
installments['PAYMENT_DIFF'] = installments['AMT_PAYMENT'] - installments['AMT_INSTALMENT']

inst_agg = installments.groupby('SK_ID_CURR').agg({
    'PAYMENT_DIFF': ['mean'],
    'DAYS_ENTRY_PAYMENT': ['mean']
})

inst_agg.columns = ['PAYMENT_DIFF_MEAN', 'PAYMENT_DAYS_MEAN']

# --------------------------------------------
# 5. MERGE DATA
# --------------------------------------------
data = application_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
data = data.merge(prev_agg, on='SK_ID_CURR', how='left')
data = data.merge(inst_agg, on='SK_ID_CURR', how='left')

# --------------------------------------------
# 6. HANDLE MISSING VALUES
# --------------------------------------------
for col in data.columns:
    if data[col].dtype != 'object':
        data[col].fillna(data[col].median(), inplace=True)
    else:
        data[col].fillna('Unknown', inplace=True)

# --------------------------------------------
# 7. ENCODING
# --------------------------------------------
le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# --------------------------------------------
# 8. SPLIT DATA
# --------------------------------------------
X = data.drop(['TARGET', 'SK_ID_CURR'], axis=1)
y = data['TARGET']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------
# 9. SCALING
# --------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------
# 10. MODELING
# --------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --------------------------------------------
# 11. EVALUATION
# --------------------------------------------
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)

print('AUC Score:', auc)
print(classification_report(y_test, model.predict(X_test)))

# --------------------------------------------
# 12. FEATURE IMPORTANCE (OPTIONAL)
# --------------------------------------------
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print(feature_importance.head(10))

# --------------------------------------------
# 13. BUSINESS USAGE
# --------------------------------------------
# Output probability default digunakan sebagai credit risk score
# Semakin tinggi probability → semakin berisiko
