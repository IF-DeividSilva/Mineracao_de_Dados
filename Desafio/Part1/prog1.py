import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score

train_data = pd.read_csv('X_trainToronto.csv')
test_data = pd.read_csv('X_testToronto.csv')

y_train = train_data['destaque']
X_train = train_data.drop(columns=['business_id', 'destaque'])
X_test = test_data.drop(columns=['business_id'])

X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)
X_train_encoded.columns = X_train_encoded.columns.str.replace(r'[\[\]<>]', '_', regex=True)
X_test_encoded.columns = X_test_encoded.columns.str.replace(r'[\[\]<>]', '_', regex=True)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# calcular peso da classe minoritária
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=6
)

model.fit(X_train_encoded, y_train)

# Ajuste de threshold
probs = model.predict_proba(X_train_encoded)[:, 1]
best_thresh = 0.5
best_f1 = 0

for t in np.arange(0.1, 0.9, 0.05):
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_train, preds)
    print(f"Threshold {t:.2f} - F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

# Previsão final com melhor threshold
test_probs = model.predict_proba(X_test_encoded)[:, 1]
test_preds = (test_probs >= best_thresh).astype(int)

submission = pd.DataFrame({
    'business_id': test_data['business_id'],
    'destaque': test_preds
})
submission.to_csv('submission_xgboost_tuned.csv', index=False)

print(f" Melhor threshold: {best_thresh:.2f} com F1: {best_f1:.4f}")
print(f" Positivos previstos no teste: {sum(test_preds)} de {len(test_preds)}")
