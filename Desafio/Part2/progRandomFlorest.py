import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score

# 1. Carregar os datasets de treino e teste
train_data = pd.read_csv('X_trainToronto.csv')
test_data = pd.read_csv('X_testToronto.csv')

print(train_data['destaque'].value_counts())


# 2. Extrair o rótulo e remover colunas desnecessárias
y_train = train_data['destaque']
X_train = train_data.drop(columns=['business_id', 'destaque'])
X_test = test_data.drop(columns=['business_id'])

# 3. Converter variáveis categóricas para dummy
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# como dados de train estão desbalenceados para 0, usando o class_weight para tentar melhorar o desempenho...
## n funcionou f
model = RandomForestClassifier(n_estimators=50,class_weight='balanced', random_state=6 )

# 7. Treinar o modelo usando todos os dados de treino
model.fit(X_train_encoded, y_train)

# Opcional: Verificar o desempenho no conjunto de treino
train_preds = model.predict(X_train_encoded)
print("F1 Score no treinamento:", f1_score(y_train, train_preds))

# 8. Realizar a previsão para o conjunto de teste
test_preds = model.predict(X_test_encoded)

# 9. Gerar o arquivo de submissão no formato requerido (business_id e destaque)
submission = pd.DataFrame({
    'business_id': test_data['business_id'],
    'destaque': test_preds
})
submission.to_csv('submissionRandomFlorest.csv', index=False)
