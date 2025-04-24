##
##  SCORE F1 = 0,05504
##
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# 1. Carregar os datasets de treino e teste.
train_data = pd.read_csv('X_trainToronto.csv')
test_data = pd.read_csv('X_testToronto.csv')

# 2. Extrair o rótulo e remover colunas desnecessárias.
#    "destaque" é o rótulo e "business_id" será usado apenas para a submissão.
y_train = train_data['destaque']
X_train = train_data.drop(columns=['business_id', 'destaque'])
X_test = test_data.drop(columns=['business_id'])

# 3. Converter as colunas categóricas em variáveis numéricas (One-Hot Encoding).
#    drop_first=True é utilizado para evitar a multicolinearidade.
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# 4. Alinhar as colunas entre os conjuntos de treino e teste.
#    Caso alguma coluna esteja presente em apenas um dos conjuntos, ela é criada no outro com valor 0.
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='outer', axis=1, fill_value=0)

# 5. Criar e treinar o modelo de Árvore de Decisão.
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_encoded, y_train)

# 6. Avaliar o modelo no conjunto de treino utilizando a métrica F1 Score.
train_preds = model.predict(X_train_encoded)
print("F1 Score no treinamento:", f1_score(y_train, train_preds))

# 7. Realizar a previsão para o conjunto de teste.
test_preds = model.predict(X_test_encoded)

# 8. Gerar o arquivo de submissão conforme o formato requerido.
submission = pd.DataFrame({
    'business_id': test_data['business_id'],
    'destaque': test_preds
})
submission.to_csv('submission.csv', index=False)
