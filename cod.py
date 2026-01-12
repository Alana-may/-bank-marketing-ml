import sklearn.utils.validation

# Este bloco resolve o erro de compatibilidade entre imblearn e sklearn
if not hasattr(sklearn.utils.validation, '_is_pandas_df'):
    def _is_pandas_df(X):
        return hasattr(X, "columns") and hasattr(X, "iloc")
    sklearn.utils.validation._is_pandas_df = _is_pandas_df

# Agora sim, seguem os seus outros imports normais...
import pandas as pd
import seaborn as sns
# ... o restante do seu código

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.pipeline import Pipeline 
from imblearn.under_sampling import NearMiss
from sklearn.linear_model import LogisticRegression

print("\n--- 1. Carregamento e Tratamento Inicial ---")


try:
    df = pd.read_csv(...)
except:
    df = pd.read_csv('bank-full.csv', sep=',') 


if df.shape[1] < 2:
     df = pd.read_csv('bank-full.csv', sep=';')

print(f"Dados carregados: {df.shape[0]} linhas e {df.shape[1]} colunas.")
cols_categoricas = df.select_dtypes(include=['object']).columns
for col in cols_categoricas:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

print(f"Base de Treino: {X_train.shape}")
print(f"Base de Teste: {X_test.shape}")

steps = [
    ('scaler', StandardScaler()),       
    ('nearmiss', NearMiss(version=1)),
    ('model', LogisticRegression(max_iter=1000)) 
]
pipeline = Pipeline(steps=steps)

print("\n--- 2. Rodando Otimização e Validação Cruzada (GridSearch) ---")
print("Aguarde... Testando combinações de vizinhos e parâmetros...")

parameters = {
    'nearmiss__n_neighbors': [1, 3],
    'model__C': [0.1, 1, 10]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(pipeline, param_grid=parameters, cv=cv, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"\n✅ Melhores Parâmetros encontrados: {grid.best_params_}")
print(f"✅ Melhor Score (F1) na validação cruzada: {grid.best_score_:.4f}")


print("\n--- 3. Avaliação Final no Conjunto de Teste ---")
y_pred = grid.predict(X_test)

nomes_classes = ['NÃO (Recusou)', 'SIM (Aceitou)']
print(classification_report(y_test, y_pred, target_names=nomes_classes))
plt.figure(figsize=(8, 6))

sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=nomes_classes,
            yticklabels=nomes_classes)

plt.title('Matriz de Acertos e Erros (NearMiss)')
plt.ylabel('REALIDADE (O que o cliente fez)')
plt.xlabel('PREVISÃO (O que a IA disse)')

plt.show()
