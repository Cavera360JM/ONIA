import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dados de treinamento e teste
train_data = pd.read_csv("treino.csv")
test_data = pd.read_csv("teste.csv")

# 2. Preparar os dados
features = [f"col_{i}" for i in range(13)]
X_train = train_data[features]
y_train = train_data["target"]
X_test = test_data[features]

# 3. Análise Exploratória de Dados (EDA)
# 3.1. Estatísticas Descritivas
print("Estatísticas Descritivas:")
print(train_data.describe())

# 3.2. Visualizações
# Exemplo: Histograma da coluna 'TempMédia'
plt.figure(figsize=(8, 6))
sns.histplot(train_data["col_0"], bins=20)
plt.title("Histograma da Temperatura Média")
plt.show()

# Exemplo: Boxplot da relação entre 'Gravidade' e 'target'
plt.figure(figsize=(8, 6))
sns.boxplot(x="target", y="col_1", data=train_data)
plt.title("Boxplot da Gravidade por Categoria de Planeta")
plt.show()

# 4. Ajuste de Hiperparâmetros com GridSearchCV
param_grid = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring="f1_macro")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Melhores hiperparâmetros: {grid_search.best_params_}")

# 5. Avaliação do Modelo com Validação Cruzada (usando F1-score)
scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="f1_macro")
print(f"Acurácia média (F1-score): {scores.mean()}")

# 6. Fazer previsões no conjunto de teste
predictions = best_model.predict(X_test)

# 7. Criar o arquivo de envio
submission = pd.DataFrame({"id": test_data["id"], "target": predictions})
submission.to_csv("submission.csv", index=False)

print("Arquivo de envio 'submission.csv' criado com sucesso!")

# 8. Avaliação do modelo no conjunto de teste para o relatório

f1 = f1_score(y_train, best_model.predict(X_train), average='macro')

print(f"F1-score no conjunto de treino: {f1}")

# 9. Análise de Erros (opcional)
# Identificar instâncias classificadas incorretamente
train_predictions = best_model.predict(X_train)
errors = train_data[train_predictions != y_train]
print("Instâncias classificadas incorretamente:")
print(errors)