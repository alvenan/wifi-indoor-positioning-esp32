import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================================
# CONFIGURAÇÕES
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

ARQUIVO_TREINO = os.path.join(DATA_DIR, "wifi_train_dataset.csv")
ARQUIVO_TESTE = os.path.join(DATA_DIR, "wifi_test_dataset.csv")

K = 3
N_AMOSTRAS_TESTE_POR_LOCAL = 30
RANDOM_STATE = 42

# =====================================================
# CARREGAR DATASETS
# =====================================================

train_df = pd.read_csv(ARQUIVO_TREINO)
test_df = pd.read_csv(ARQUIVO_TESTE)

# Colunas de RSSI usadas como entrada do modelo.
feature_cols = [col for col in train_df.columns if col.startswith("ap")]

# A coluna local é a classe correta esperada.
# Exemplo: RP1, RP2, RP3...
LABEL_COL = "local"

# Usa 30 amostras de CADA região/local da base de teste.
# A ordem do CSV não importa, pois o groupby separa por local_real.
test_df = (
    test_df
    .groupby(LABEL_COL, group_keys=False)
    .sample(n=N_AMOSTRAS_TESTE_POR_LOCAL, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

X_train = train_df[feature_cols].values
y_train = train_df[LABEL_COL].values

X_test = test_df[feature_cols].values
y_test = test_df[LABEL_COL].values

# =====================================================
# FUNÇÃO W-KNN
# =====================================================

def wknn_predict(x, X_train, y_train, k=3):
    # Calcula a distância Euclidiana entre a amostra de teste
    # e todas as amostras da base de treinamento.
    distances = np.linalg.norm(X_train - x, axis=1)

    # Seleciona os k vizinhos mais próximos.
    nearest_idx = np.argsort(distances)[:k]

    nearest_labels = y_train[nearest_idx]
    nearest_distances = distances[nearest_idx]

    # Calcula os pesos pelo inverso da distância.
    # O termo 1e-6 evita divisão por zero.
    weights = 1 / (nearest_distances + 1e-6)

    # Soma os pesos por classe/local.
    scores = {}
    for label, weight in zip(nearest_labels, weights):
        scores[label] = scores.get(label, 0) + weight

    # Retorna a classe/local com maior peso acumulado.
    return max(scores, key=scores.get)

# =====================================================
# PREDIÇÃO
# =====================================================

y_pred = []

for x in X_test:
    pred = wknn_predict(x, X_train, y_train, k=K)
    y_pred.append(pred)

# =====================================================
# AVALIAÇÃO
# =====================================================

acc = accuracy_score(y_test, y_pred)

print("Resumo da avaliação")
print("-------------------")
print(f"K usado no W-KNN: {K}")
print(f"Amostras de treino: {len(train_df)}")
print(f"Amostras de teste usadas: {len(test_df)}")
print(f"Amostras por local no teste: {N_AMOSTRAS_TESTE_POR_LOCAL}")
print(f"Acurácia: {acc:.4f}")
print(f"Acurácia percentual: {acc * 100:.2f}%")

print("\nQuantidade de amostras por local no teste:")
print(test_df[LABEL_COL].value_counts().sort_index())

labels = sorted(train_df[LABEL_COL].unique(), key=lambda x: int(str(x).replace("RP", "")) if str(x).replace("RP", "").isdigit() else str(x))

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred, labels=labels))

print("\nOrdem das classes na matriz de confusão:")
print(labels)

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, labels=labels))

# Opcional: salva as predições para análise posterior.
resultados = test_df.copy()
resultados["local_predito"] = y_pred
resultados["acerto"] = resultados[LABEL_COL] == resultados["local_predito"]
caminho_saida = os.path.join(DATA_DIR, "resultado_wknn_2.csv")
resultados.to_csv(caminho_saida, index=False)
print(f"\nArquivo {caminho_saida} gerado com as predições.")
