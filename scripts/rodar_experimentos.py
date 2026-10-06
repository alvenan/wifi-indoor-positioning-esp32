import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import json

# =====================================================
# CONFIGURAÇÕES E CONSTANTES
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

ARQUIVO_TREINO = os.path.join(DATA_DIR, "wifi_train_dataset.csv")
ARQUIVO_TESTE = os.path.join(DATA_DIR, "wifi_test_dataset.csv")
RANDOM_STATE = 42

# Carregar datasets
train_df = pd.read_csv(ARQUIVO_TREINO)
test_df = pd.read_csv(ARQUIVO_TESTE)

# Colunas de RSSI (começam com "ap")
feature_cols = [col for col in train_df.columns if col.startswith("ap")]
LABEL_COL = "local"

X_train = train_df[feature_cols].values
y_train = train_df[LABEL_COL].values

# =====================================================
# IMPLEMENTAÇÃO DO CUSTOM W-KNN (CONFORME CÓDIGO DA AULA)
# =====================================================
def wknn_predict_single(x, X_train, y_train, k=3):
    # Distância Euclidiana
    distances = np.linalg.norm(X_train - x, axis=1)
    nearest_idx = np.argsort(distances)[:k]
    
    nearest_labels = y_train[nearest_idx]
    nearest_distances = distances[nearest_idx]
    
    # Evita divisão por zero
    weights = 1 / (nearest_distances + 1e-6)
    
    scores = {}
    for label, weight in zip(nearest_labels, weights):
        scores[label] = scores.get(label, 0) + weight
        
    return max(scores, key=scores.get)

def wknn_predict(X_test, X_train, y_train, k=3):
    return np.array([wknn_predict_single(x, X_train, y_train, k=k) for x in X_test])

# Wrapper de compatibilidade para o Custom W-KNN
class CustomWKNN:
    def __init__(self, k=3):
        self.k = k
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        return self
    def predict(self, X):
        return wknn_predict(X, self.X_train, self.y_train, k=self.k)

# =====================================================
# ETAPA 1: CROSS-VALIDATION NO TREINO PARA AJUSTE DE PARÂMETROS
# =====================================================
print("Iniciando a sintonia de hiperparâmetros (Validação Cruzada 5-Fold no Treino)...")
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Param sweeps
knn_ks = [1, 3, 5, 7]
wknn_ks = [1, 3, 5, 7]
svm_params = [
    {"kernel": "linear", "C": 1},
    {"kernel": "linear", "C": 10},
    {"kernel": "rbf", "C": 1, "gamma": "scale"},
    {"kernel": "rbf", "C": 10, "gamma": "scale"}
]

# Avaliação do KNN
best_knn_k = None
best_knn_score = -1
for k in knn_ks:
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        clf = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
    mean_score = np.mean(scores)
    print(f"  KNN (K={k}) -> Acurácia Média CV: {mean_score:.4f}")
    if mean_score > best_knn_score:
        best_knn_score = mean_score
        best_knn_k = k

# Avaliação do WKNN
best_wknn_k = None
best_wknn_score = -1
for k in wknn_ks:
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        clf = CustomWKNN(k=k)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
    mean_score = np.mean(scores)
    print(f"  W-KNN (K={k}) -> Acurácia Média CV: {mean_score:.4f}")
    if mean_score > best_wknn_score:
        best_wknn_score = mean_score
        best_wknn_k = k

# Avaliação do SVM
best_svm_config = None
best_svm_score = -1
for param in svm_params:
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        clf = SVC(kernel=param["kernel"], C=param["C"], gamma=param.get("gamma", "scale"), random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
    mean_score = np.mean(scores)
    print(f"  SVM (kernel={param['kernel']}, C={param['C']}) -> Acurácia Média CV: {mean_score:.4f}")
    if mean_score > best_svm_score:
        best_svm_score = mean_score
        best_svm_config = param

print("\n--- Melhores Hiperparâmetros Encontrados ---")
print(f"  Melhor KNN: K={best_knn_k} (Acurácia CV: {best_knn_score:.4f})")
print(f"  Melhor W-KNN: K={best_wknn_k} (Acurácia CV: {best_wknn_score:.4f})")
print(f"  Melhor SVM: kernel={best_svm_config['kernel']}, C={best_svm_config['C']} (Acurácia CV: {best_svm_score:.4f})")
print("-------------------------------------------\n")

# =====================================================
# ETAPA 2: AVALIAÇÃO NOS TRÊS CENÁRIOS DE TESTE (10, 30, 50 AMOSTRAS)
# =====================================================

resultados_experimentos = {}

# Definir classificadores finais treinados na base COMPLETA
clf_knn = KNeighborsClassifier(n_neighbors=best_knn_k, weights='uniform', metric='euclidean')
clf_knn.fit(X_train, y_train)

clf_wknn = CustomWKNN(k=best_wknn_k)
clf_wknn.fit(X_train, y_train)

clf_svm = SVC(kernel=best_svm_config["kernel"], C=best_svm_config["C"], gamma=best_svm_config.get("gamma", "scale"), random_state=RANDOM_STATE)
clf_svm.fit(X_train, y_train)

classificadores = {
    "KNN": clf_knn,
    "WKNN": clf_wknn,
    "SVM": clf_svm
}

tamanhos_teste = [10, 30, 50]

for n_amostras in tamanhos_teste:
    print(f"Avaliando modelos com {n_amostras} amostras por TP...")
    
    # Amostrar n_amostras de cada TP
    if n_amostras < 50:
        sampled_test = (
            test_df
            .groupby(LABEL_COL, group_keys=False)
            .sample(n=n_amostras, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )
    else:
        # Usa todas as 50 amostras por local
        sampled_test = test_df.copy()
        
    X_te = sampled_test[feature_cols].values
    y_te = sampled_test[LABEL_COL].values
    ponto_ids_te = sampled_test["ponto"].values
    
    # Identificar TPs coincidentes vs não coincidentes
    # Coincidentes: TP1 até TP4 (correspondem a RP1 até RP4)
    # Não Coincidentes: TP5 até TP8 (correspondem a RP5 até RP8)
    idx_coincident = np.array([int(str(p).replace("TP", "")) <= 4 for p in ponto_ids_te])
    idx_non_coincident = ~idx_coincident
    
    resultados_experimentos[n_amostras] = {}
    
    for nome_alg, clf in classificadores.items():
        preds = clf.predict(X_te)
        
        # Métricas Globais
        acc = accuracy_score(y_te, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_te, preds, average='macro', zero_division=0)
        
        # Métricas Separadas: Coincidentes vs Não Coincidentes
        preds_coinc = preds[idx_coincident]
        y_te_coinc = y_te[idx_coincident]
        acc_coinc = accuracy_score(y_te_coinc, preds_coinc) if len(y_te_coinc) > 0 else 0.0
        
        preds_non_coinc = preds[idx_non_coincident]
        y_te_non_coinc = y_te[idx_non_coincident]
        acc_non_coinc = accuracy_score(y_te_non_coinc, preds_non_coinc) if len(y_te_non_coinc) > 0 else 0.0
        
        # Matriz de Confusão
        labels_ordenados = sorted(train_df[LABEL_COL].unique(), key=lambda x: int(str(x).replace("RP", "")) if str(x).replace("RP", "").isdigit() else str(x))
        cm = confusion_matrix(y_te, preds, labels=labels_ordenados)
        
        resultados_experimentos[n_amostras][nome_alg] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy_coincidente": float(acc_coinc),
            "accuracy_nao_coincidente": float(acc_non_coinc),
            "confusion_matrix": cm.tolist(),
            "labels_order": labels_ordenados
        }

# =====================================================
# ETAPA 3: GERAR RELATÓRIO FORMATADO EM MARKDOWN E ARQUIVO JSON
# =====================================================

# Gerar tabela geral de acurácia
md_report = []
md_report.append("# Relatório Técnico Final de Experimentos - Localização WiFi RSSI\n")
md_report.append("Este relatório resume o desempenho dos algoritmos KNN, W-KNN e SVM em diferentes tamanhos de amostra por Test Point (TP), distinguindo o comportamento em TPs coincidentes com RPs (TP1-TP4) e TPs não coincidentes com RPs (TP5-TP8).\n")

md_report.append("## 1. Configuração dos Modelos Encontrada via CV (5-Fold)")
md_report.append(f"- **K-Nearest Neighbors (KNN)**: $K = {best_knn_k}$ (Métrica: Euclidiana, Pesos: Uniforme)")
md_report.append(f"- **Weighted K-Nearest Neighbors (W-KNN)**: $K = {best_wknn_k}$ (Métrica: Euclidiana, Pesos: Inverso da Distância $w = 1 / (d + 1e-6)$)")
md_report.append(f"- **Support Vector Machine (SVM)**: Kernel = `{best_svm_config['kernel']}`, $C = {best_svm_config['C']}$, $\\gamma = $ `scale` (Multiclasse: OVR/OVO padrão)\n")

md_report.append("## 2. Visão Geral dos Resultados de Acurácia\n")
md_report.append("| Algoritmo | Amostras por TP | Acurácia Geral | Acurácia Coincidente (TP1-4) | Acurácia Não-Coincidente (TP5-8) | F1-Score (Macro) |")
md_report.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

for n_amostras in tamanhos_teste:
    for nome_alg in ["KNN", "WKNN", "SVM"]:
        res = resultados_experimentos[n_amostras][nome_alg]
        md_report.append(f"| {nome_alg} | {n_amostras} | {res['accuracy']*100:.2f}% | {res['accuracy_coincidente']*100:.2f}% | {res['accuracy_nao_coincidente']*100:.2f}% | {res['f1_score']:.4f} |")

md_report.append("\n## 3. Detalhamento por Tamanho de Amostra\n")

for n_amostras in tamanhos_teste:
    md_report.append(f"### 3.1. Cenário: {n_amostras} Amostras de Teste por TP")
    for nome_alg in ["KNN", "WKNN", "SVM"]:
        res = resultados_experimentos[n_amostras][nome_alg]
        md_report.append(f"\n#### Algoritmo: {nome_alg}")
        md_report.append(f"- **Acurácia Global**: {res['accuracy']*100:.2f}%")
        md_report.append(f"- **Precisão (Macro)**: {res['precision']:.4f}")
        md_report.append(f"- **Recall (Macro)**: {res['recall']:.4f}")
        md_report.append(f"- **F1-Score (Macro)**: {res['f1_score']:.4f}")
        md_report.append(f"- **Acurácia em Pontos Coincidentes (TP1-TP4)**: {res['accuracy_coincidente']*100:.2f}%")
        md_report.append(f"- **Acurácia em Pontos Não-Coincidentes (TP5-TP8)**: {res['accuracy_nao_coincidente']*100:.2f}%")
        
        # Mostrar Matriz de Confusão resumida
        cm = np.array(res["confusion_matrix"])
        md_report.append("\n**Matriz de Confusão**:")
        md_report.append("```")
        md_report.append(str(cm))
        md_report.append("```")
        md_report.append(f"Ordem das classes na matriz: {res['labels_order']}\n")

# Salvar relatório MD
caminho_relatorio = os.path.join(DATA_DIR, "relatorio_experimentos.md")
with open(caminho_relatorio, "w", encoding="utf-8") as f:
    f.write("\n".join(md_report))

# Salvar dados JSON para análises e relatórios programáticos
caminho_json = os.path.join(DATA_DIR, "resultados_detalhados.json")
with open(caminho_json, "w", encoding="utf-8") as f:
    json.dump(resultados_experimentos, f, indent=4)

print("\nResultados processados e salvos com sucesso!")
print(f"- Relatório em Markdown salvo em: {caminho_relatorio}")
print(f"- Dados estruturados salvos em: {caminho_json}")
