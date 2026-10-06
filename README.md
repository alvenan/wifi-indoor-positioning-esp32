# Posicionamento Indoor via WiFi RSSI com ESP32 e Machine Learning

Este projeto implementa um sistema completo de **localização interna cômodo a cômodo** baseado na técnica de **WiFi Fingerprinting**. O sistema utiliza um microcontrolador **ESP32** para coletar a intensidade de sinal de rádio (**RSSI**) dos pontos de acesso (APs) e modelos de **Machine Learning** em Python para classificar a localização do dispositivo.

> **Contexto:** Trabalho Final de Engenharia da disciplina de Redes de Sensores Sem Fio (RSSF) — PPGEE / UFAM (Prof. Dr. Celso Carvalho).  
> **Autores:** Álison Venâncio, Matheus Uchoa, Ayrton Lemes e Victor Cavalcante.

---

## Organização de Pastas

```text
wifi-indoor-positioning-esp32/
├── firmware/                          # Firmware para o microcontrolador ESP32 (PlatformIO / Arduino)
│   ├── platformio.ini                 # Configuração de compilação da placa (esp32dev)
│   └── scan.ino                      # Código C++ gravado no microcontrolador ESP32
├── data/                              # Datasets oficiais do experimento (CSV)
│   ├── wifi_train_dataset.csv         # Dataset de calibração (8 RPs x 50 = 400 amostras)
│   └── wifi_test_dataset.csv          # Dataset de validação (8 TPs x 50 = 400 amostras)
├── scripts/                           # Scripts em Python para coleta, pré-processamento e ML
│   ├── salvar_wifi_serial.py          # Script Python que grava o log serial bruto
│   ├── salvar_csv_formatado-v2.py     # Pré-processador com imputação de sinais e rotulação
│   ├── w-knn-2-atualizado.py          # Classificador W-KNN (distância ponderada)
│   └── rodar_experimentos.py          # Benchmark completo e validação cruzada 5-fold (KNN/W-KNN/SVM)
├── requirements.txt                   # Dependências do Python
├── .gitignore                         # Filtros para Git e PlatformIO
├── LICENSE                            # Licença MIT
└── README.md                          # Guia rápido e documentação do projeto
```

---

## Descrição dos Arquivos

* **`firmware/scan.ino`**: Firmware C++ para o ESP32. Executa varreduras síncronas de redes WiFi a cada 3 segundos e transmite os valores de RSSI pela porta serial (`115200 bps`) no formato `SCAN_ID,SSID,BSSID,CHANNEL,RSSI`.
* **`scripts/salvar_wifi_serial.py`**: Conecta à porta serial USB do computador (`/dev/ttyUSB0`) e grava o fluxo contínuo de leituras no arquivo de log bruto `data/rssi_dados.txt`.
* **`scripts/salvar_csv_formatado-v2.py`**: Processador de dados de rádio. Filtra os 8 BSSIDs fixos monitorados, aplica a imputação padrão de -100 dBm para sinais ausentes e gera o dataset em `data/`.
* **`scripts/w-knn-2-atualizado.py`**: Algoritmo de classificação **W-KNN (Weighted K-Nearest Neighbors)**. Avalia as 400 amostras de teste contra as 400 de calibração nos 8 pontos físicos, pondera os votos pelo inverso da distância ($w_i = \frac{1}{d_i + \epsilon}$) e exibe acurácia e matriz de confusão.
* **`scripts/rodar_experimentos.py`**: Suíte completa de validação científica. Realiza sintonia de hiperparâmetros por **validação cruzada 5-fold** no treino e executa a avaliação comparativa completa dos modelos **KNN**, **W-KNN** e **SVM** em 10, 30 e 50 amostras por TP, gerando os resultados oficiais do artigo.

---

## Como Executar

### 1. Firmware no ESP32
O código do firmware está localizado no diretório `firmware/` no formato PlatformIO e Arduino (`scan.ino`). Abra a pasta em um ambiente compatível com PlatformIO (como no VSCode) ou carregue o arquivo `scan.ino` via Arduino IDE para gravar o código na placa conectada via USB.

### 2. Ambiente e Classificação (Python)
```bash
# 1. Instalar dependências a partir da raiz
pip install -r requirements.txt

# 2. Executar o classificador W-KNN
python scripts/w-knn-2-atualizado.py

# 3. Executar o benchmark comparativo completo (KNN vs. W-KNN vs. SVM)
python scripts/rodar_experimentos.py
```

---

## Licença
Distribuído sob a licença [MIT](LICENSE).
