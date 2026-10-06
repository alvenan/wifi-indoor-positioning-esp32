import os
import serial
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
ARQUIVO_SAIDA = os.path.join(DATA_DIR, "rssi_dados.txt")

# Definir a porta serial correta (ajuste se necessário)
porta_serial = "/dev/ttyUSB0"  # Porta serial do ESP32 no Linux
baudrate = 115200       # Deve ser o mesmo usado no Arduino

try:
    # Abrir conexão com a porta serial
    ser = serial.Serial(porta_serial, baudrate, timeout=1)
    time.sleep(2)  # Aguarda estabilização da conexão

    print(f"Conectado à {porta_serial}. Capturando dados RSSI em {ARQUIVO_SAIDA}...")

    # Abrir arquivo para salvar os dados
    with open(ARQUIVO_SAIDA, "w", encoding='utf-8') as arquivo:
        while True:
            # Lê a linha enviada pelo Arduino e remove espaços extras
            linha = ser.readline().decode("utf-8", errors="ignore").strip()

            if linha:
                print(linha)  # Exibe no terminal
                arquivo.write(linha + "\n")  # Salva no arquivo

except serial.SerialException as e:
    print(f"Erro ao acessar {porta_serial}: {e}")
except Exception as e:
    print(f"Erro inesperado: {e}")
