# CNN para reconocimiento de digitos

Red neuronal convolucional entrenada con PyTorch para reconocer digitos escritos a mano (0-9), con una interfaz interactiva en Streamlit donde se puede dibujar un numero y obtener la prediccion en tiempo real.

**Precision en test: 99.14%**

## Demo

[digit-recognizer-app-tmo.streamlit.app](https://digit-recognizer-app-tmo.streamlit.app/)

## Como funciona

1. El usuario dibuja un digito en un canvas de 280x280 px.
2. La imagen se redimensiona a 28x28 px (formato MNIST) y se convierte a escala de grises.
3. La CNN procesa la imagen y devuelve la probabilidad para cada digito.
4. Si la confianza supera el 85%, muestra la prediccion con un grafico de barras; si no, pide reintentar.

## Arquitectura de la CNN

| Capa | Detalle |
|------|---------|
| Conv1 | 1 &rarr; 3 canales, kernel 5x5 |
| Conv2 | 3 &rarr; 5 canales, kernel 5x5 |
| Conv3 | 5 &rarr; 10 canales, kernel 5x5 |
| MaxPool | 2x2, stride 2 |
| Linear1 | 640 &rarr; 256 + Dropout(0.2) |
| Linear2 | 256 &rarr; 128 + Dropout(0.2) |
| Linear3 | 128 &rarr; 10 (softmax) |

## Stack

- **PyTorch** - entrenamiento y modelo
- **Streamlit** - interfaz web interactiva
- **NumPy / Pillow** - preprocesamiento de imagen

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run UI.py
```
