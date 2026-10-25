# Simulación y análisis de señales con la transformada de Fourier

Programa en Python para representar un pulso rectangular, una función escalón y una senoide en el dominio del tiempo y de la frecuencia. Calcula magnitud y fase mediante `np.fft.fft()`, comprueba linealidad y desplazamiento temporal, y muestra el efecto de duplicar la frecuencia de la senoide.

## Requisitos

- Python 3.10 o posterior.
- NumPy y Matplotlib, indicados en `requirements.txt`.

## Ejecución

Desde la carpeta del proyecto:

```bash
python -m venv .venv
```

Activa el entorno virtual con `.venv\Scripts\activate` en Windows o `source .venv/bin/activate` en macOS y Linux. Después ejecuta:

```bash
python -m pip install -r requirements.txt
python analisis_fourier.py
```

El programa crea la carpeta `resultados/` con cuatro imágenes PNG y `resumen.txt`. No necesita conexión a Internet después de instalar las dependencias.

## Archivos generados

| Archivo | Qué muestra |
| --- | --- |
| `pulso.png` | Pulso rectangular en tiempo, magnitud y fase. |
| `escalon.png` | Escalón muestreado y su espectro. |
| `senoide.png` | Senoide de 8 Hz con picos en −8 y +8 Hz. |
| `escalamiento_frecuencia.png` | Comparación de 8 y 16 Hz. |
| `resumen.txt` | Errores numéricos y frecuencias de los picos. |

## Interpretación

La FFT utiliza 512 muestras tomadas a 256 Hz, por lo que la separación entre bins es 0.5 Hz. La magnitud se divide entre 512; así, cada uno de los dos picos de una senoide real de amplitud unitaria mide aproximadamente 0.5. La fase se oculta donde la magnitud es casi cero, pues allí su valor carece de interpretación estable.

La propiedad de desplazamiento se comprueba con `np.roll`, que hace un desplazamiento **circular** de 32 muestras (0.125 s). Esta precisión importa: la DFT supone que la secuencia se repite periódicamente. El escalón de la simulación es un registro finito; su FFT no es la transformada continua de un escalón ideal infinito.