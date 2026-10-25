"""Simula señales elementales y analiza su transformada discreta de Fourier.

Ejecutar: python analisis_fourier.py
Las imágenes se guardan en la carpeta resultados/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Permite generar imágenes sin abrir una ventana gráfica.
import matplotlib.pyplot as plt
import numpy as np


# Parámetros compartidos por todas las señales: 512 muestras en 2 segundos.
FS = 256  # Frecuencia de muestreo, en muestras por segundo (Hz).
DURACION = 2.0  # Duración del registro, en segundos.
N = int(FS * DURACION)
TIEMPO = np.arange(N) / FS
FRECUENCIAS = np.fft.fftshift(np.fft.fftfreq(N, d=1 / FS))
SALIDA = Path(__file__).resolve().parent / "resultados"


def transformar(senal):
    """Calcula la FFT centrada y normalizada por la cantidad de muestras."""
    return np.fft.fftshift(np.fft.fft(senal)) / N


def guardar_panel(senal, nombre, archivo):
    """Grafica la señal, la magnitud y la fase de sus componentes relevantes."""
    espectro = transformar(senal)
    magnitud = np.abs(espectro)
    fase = np.angle(espectro)

    # La fase de un componente de magnitud casi nula no se interpreta:
    # el redondeo numérico puede asignarle un ángulo arbitrario.
    fase[magnitud < 0.005 * magnitud.max()] = np.nan
    zona = np.abs(FRECUENCIAS) <= 32

    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
    fig.suptitle(nombre, fontsize=15)
    ejes[0].plot(TIEMPO, senal, color="#186d70", linewidth=2)
    ejes[0].set(title="Dominio del tiempo", xlabel="Tiempo (s)", ylabel="Amplitud", xlim=(0, DURACION))
    ejes[1].plot(FRECUENCIAS[zona], magnitud[zona], color="#b55b36", linewidth=1.8)
    ejes[1].set(title="Magnitud", xlabel="Frecuencia (Hz)", ylabel="|X[k]| / N")
    ejes[2].scatter(FRECUENCIAS[zona], fase[zona], color="#584a8a", s=9)
    ejes[2].set(title="Fase", xlabel="Frecuencia (Hz)", ylabel="Fase (rad)", ylim=(-3.4, 3.4))
    for eje in ejes:
        eje.grid(alpha=0.2)
    fig.savefig(SALIDA / archivo, dpi=160)
    plt.close(fig)


def guardar_comparacion_frecuencia(seno_8, seno_16):
    """Compara una senoide de 8 Hz con otra de frecuencia duplicada."""
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    ejes[0].plot(TIEMPO, seno_8, label="8 Hz", color="#186d70")
    ejes[0].plot(TIEMPO, seno_16, label="16 Hz", color="#b55b36", alpha=0.85)
    ejes[0].set(title="Periodo reducido a la mitad", xlabel="Tiempo (s)", ylabel="Amplitud", xlim=(0, 0.5))

    zona = (FRECUENCIAS >= 0) & (FRECUENCIAS <= 24)
    for senal, etiqueta, color in [(seno_8, "8 Hz", "#186d70"), (seno_16, "16 Hz", "#b55b36")]:
        ejes[1].stem(FRECUENCIAS[zona], np.abs(transformar(senal)[zona]),
                     linefmt=color, markerfmt="o", basefmt=" ", label=etiqueta)
    ejes[1].set(title="Picos espectrales", xlabel="Frecuencia (Hz)", ylabel="|X[k]| / N", xlim=(0, 24))
    for eje in ejes:
        eje.grid(alpha=0.2)
        eje.legend()
    fig.savefig(SALIDA / "escalamiento_frecuencia.png", dpi=160)
    plt.close(fig)


def main():
    SALIDA.mkdir(exist_ok=True)

    # Señales: pulso de 0.25 s centrado en 0.75 s, escalón y seno de 8 Hz.
    pulso = (np.abs(TIEMPO - 0.75) < 0.125).astype(float)
    escalon = (TIEMPO >= 0.75).astype(float)
    seno_8 = np.sin(2 * np.pi * 8 * TIEMPO)
    seno_16 = np.sin(2 * np.pi * 16 * TIEMPO)

    guardar_panel(pulso, "Pulso rectangular", "pulso.png")
    guardar_panel(escalon, "Función escalón", "escalon.png")
    guardar_panel(seno_8, "Senoide de 8 Hz", "senoide.png")
    guardar_comparacion_frecuencia(seno_8, seno_16)

    # Linealidad de la DFT: FFT(a·p + b·s) = a·FFT(p) + b·FFT(s).
    a, b = 0.7, 1.2
    error_linealidad = np.max(np.abs(transformar(a * pulso + b * seno_8)
                                      - a * transformar(pulso) - b * transformar(seno_8)))

    # Un desplazamiento circular entero conserva magnitudes y añade fase.
    # np.roll implementa la periodicidad que asume la DFT; aquí son 32 muestras.
    desplazamiento = 32
    pulso_desplazado = np.roll(pulso, desplazamiento)
    indices = np.fft.fftshift(np.fft.fftfreq(N))
    factor_fase = np.exp(-2j * np.pi * indices * desplazamiento)
    error_desplazamiento = np.max(np.abs(transformar(pulso_desplazado)
                                           - transformar(pulso) * factor_fase))
    error_magnitud = np.max(np.abs(np.abs(transformar(pulso_desplazado))
                                    - np.abs(transformar(pulso))))

    # Duplicar la frecuencia desplaza los picos de ±8 Hz a ±16 Hz.
    positivos = FRECUENCIAS >= 0
    pico_8 = FRECUENCIAS[positivos][np.argmax(np.abs(transformar(seno_8)[positivos]))]
    pico_16 = FRECUENCIAS[positivos][np.argmax(np.abs(transformar(seno_16)[positivos]))]

    informe = (
        f"Muestras: {N}; frecuencia de muestreo: {FS} Hz; resolución: {FS/N:.2f} Hz\n"
        f"Error máximo de linealidad: {error_linealidad:.3e}\n"
        f"Desplazamiento circular: {desplazamiento} muestras = {desplazamiento/FS:.3f} s\n"
        f"Error máximo de la propiedad de desplazamiento: {error_desplazamiento:.3e}\n"
        f"Cambio máximo en magnitud por desplazamiento: {error_magnitud:.3e}\n"
        f"Picos positivos al duplicar la frecuencia: {pico_8:.1f} Hz y {pico_16:.1f} Hz\n"
    )
    (SALIDA / "resumen.txt").write_text(informe, encoding="utf-8")
    print(informe)
    print(f"Gráficas y resumen guardados en: {SALIDA}")


if __name__ == "__main__":
    main()
