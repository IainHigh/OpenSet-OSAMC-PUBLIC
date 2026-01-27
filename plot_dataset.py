#!/usr/bin/python3
# pylint: disable=import-error

"""
plot_dataset.py:
Code for plotting the time domain, frequency domain, spectrogram, and constellation diagrams.
"""

import os
import json
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

np.Inf = np.inf  # Fix for a bug in the numpy library

#####################################################################
####################### MODIFIABLE VARIABLES ########################
#####################################################################

# Directories:
DATASET_PATH = "/exports/eddie/scratch/s2062378/data/data_analysis"

TIME_DOMAIN_OUTPUT_PATH = "./OutputFiles/Time-Domain"
FREQUENCY_DOMAIN_OUTPUT_PATH = "./OutputFiles/Frequency-Domain"
CONSTELLATION_DIAGRAM_OUTPUT_PATH = "./OutputFiles/Constellation-Diagrams"
SPECTROGRAM_OUTPUT_PATH = "./OutputFiles/Spectrograms"

# Time domain plotting parameters - time domain is too large to plot all at once
TIME_DOMAIN_LENGTH = 200

# Spectrogram plotting parameters:
SPECTROGRAM_FFT_SIZE = 1024  # Size of the FFT
SPECTROGRAM_SAMPLE_RATE = 1e6  # Sample rate of the signal

#####################################################################
#################### END OF MODIFIABLE VARIABLES ####################
#####################################################################


def _make_output_dirs():
    # Create output directories if they do not exist
    os.makedirs(TIME_DOMAIN_OUTPUT_PATH, exist_ok=True)
    os.makedirs(FREQUENCY_DOMAIN_OUTPUT_PATH, exist_ok=True)
    os.makedirs(CONSTELLATION_DIAGRAM_OUTPUT_PATH, exist_ok=True)
    os.makedirs(SPECTROGRAM_OUTPUT_PATH, exist_ok=True)


def _delete_existing_plots():
    # Empty out the Time-Domain diagram directory
    for file in os.listdir(TIME_DOMAIN_OUTPUT_PATH):
        os.remove(f"{TIME_DOMAIN_OUTPUT_PATH}/{file}")

    # Empty out the Frequency-Domain diagram directory
    for file in os.listdir(FREQUENCY_DOMAIN_OUTPUT_PATH):
        os.remove(f"{FREQUENCY_DOMAIN_OUTPUT_PATH}/{file}")

    # Empty out the Constellation diagram directory
    for file in os.listdir(CONSTELLATION_DIAGRAM_OUTPUT_PATH):
        os.remove(f"{CONSTELLATION_DIAGRAM_OUTPUT_PATH}/{file}")

    # Empty out the Spectrogram directory
    for file in os.listdir(SPECTROGRAM_OUTPUT_PATH):
        os.remove(f"{SPECTROGRAM_OUTPUT_PATH}/{file}")


def _get_data(file):
    ## get meta
    with open(file + ".sigmf-meta", encoding="utf-8") as _f:
        f_meta = json.load(_f)
    f_meta = f_meta["annotations"][0]

    ## get data
    with open(file + ".sigmf-data", "rb") as _f:
        f_data = np.load(_f)

    rfml_labels = f_meta["rfml_labels"]
    modscheme = rfml_labels.get("modclass", rfml_labels.get("modulations"))
    return f_data, modscheme


def _plot_time_domain_diagram(f_data, modscheme, index):
    i = f_data[0 : TIME_DOMAIN_LENGTH * 2 : 2]
    q = f_data[1 : (TIME_DOMAIN_LENGTH * 2) + 1 : 2]

    # Plot
    plt.figure()
    plt.plot(i, label="I")
    plt.plot(q, label="Q")
    plt.grid()
    plt.title(f"Time Domain Diagram ({modscheme})")
    plt.xlabel("Time [samples]")
    plt.ylabel("Amplitude [V]")
    plt.legend()
    plt.savefig(f"{TIME_DOMAIN_OUTPUT_PATH}/{modscheme}_{index}.png")
    plt.close()


def _plot_frequency_domain_diagram(f_data, modscheme, index):
    i = f_data[0::2]
    q = f_data[1::2]

    # combine the real (I) and imaginary (Q) parts into a complex signal
    x = i + 1j * q

    # Fourier Transform
    x = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2)
    f = np.linspace(-0.5, 0.5, len(x))

    # Plot
    plt.figure()
    plt.plot(f, x)
    plt.grid()
    plt.title(f"Frequency Domain Diagram ({modscheme})")
    plt.xlabel("Frequency [Hz Normalized]")
    plt.ylabel("PSD [dB]")
    plt.savefig(f"{FREQUENCY_DOMAIN_OUTPUT_PATH}/{modscheme}_{index}.png")
    plt.close()


def _plot_constellation_diagram(f_data, modscheme, index):
    symbol_rate = 2

    i = f_data[0::2]
    q = f_data[1::2]

    i = i[::symbol_rate]
    q = q[::symbol_rate]

    # Plot
    plt.figure()
    # Density-based coloring
    # xy = np.vstack([i, q])
    # z = gaussian_kde(xy)(xy)  # Compute density
    # idx = z.argsort()  # Sort points by density for better visualization
    # i, q, z = i[idx], q[idx], z[idx]
    # plt.scatter(i, q, c=z, s=1, cmap="viridis")  # Density-colored scatter plot
    # plt.axhline(0, color="gray", linewidth=0.5, linestyle="--")  # Horizontal axis
    # plt.axvline(0, color="gray", linewidth=0.5, linestyle="--")  # Vertical axis
    # plt.xlabel("I (In-phase)")
    # plt.ylabel("Q (Quadrature)")

    plt.scatter(i, q, s=1)
    plt.axis("off")
    plt.axis("equal")
    plt.tight_layout()

    # plt.grid(True, linestyle="--", alpha=0.5)
    # cbar = plt.colorbar()  # Add colorbar to explain density
    # cbar.set_label("Density")
    plt.savefig(
        f"{CONSTELLATION_DIAGRAM_OUTPUT_PATH}/{modscheme}_{index}.png", transparent=True
    )

    plt.close()


def _plot_spectrogram(f_data, modscheme, index):
    i = f_data[0::2]
    q = f_data[1::2]

    # combine the real (I) and imaginary (Q) parts into a complex signal
    x = i + 1j * q

    num_rows = int(np.floor(len(x) / SPECTROGRAM_FFT_SIZE))
    spectrogram = np.zeros((num_rows, SPECTROGRAM_FFT_SIZE))
    for i in range(num_rows):
        spectrogram[i, :] = 10 * np.log10(
            np.abs(
                np.fft.fftshift(
                    np.fft.fft(
                        x[i * SPECTROGRAM_FFT_SIZE : (i + 1) * SPECTROGRAM_FFT_SIZE]
                    )
                )
            )
            ** 2
        )

    # Plot
    plt.figure()
    plt.imshow(
        spectrogram,
        aspect="auto",
        extent=[
            SPECTROGRAM_SAMPLE_RATE / -2,
            SPECTROGRAM_SAMPLE_RATE / 2,
            0,
            len(x) / SPECTROGRAM_SAMPLE_RATE,
        ],
    )
    plt.title(f"Spectrogram ({modscheme})")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Time [samples]")
    plt.savefig(f"{SPECTROGRAM_OUTPUT_PATH}/{modscheme}_{index}.png")
    plt.close()


def main():
    """
    Main function to execute the plotting of various diagrams.
    """
    _make_output_dirs()
    _delete_existing_plots()

    # Get the list of files
    files = os.listdir(os.path.abspath(DATASET_PATH))
    files = [os.path.join(DATASET_PATH, f.split(".")[0]) for f in files]
    files = list(set(files))

    for i in range(len(files)):
        f_data, modscheme = _get_data(files[i])
        # _plot_time_domain_diagram(f_data, modscheme, i)
        # _plot_frequency_domain_diagram(f_data, modscheme, i)
        _plot_constellation_diagram(f_data, modscheme, i)
        # _plot_spectrogram(f_data, modscheme, i)


if __name__ == "__main__":
    main()
