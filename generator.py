#!/usr/bin/python3
# pylint: disable=import-error

"""
generator.py:
This script generates synthetic signals based on the provided configuration.
It supports various modulation schemes, channel types, and parameters.
The generated signals are saved in SigMF format for further analysis.
"""

## standard imports
import argparse
import json
import os
import ctypes

# third party imports
from tqdm import tqdm
import numpy as np

## internal imports
from utils.sigmf_utils import save_sigmf, archive_sigmf
from utils.config_utils import map_config


BUF = 64
HALF_BUF = BUF // 2
## load c modules
clinear = ctypes.CDLL(os.path.abspath("./cmodules/linear_modulate"))
ctx = ctypes.CDLL(os.path.abspath("./cmodules/rrc_tx"))


def generate_linear(config):
    """Generate linear modulated signals with optional overlaps.

    Args:
        config (dict): Configuration dictionary containing parameters for signal generation.
    """

    # Ensure output directory exists
    os.makedirs(config["savepath"], exist_ok=True)

    for i in tqdm(range(0, config["n_captures"]), desc="Generating Data"):
        verbose = ctypes.c_int(config["verbose"])
        final_n_samps = config["n_samps"]
        n_samps_buf = final_n_samps + BUF

        k = np.random.choice(
            np.arange(1, config["max_overlaps"] + 1), p=config["overlap_split"]
        )

        open_set_mods = config.get("open_set_modulations", [])
        candidate_mods = config["modulation"] + open_set_mods
        mod_indices = np.random.choice(len(candidate_mods), k, replace=False)
        mods = [candidate_mods[idx] for idx in mod_indices]

        y_i_total = np.zeros(final_n_samps)
        y_q_total = np.zeros(final_n_samps)
        modnames = []
        modclasses = []
        orders = []

        for mod in mods:
            modtype = ctypes.c_int(mod[0])
            order = ctypes.c_int(mod[1])

            sps = int(np.random.choice(config["symbol_rate"]))
            beta = float(np.random.choice(config["rrc_filter"]["beta"]))
            delay = int(np.random.choice(config["rrc_filter"]["delay"]))
            dt = float(np.random.choice(config["rrc_filter"]["dt"]))

            chunk_n_sym = ctypes.c_int(max(1, n_samps_buf // sps))

            s = (ctypes.c_uint * chunk_n_sym.value)(
                *np.zeros(chunk_n_sym.value, dtype=int)
            )
            sm_i = (ctypes.c_float * chunk_n_sym.value)(*np.zeros(chunk_n_sym.value))
            sm_q = (ctypes.c_float * chunk_n_sym.value)(*np.zeros(chunk_n_sym.value))
            x_i = (ctypes.c_float * n_samps_buf)(*np.zeros(n_samps_buf))
            x_q = (ctypes.c_float * n_samps_buf)(*np.zeros(n_samps_buf))

            seed = ctypes.c_int(np.random.randint(1e9))
            clinear.linear_modulate(
                modtype, order, chunk_n_sym, s, sm_i, sm_q, verbose, seed
            )
            ctx.rrc_tx(
                chunk_n_sym,
                ctypes.c_int(sps),
                ctypes.c_uint(delay),
                ctypes.c_float(beta),
                ctypes.c_float(dt),
                sm_i,
                sm_q,
                x_i,
                x_q,
                verbose,
            )

            # Slice out buffer region
            sig_i = np.array(x_i)[HALF_BUF:-HALF_BUF]
            sig_q = np.array(x_q)[HALF_BUF:-HALF_BUF]

            # Draw initial phase uniformly on [-pi, pi)
            phi0 = np.random.uniform(-np.pi, np.pi)
            rot = np.exp(1j * phi0)
            sig_c = (sig_i + 1j * sig_q) * rot
            sig_i = sig_c.real.astype(np.float32, copy=False)
            sig_q = sig_c.imag.astype(np.float32, copy=False)

            amp = np.random.uniform(0.9, 1.1)  # per overlapped signal
            y_i_total += amp * sig_i
            y_q_total += amp * sig_q

            modnames.append(mod[-1])
            modclasses.append(modtype.value)
            orders.append(order.value)

        # choose SNR parameter for this capture
        snr_db, _, _ = config["channel_params"][
            np.random.randint(0, len(config["channel_params"]))
        ]

        # Form complex baseband signal from summed I/Q
        x = y_i_total + 1j * y_q_total

        # --- Add AWGN with correct variance for the desired SNR ---

        # Measure average signal power (Es per complex sample)
        sig_power = np.mean(np.abs(x) ** 2).astype(np.float64)

        # SNR in linear scale: gamma = Ps / Pn
        snr_linear = 10.0 ** (snr_db / 10.0)

        # Noise power per complex sample
        noise_power = sig_power / snr_linear

        # Standard deviation per real/imag component
        noise_std = np.sqrt(noise_power / 2.0)

        noise = noise_std * (
            np.random.normal(size=final_n_samps)
            + 1j * np.random.normal(size=final_n_samps)
        )

        # Noisy received signal
        y = x + noise

        # --- Per-capture AGC-style normalisation (unit RMS power) ---
        # This is analogous to scaling each example to unit energy/RMS as in RadioML,
        # and removes simulation artefacts while preserving SNR.
        rms = np.sqrt(np.mean(np.abs(y) ** 2)).astype(np.float64)
        if rms > 0.0:
            y = y / rms

        # Back to float32 I/Q for storage
        y_i_total = y.real.astype(np.float32, copy=False)
        y_q_total = y.imag.astype(np.float32, copy=False)

        metadata = {
            "modnames": modnames,
            "modclasses": modclasses,
            "orders": orders,
            "n_samps": final_n_samps,
            "channel_type": config["channel_type"],
            "snr": snr_db,
            "savepath": config["savepath"],
            "savename": config["savename"],
        }

        save_sigmf(y_i_total, y_q_total, metadata, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic RF datasets from a JSON configuration."
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to configuration file to use for data generation.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory in which to save the generated dataset.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="Random seed for data generation.",
    )
    args = parser.parse_args()

    if args.rng_seed is not None:
        np.random.seed(args.rng_seed)

    with open(args.config_file, encoding="utf-8") as f:
        config_file = json.load(f)

    # Print a copy of the loaded configuration for archiving of results.
    print(f"Contents of loaded configuration file ({args.config_file}):")
    print(json.dumps(config_file, indent=4))
    print("\n")

    # Inject savepath from CLI so map_config does not depend on JSON having it
    config_file["savepath"] = args.save_dir

    config_file = map_config(config_file, config_file)
    generate_linear(config_file)

    if config_file.get("archive", False):
        archive_sigmf(config_file["savepath"])
