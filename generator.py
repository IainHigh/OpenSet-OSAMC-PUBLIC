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
import copy
import json
import os
import ctypes
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

# third party imports
from tqdm import tqdm
import numpy as np

## internal imports
from utils.sigmf_utils import save_sigmf
from utils.config_utils import map_config


BUF = 64
HALF_BUF = BUF // 2

PROJECT_ROOT = Path(__file__).resolve().parent
CMODULE_DIR = PROJECT_ROOT / "cmodules"


def _load_cdll(module_name: str) -> ctypes.CDLL:
    """Load a shared library relative to this file, not the cwd."""
    return ctypes.CDLL(str((CMODULE_DIR / module_name).resolve()))


clinear = _load_cdll("linear_modulate")
ctx = _load_cdll("rrc_tx")


def load_config_file(config_file: str) -> Dict[str, Any]:
    """Load a raw JSON config into memory."""
    with open(config_file, encoding="utf-8") as f:
        return json.load(f)


def prepare_generation_config(
    raw_config: Dict[str, Any],
    save_dir: str,
) -> Dict[str, Any]:
    """
    Copy, inject the save path, and map the config into the form expected by
    generate_linear().
    """
    config = copy.deepcopy(raw_config)
    config["savepath"] = save_dir
    return map_config(config, config)


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


def run_generator_from_config(
    raw_config: Dict[str, Any],
    save_dir: str,
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate a dataset from a config already loaded into memory.
    Returns the mapped config actually used.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    print("Contents of loaded configuration snapshot:")
    print(json.dumps(raw_config, indent=4))
    print("\n")

    mapped_config = prepare_generation_config(raw_config, save_dir)
    generate_linear(mapped_config)
    return mapped_config


def run_generator(
    config_file: str,
    save_dir: str,
    rng_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """CLI-friendly wrapper that loads the config from disk and runs generation."""
    raw_config = load_config_file(config_file)
    return run_generator_from_config(raw_config, save_dir, rng_seed)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parses the command line arguments if entered through CLI.
    Args:
        argv (Optional[Sequence[str]], optional): Command line arguments. Defaults to None.

    Returns:
        argparse.Namespace: Parsed arguments from the command line.
    """
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
    return parser.parse_args(argv)


def cli(argv: Optional[Sequence[str]] = None) -> None:
    """Helper function for when script is excecuted through CLI.

    Args:
        argv (Optional[Sequence[str]], optional): Command line arguments. Defaults to None.
    """
    args = parse_args(argv)
    run_generator(
        config_file=args.config_file,
        save_dir=args.save_dir,
        rng_seed=args.rng_seed,
    )


if __name__ == "__main__":
    cli()
