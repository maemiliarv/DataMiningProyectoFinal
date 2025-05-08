import os
import joblib
import logging
import argparse
import pandas as pd


def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger that logs to stdout and optionally to a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Stream handler
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(path)


def save_model(model, path: str):
    """
    Persist a model to disk using joblib.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def parse_args():
    """
    Parse command-line arguments for the training script.
    """
    parser = argparse.ArgumentParser(description="Train LTV/CAC prediction model.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file with data and model parameters."
    )
    return parser.parse_args()