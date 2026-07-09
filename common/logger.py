# common/logger.py
from __future__ import annotations

import os
import sys
import time
import json
import csv
from collections import defaultdict
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

import numpy as np
import torch as th


class FormatUnsupportedError(Exception):
    pass


class KVWriter(object):
    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class HumanOutputFormat(KVWriter):
    def __init__(self, file: TextIO):
        self.file = file

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        # Create a tabular format
        key_width = max(len(key) for key in key_values.keys())
        val_width = max(len(str(val)) for val in key_values.values())
        
        hr = "-" * (key_width + val_width + 7)
        self.file.write(hr + "\n")
        for key, val in key_values.items():
            self.file.write(f"| {key:<{key_width}} | {str(val):<{val_width}} |\n")
        self.file.write(hr + "\n")
        self.file.flush()


class JSONOutputFormat(KVWriter):
    def __init__(self, file_path: str, append: bool = False):
        self.file = open(file_path, "at" if append else "wt")

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        # Add step to the data
        # key_values["step"] = step # Add steps later
        self.file.write(json.dumps(key_values) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, file_path: str, append: bool = False):
        # When appending across a resume, reuse the existing header so rows stay
        # column-aligned and no duplicate header line is written mid-file.
        self._existing_keys = None
        if append and os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, "r", newline="") as f:
                header = f.readline().strip()
            if header:
                self._existing_keys = header.split(",")
        self.file = open(file_path, "a" if append else "w", newline="")
        self.writer = None
        self.keys = []

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        if self.writer is None:
            if self._existing_keys is not None:
                self.keys = self._existing_keys
                self.writer = csv.DictWriter(self.file, fieldnames=self.keys)
                # header already present in the file we are appending to
            else:
                self.keys = sorted(key_values.keys())
                self.writer = csv.DictWriter(self.file, fieldnames=self.keys)
                self.writer.writeheader()

        self.writer.writerow(
            {key: key_values.get(key, "") for key in self.keys}
        )
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    def __init__(self, folder: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise FormatUnsupportedError("tensorboard")
        self.writer = SummaryWriter(folder)

    def write(self, key_values: Dict[str, Any], step: int = 0) -> None:
        for key, value in key_values.items():
            if isinstance(value, (np.ScalarType, int, float)):
                self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        self.writer.close()


class Logger:
    def __init__(
        self,
        folder: Optional[str],
        output_formats: List[str],
        append: bool = False,
    ):
        self.name_to_value: Dict[str, Union[float, int]] = defaultdict(float)
        self.name_to_count: Dict[str, int] = defaultdict(int)
        self.name_to_excluded: Dict[str, str] = {}
        self.folder = folder

        self.output_formats: List[KVWriter] = []
        if folder is not None:
            os.makedirs(folder, exist_ok=True)
            for fmt in output_formats:
                if fmt == "stdout":
                    self.output_formats.append(HumanOutputFormat(sys.stdout))
                elif fmt == "log":
                    self.output_formats.append(
                        HumanOutputFormat(open(f"{folder}/log.txt", "at" if append else "wt"))
                    )
                elif fmt == "json":
                    self.output_formats.append(
                        JSONOutputFormat(f"{folder}/progress.json", append=append)
                    )
                elif fmt == "csv":
                    self.output_formats.append(
                        CSVOutputFormat(f"{folder}/progress.csv", append=append)
                    )
                elif fmt == "tensorboard":
                    self.output_formats.append(TensorBoardOutputFormat(folder))

    def record(self, key: str, value: Any, exclude: Optional[str] = None) -> None:
        """
        Log a value of some diagnostic.
        If it has a '/' in its name, it will be saved with that hierarchy.
        """
        self.name_to_value[key] = value
        if exclude is not None:
            self.name_to_excluded[key] = exclude

    def record_mean(self, key: str, value: float, exclude: Optional[str] = None) -> None:
        """
        Log the mean of a diagnostic.
        """
        if value is None:
            return
        self.name_to_value[key] = (self.name_to_value[key] * self.name_to_count[key] + value) / (self.name_to_count[key] + 1)
        self.name_to_count[key] += 1
        if exclude is not None:
            self.name_to_excluded[key] = exclude

    def dump(self, step: int = 0) -> None:
        """
        Write all recorded values to the configured output formats and clear the buffer.
        """
        key_values = {key: val for key, val in self.name_to_value.items()}
        
        for writer in self.output_formats:
            writer.write(key_values, step)

        self.name_to_value.clear()
        self.name_to_count.clear()
        self.name_to_excluded.clear()

    def close(self) -> None:
        for writer in self.output_formats:
            writer.close()


class LogManager:
    def __init__(self):
        self._loggers: Dict[str, Logger] = {}

    def configure(
        self, folder: Optional[str], output_formats: List[str], append: bool = False
    ) -> None:
        self._loggers["default"] = Logger(folder, output_formats, append=append)

    def get_logger(self) -> Logger:
        return self._loggers.get("default", None)

    def __getattr__(self, name):
        """
        Redirect method calls to the default logger.
        """
        if "default" not in self._loggers:
            # Default logger not configured, create a dummy one
            self._loggers["default"] = Logger(None, [])

        logger = self.get_logger()
        
        def method(*args, **kwargs):
            if hasattr(logger, name) and callable(getattr(logger, name)):
                return getattr(logger, name)(*args, **kwargs)
            raise AttributeError(f"'Logger' object has no attribute '{name}'")
        
        return method


# Global logger instance
logger = LogManager()


def configure(
    folder: Optional[str], output_formats: List[str], append: bool = False
) -> None:
    """
    Configure the global logger.
    """
    logger.configure(folder, output_formats, append=append)


def record(key: str, value: Any, exclude: Optional[str] = None) -> None:
    """
    Log a value of some diagnostic.
    """
    logger.record(key, value, exclude)


def record_mean(key: str, value: float, exclude: Optional[str] = None) -> None:
    """
    Log the mean of a diagnostic.
    """
    logger.record_mean(key, value, exclude)


def dump(step: int = 0) -> None:
    """
    Write all recorded values to the configured output formats.
    """
    logger.dump(step)


def get_logger() -> Logger:
    """
    Get the global logger instance.
    """
    return logger.get_logger()


def close() -> None:
    """
    Close the global logger.
    """
    logger.close()


def safe_mean(arr: Union[List, np.ndarray]) -> float:
    """
    Compute the mean of an array if it is not empty, otherwise return NaN.
    """
    return np.nan if len(arr) == 0 else float(np.mean(arr))