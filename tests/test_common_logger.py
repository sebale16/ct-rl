# tests/test_common_logger.py

import unittest
import os
import shutil
import json
import csv

from common.logger import configure, record, record_mean, dump, close, get_logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.log_folder = "test_logs/"
        # Ensure the folder is clean before each test
        if os.path.exists(self.log_folder):
            shutil.rmtree(self.log_folder)

    def tearDown(self):
        # Clean up the folder after each test
        if os.path.exists(self.log_folder):
            shutil.rmtree(self.log_folder)
        # Close the logger to release file handles
        close()

    def test_configure_and_get_logger(self):
        configure(folder=self.log_folder, output_formats=["stdout"])
        logger_instance = get_logger()
        self.assertIsNotNone(logger_instance)
        self.assertEqual(logger_instance.folder, self.log_folder)

    def test_record_and_dump_json(self):
        configure(folder=self.log_folder, output_formats=["json"])
        record("test/value", 10)
        record("other/value", 20.5)
        dump(step=1)

        json_path = os.path.join(self.log_folder, "progress.json")
        self.assertTrue(os.path.exists(json_path))

        with open(json_path, "r") as f:
            data = json.load(f)
            self.assertEqual(data["test/value"], 10)
            self.assertEqual(data["other/value"], 20.5)

    def test_record_and_dump_csv(self):
        configure(folder=self.log_folder, output_formats=["csv"])
        record("a", 1)
        record("b", 2)
        dump(step=0)
        record("a", 3)
        record("b", 4)
        dump(step=1)

        csv_path = os.path.join(self.log_folder, "progress.csv")
        self.assertTrue(os.path.exists(csv_path))

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ["a", "b"])
            row1 = next(reader)
            self.assertEqual(row1, ["1", "2"])
            row2 = next(reader)
            self.assertEqual(row2, ["3", "4"])

    def test_record_mean(self):
        configure(folder=self.log_folder, output_formats=[])  # No output needed
        record_mean("mean_val", 10.0)
        record_mean("mean_val", 20.0)
        logger = get_logger()
        self.assertAlmostEqual(logger.name_to_value["mean_val"], 15.0)


if __name__ == "__main__":
    unittest.main()
