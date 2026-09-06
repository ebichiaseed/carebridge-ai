import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.run_log_service import write_run_log


class RunLogServiceTests(unittest.TestCase):
    def test_writes_one_readable_json_file_per_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"CAREBRIDGE_RUN_LOG_DIR": directory}):
                path = write_run_log(
                    "example-run",
                    {"status": "verified", "translation": "Take medicine."},
                )

            self.assertEqual(path, Path(directory) / "example-run.json")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "verified", "translation": "Take medicine."},
            )
            self.assertFalse((Path(directory) / ".example-run.tmp").exists())


if __name__ == "__main__":
    unittest.main()
