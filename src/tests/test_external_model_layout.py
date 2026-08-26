"""Guard pinned external-model package boundaries and provenance."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ExternalModelLayoutTest(unittest.TestCase):
    def test_data_owners_are_cohesive_modules(self):
        data = ROOT / "src" / "data"
        self.assertFalse((data / "dataset.py").exists())
        for name in (
            "core.py",
            "sampling.py",
            "frames.py",
            "io.py",
            "splits.py",
            "statistics.py",
            "loaders.py",
        ):
            self.assertTrue((data / name).is_file(), name)

    def test_patchtst_is_a_named_pinned_package(self):
        external = ROOT / "src" / "external_models"
        self.assertFalse((external / "patchtst.py").exists())
        model = external / "patchtst" / "model.py"
        self.assertTrue(model.is_file())
        self.assertIn(
            "204c21efe0b39603ad6e2ca640ef5896646ab1a9",
            model.read_text(encoding="utf-8"),
        )

    def test_dlinear_is_a_named_pinned_package_when_used(self):
        external = ROOT / "src" / "external_models"
        package = external / "dlinear"
        if not package.exists():
            return
        self.assertFalse((external / "dlinear.py").exists())
        model = package / "model.py"
        self.assertIn(
            "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6",
            model.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
