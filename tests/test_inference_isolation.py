import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_FILES = (
    PROJECT_ROOT / "test_lut.py",
    PROJECT_ROOT / "scripts" / "calculate.py",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "teachers",
    "vgg",
    "fine_tune_lut",
    "train_distillation",
    "scripts.loss_lut",
)


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


class InferenceIsolationTest(unittest.TestCase):
    def test_inference_files_do_not_import_training_only_components(self):
        for path in INFERENCE_FILES:
            with self.subTest(path=path):
                modules = imported_modules(path)
                forbidden = [
                    module
                    for module in modules
                    if module.startswith(FORBIDDEN_IMPORT_PREFIXES)
                ]
                self.assertEqual(
                    forbidden,
                    [],
                    f"{path.name} imports training-only modules: {forbidden}",
                )


if __name__ == "__main__":
    unittest.main()
