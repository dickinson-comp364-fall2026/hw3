import io
import os
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DRY_BEAN_SRC_DIR = ROOT / "dry_bean" / "src"

for path in (ROOT, DRY_BEAN_SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from dry_bean import (
    load_dry_bean_data,
    MLP,
    train_model,
    validate_model,
    generate_diagnostic_report,
)
from dry_bean_utilities import load_dry_bean_data_from_ucirepo


class DryBeanDataLoadingTests(unittest.TestCase):
    def test_load_dry_bean_data_encodes_ucirepo_class_labels(self) -> None:
        df = pd.DataFrame(
            {
                "Area": [1.0, 2.0, 3.0],
                "Perimeter": [4.0, 5.0, 6.0],
                "Class": ["SEKER", "BARBUNYA", "SEKER"],
            }
        )

        with patch(
            "dry_bean.load_dry_bean_data_from_ucirepo",
            return_value=(df, df.drop(columns=["Class"]), pd.Series([1, 0, 1]), {"BARBUNYA": 0, "SEKER": 1}, {}, pd.DataFrame()),
        ):
            loaded_df, X, y_ids, class_to_id = load_dry_bean_data()

        self.assertEqual(len(loaded_df), 3)
        self.assertListEqual(list(X.columns), ["Area", "Perimeter"])

        # Sorted class names should map BARBUNYA->0, SEKER->1.
        self.assertEqual(class_to_id, {"BARBUNYA": 0, "SEKER": 1})
        self.assertListEqual(list(y_ids.values), [1, 0, 1])

    def test_load_dry_bean_data_from_ucirepo_downloads_and_encodes(self) -> None:
        features = pd.DataFrame({"Area": [1.0, 2.0], "Perimeter": [3.0, 4.0]})
        targets = pd.DataFrame({"Class": ["SEKER", "BARBUNYA"]})

        class FakeDataset:
            def __init__(self):
                self.data = type("Data", (), {"features": features, "targets": targets})()
                self.metadata = {"id": 602}
                self.variables = pd.DataFrame({"name": ["Area", "Perimeter", "Class"]})

        fetcher = unittest.mock.Mock(return_value=FakeDataset())

        with tempfile.TemporaryDirectory() as temp_dir:
            df, X, y_ids, class_to_id, metadata, variables = load_dry_bean_data_from_ucirepo(
                fetcher, cache_dir=Path(temp_dir)
            )

        fetcher.assert_called_once_with(id=602)
        self.assertListEqual(list(df.columns), ["Area", "Perimeter", "Class"])
        self.assertListEqual(list(X.columns), ["Area", "Perimeter"])
        self.assertEqual(class_to_id, {"BARBUNYA": 0, "SEKER": 1})
        self.assertListEqual(list(y_ids.values), [1, 0])
        self.assertEqual(metadata, {"id": 602, "uci_id": 602})
        self.assertListEqual(list(variables["name"]), ["Area", "Perimeter", "Class"])

    def test_load_dry_bean_data_from_ucirepo_uses_cache_after_first_download(self) -> None:
        row_count = 1001
        features = pd.DataFrame(
            {
                **{f"feature_{index}": list(range(row_count)) for index in range(16)},
            }
        )
        targets = pd.DataFrame({"Class": ["SEKER"] * row_count})

        class FakeDataset:
            def __init__(self):
                self.data = type("Data", (), {"features": features, "targets": targets})()
                self.metadata = {"id": 602, "source": "uci"}
                self.variables = pd.DataFrame({"name": [*features.columns, "Class"]})

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            fetcher = unittest.mock.Mock(return_value=FakeDataset())

            first_result = load_dry_bean_data_from_ucirepo(fetcher, cache_dir=cache_dir)
            fetcher.assert_called_once_with(id=602)

            cached_fetcher = unittest.mock.Mock(side_effect=AssertionError("cache should prevent a second download"))
            second_result = load_dry_bean_data_from_ucirepo(cached_fetcher, cache_dir=cache_dir)

        first_df, first_X, first_y_ids, first_class_to_id, first_metadata, first_variables = first_result
        second_df, second_X, second_y_ids, second_class_to_id, second_metadata, second_variables = second_result

        self.assertTrue(first_df.equals(second_df))
        self.assertTrue(first_X.equals(second_X))
        self.assertListEqual(list(first_y_ids.values), list(second_y_ids.values))
        self.assertEqual(first_class_to_id, second_class_to_id)
        self.assertEqual(first_metadata, second_metadata)
        self.assertTrue(first_variables.equals(second_variables))

    def test_load_dry_bean_data_from_ucirepo_redownloads_invalid_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir)
            (cache_dir / "dry_bean_uci.csv").write_text("Area,Class\n1,SEKER\n", encoding="utf-8")
            (cache_dir / "dry_bean_uci_metadata.json").write_text("{}", encoding="utf-8")
            pd.DataFrame({"name": ["Area", "Class"]}).to_csv(cache_dir / "dry_bean_uci_variables.csv", index=False)

            features = pd.DataFrame(
                {"Area": list(range(1001)), "Perimeter": list(range(1001, 2002))}
            )
            targets = pd.DataFrame({"Class": ["SEKER"] * 1001})

            class FakeDataset:
                def __init__(self):
                    self.data = type("Data", (), {"features": features, "targets": targets})()
                    self.metadata = {"uci_id": 602, "source": "uci"}
                    self.variables = pd.DataFrame({"name": ["Area", "Perimeter", "Class"]})

            fetcher = unittest.mock.Mock(return_value=FakeDataset())

            df, X, y_ids, class_to_id, metadata, variables = load_dry_bean_data_from_ucirepo(
                fetcher, cache_dir=cache_dir
            )

        fetcher.assert_called_once_with(id=602)
        self.assertEqual(df.shape, (1001, 3))
        self.assertEqual(X.shape, (1001, 2))
        self.assertEqual(class_to_id, {"SEKER": 0})
        self.assertEqual(metadata["uci_id"], 602)
        self.assertListEqual(list(variables["name"]), ["Area", "Perimeter", "Class"])
        self.assertEqual(len(y_ids), 1001)

    def test_load_dry_bean_data_from_ucirepo_creates_default_cache_directory(self) -> None:
        row_count = 1001
        features = pd.DataFrame({**{f"feature_{index}": list(range(row_count)) for index in range(16)}})
        targets = pd.DataFrame({"Class": ["SEKER"] * row_count})

        class FakeDataset:
            def __init__(self):
                self.data = type("Data", (), {"features": features, "targets": targets})()
                self.metadata = {"id": 602}
                self.variables = pd.DataFrame({"name": [*features.columns, "Class"]})

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            old_cwd = Path.cwd()
            try:
                (repo_root / "dry_bean").mkdir()
                fetcher = unittest.mock.Mock(return_value=FakeDataset())
                import os

                os.chdir(repo_root)
                load_dry_bean_data_from_ucirepo(fetcher)
            finally:
                os.chdir(old_cwd)

            self.assertTrue((repo_root / "dry_bean" / "data").is_dir())
            self.assertTrue((repo_root / "dry_bean" / "data" / "dry_bean_uci.csv").exists())

    def test_load_dry_bean_data_creates_default_cache_directory_via_main_loader(self) -> None:
        row_count = 1001
        features = pd.DataFrame({**{f"feature_{index}": list(range(row_count)) for index in range(16)}})
        targets = pd.DataFrame({"Class": ["SEKER"] * row_count})

        class FakeDataset:
            def __init__(self):
                self.data = type("Data", (), {"features": features, "targets": targets})()
                self.metadata = {"id": 602}
                self.variables = pd.DataFrame({"name": [*features.columns, "Class"]})

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            cache_dir = repo_root / "dry_bean" / "data"
            old_cwd = Path.cwd()
            try:
                (repo_root / "dry_bean").mkdir()
                os.chdir(repo_root)
                with patch("ucimlrepo.fetch_ucirepo", return_value=FakeDataset()):
                    load_dry_bean_data()
            finally:
                os.chdir(old_cwd)

            self.assertTrue(cache_dir.is_dir())
            self.assertTrue((cache_dir / "dry_bean_uci.csv").exists())


class DryBeanModelTests(unittest.TestCase):
    def test_mlp_depth_zero_has_single_linear_layer(self) -> None:
        model = MLP(input_dim=5, width=8, depth=0, output_dim=3)

        self.assertEqual(len(model.layers), 1)
        self.assertIsInstance(model.layers[0], torch.nn.Linear)
        self.assertEqual(model.layers[0].in_features, 5)
        self.assertEqual(model.layers[0].out_features, 3)

    def test_mlp_forward_shape_with_hidden_layers(self) -> None:
        model = MLP(input_dim=4, width=6, depth=2, output_dim=3)
        X = torch.randn(7, 4)

        logits = model(X)

        self.assertEqual(logits.shape, (7, 3))


class DryBeanTrainingAndValidationTests(unittest.TestCase):
    def test_train_model_returns_loss_history(self) -> None:
        torch.manual_seed(123)
        model = MLP(input_dim=4, width=10, depth=1, output_dim=3)

        X_train = torch.randn(24, 4)
        y_train = torch.randint(0, 3, (24,), dtype=torch.long)

        loss_history = train_model(model, X_train, y_train, num_epochs=4, batch_size=8)

        self.assertEqual(len(loss_history), 4)
        self.assertTrue(all(isinstance(loss, float) for loss in loss_history))
        self.assertTrue(all(loss >= 0.0 for loss in loss_history))

    def test_validate_model_returns_expected_error_rate_without_diagnostics(self) -> None:
        # Build logits that produce predictions [0, 1, 2, 1].
        logits = torch.tensor(
            [
                [5.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 5.0],
                [0.0, 5.0, 0.0],
            ],
            dtype=torch.float32,
        )

        class FixedModel(torch.nn.Module):
            def __init__(self, fixed_logits: torch.Tensor):
                super().__init__()
                self.fixed_logits = fixed_logits

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fixed_logits[: x.shape[0]]

        model = FixedModel(logits)
        X_test = torch.zeros(4, 4)
        y_test = torch.tensor([0, 1, 2, 0], dtype=torch.long)

        error_rate = validate_model(model, X_test, y_test, show_diagnostics=False)

        # 3 correct out of 4 => error rate 0.25
        self.assertAlmostEqual(error_rate, 0.25, places=6)


class DryBeanDiagnosticsTests(unittest.TestCase):
    def test_generate_diagnostic_report_prints_class_names_and_summary(self) -> None:
        y_test = torch.tensor([0, 1, 1, 0], dtype=torch.long)
        predictions = torch.tensor([0, 1, 0, 0], dtype=torch.long)
        class_to_id = {"ALPHA": 0, "BETA": 1}

        with patch("sys.stdout", new_callable=io.StringIO) as fake_stdout:
            generate_diagnostic_report(y_test, predictions, class_to_id=class_to_id)
            output = fake_stdout.getvalue()

        self.assertIn("Validation diagnostics (multi-class classification)", output)
        self.assertIn("Overall accuracy:", output)
        self.assertIn("Macro-avg F1:", output)
        self.assertIn("Weighted-avg F1:", output)
        self.assertIn("ALPHA", output)
        self.assertIn("BETA", output)


if __name__ == "__main__":
    unittest.main()
