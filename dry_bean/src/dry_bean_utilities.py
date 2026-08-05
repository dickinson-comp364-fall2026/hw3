"""Provide utilities for the Dry Bean dataset and PyTorch MLP."""

import json
from pathlib import Path

import pandas as pd
import torch


MIN_DRY_BEAN_ROWS = 1000
MIN_DRY_BEAN_COLUMNS = 17


def is_valid_dry_bean_cache(df: pd.DataFrame, metadata, variables: pd.DataFrame) -> bool:
    """Return True when cached Dry Bean artifacts look complete enough to trust."""
    if "Class" not in df.columns:
        return False
    if df.shape[0] < MIN_DRY_BEAN_ROWS:
        return False
    if df.shape[1] < MIN_DRY_BEAN_COLUMNS:
        return False
    if not isinstance(metadata, dict) or metadata.get("uci_id") != 602:
        return False
    if variables.empty:
        return False
    return True


def configure_torch_determinism(enabled: bool) -> None:
    """Configure PyTorch deterministic behavior.

    When enabled, PyTorch is asked to use deterministic kernels where
    available. This improves reproducibility, especially on GPU.
    """
    torch.use_deterministic_algorithms(enabled)
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled


def seed_torch(seed: int) -> None:
    """Set PyTorch global RNG state using a fixed seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configure_torch_determinism(enabled=True)


def load_dry_bean_data_from_dataframe(df: pd.DataFrame):
    """Encode Dry Bean labels from a DataFrame with a Class column."""
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Encode string labels into contiguous class IDs for CrossEntropyLoss.
    class_names = sorted(y.unique())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    y_ids = y.map(class_to_id).astype("int64")

    return df, X, y_ids, class_to_id


def load_dry_bean_data_from_ucirepo(fetch_dataset=None, cache_dir: Path | None = None):
    """Load Dry Bean data from a local cache, downloading it from UCI on cache miss."""
    if cache_dir is None:
        cache_dir = Path.cwd() / "dry_bean" / "data"

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = cache_dir / "dry_bean_uci.csv"
    metadata_path = cache_dir / "dry_bean_uci_metadata.json"
    variables_path = cache_dir / "dry_bean_uci_variables.csv"

    if dataset_path.exists() and metadata_path.exists() and variables_path.exists():
        try:
            df = pd.read_csv(dataset_path)
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
            variables = pd.read_csv(variables_path)
        except (OSError, json.JSONDecodeError, pd.errors.EmptyDataError):
            df = None
            metadata = None
            variables = None

        if df is not None and is_valid_dry_bean_cache(df, metadata, variables):
            encoded_df, encoded_X, y_ids, class_to_id = load_dry_bean_data_from_dataframe(df)
            return encoded_df, encoded_X, y_ids, class_to_id, metadata, variables

    if fetch_dataset is None:
        from ucimlrepo import fetch_ucirepo

        fetch_dataset = fetch_ucirepo

    dry_bean = fetch_dataset(id=602)
    X = dry_bean.data.features
    y = dry_bean.data.targets

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    df = pd.concat([X, y.rename("Class")], axis=1)
    metadata = dict(dry_bean.metadata)
    metadata["uci_id"] = 602
    df.to_csv(dataset_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    dry_bean.variables.to_csv(variables_path, index=False)

    encoded_df, encoded_X, y_ids, class_to_id = load_dry_bean_data_from_dataframe(df)
    return encoded_df, encoded_X, y_ids, class_to_id, metadata, dry_bean.variables


def instances_to_tensors(instances) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of instances into feature and label tensors."""
    X = torch.tensor([[inst.v, inst.p, inst.e] for inst in instances], dtype=torch.float32)
    y = torch.tensor([inst.calc_target_value() for inst in instances], dtype=torch.float32).unsqueeze(1)
    return X, y


def print_dataset_info(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> None:
    """Print lightweight dataset diagnostics for interactive runs."""
    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    print(f"X.head():\n{X.head()}")
    print(f"y.head():\n{y.head()}")
    print(f"X.columns:\n{X.columns}")
    print(f"y.unique():\n{y.unique()}")
    print(f"df.describe():\n{df.describe()}")
    print(f"df.info():\n{df.info()}")
    print(f"df['Class'].value_counts():\n{df['Class'].value_counts()}")
    print(f"df.isnull().sum():\n{df.isnull().sum()}")
    print(f"df.sample(5):\n{df.sample(5)}")


def generate_diagnostic_report(y_test, predictions, class_to_id: dict[str, int] | None = None):
    """Generate multi-class diagnostics with per-class precision/recall/F1."""

    def safe_divide(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator != 0 else 0.0

    classes = torch.unique(torch.cat([y_test, predictions])).numpy()
    num_classes = len(classes)

    id_to_class = None
    if class_to_id:
        id_to_class = {v: k for k, v in class_to_id.items()}

    correct = (predictions == y_test).sum().item()
    total = y_test.shape[0]
    overall_accuracy = safe_divide(correct, total)

    per_class_metrics = {}

    for cls in classes:
        cls_int = int(cls)
        tp = ((predictions == cls_int) & (y_test == cls_int)).sum().item()
        fp = ((predictions == cls_int) & (y_test != cls_int)).sum().item()
        fn = ((predictions != cls_int) & (y_test == cls_int)).sum().item()

        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)

        per_class_metrics[cls_int] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": (y_test == cls_int).sum().item(),
        }

    macro_f1 = sum(m["f1"] for m in per_class_metrics.values()) / num_classes
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_class_metrics.values()) / total

    print("\nValidation diagnostics (multi-class classification):")
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print("\nPer-class metrics:")
    print(f"{'Class':>6} {'Name':>16} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for cls in sorted(per_class_metrics.keys()):
        m = per_class_metrics[cls]
        name = id_to_class[cls] if id_to_class and cls in id_to_class else str(cls)
        print(f"{cls:>6} {name:>16} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")
    print(f"\nMacro-avg F1:    {macro_f1:.4f}")
    print(f"Weighted-avg F1: {weighted_f1:.4f}")


def plot_training_comparison(loss_history: list[float]) -> None:
    """Plot training loss curve."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    epoch_history = list(range(1, len(loss_history) + 1))
    plt.plot(epoch_history, loss_history, linewidth=2, label="Training")

    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
