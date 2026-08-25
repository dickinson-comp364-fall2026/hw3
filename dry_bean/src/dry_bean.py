"""
Train a multi-layer perceptron (MLP) for the dry bean classification task.

This script demonstrates how to build and train an MLP using PyTorch for a
multi-class classification problem.
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dry_bean_utilities import (
    generate_diagnostic_report,
    load_dry_bean_data_from_ucirepo,
    plot_training_comparison,
    print_dataset_info,
    seed_torch,
)

############################################################
## Hyperparameters and model configuration
############################################################
# TODO: Tune these hyperparameters to achieve good performance on the test set.
# number of hidden layers (not counting the final output layer)
MODEL_DEPTH = 1
# number of neurons in each hidden layer
MODEL_WIDTH = 1
EPOCHS = 3
BATCH_SIZE = 10
LEARNING_RATE = 0.005
############################################################




# Fixed seed for reproducibility
TORCH_SEED = 54545


def load_dry_bean_data():
    df, X, y_ids, class_to_id, _, _ = load_dry_bean_data_from_ucirepo()
    return df, X, y_ids, class_to_id


class MLP(nn.Module):
    def __init__(self, input_dim: int, width: int, depth: int, output_dim: int):
        """
        Args:
            input_dim: Dimension of the input features (int)
            width: Width of each hidden layer (int)
            depth: Number of hidden layers (int)
            output_dim: Dimension of the output (int)

        """
        super().__init__()

        layers = []

        if depth == 0:
            # No hidden layers, just a direct linear mapping from input to output
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.relu = nn.ReLU()
            # TODO: Implement the MLP architecture with the specified number of hidden layers and width.
            # Don't forget the activation functions.
            raise NotImplementedError()

        self.layers = nn.ModuleList(layers)

    def forward(self, x):     # x shape is (N, MODEL_INPUT_DIM), where N is batch size
        for layer in self.layers:
            # hidden layers keep (N, MODEL_WIDTH); final layer outputs (N, 1)
            # TODO: Implement the forward pass of the MLP. It's a single line of code.
            raise NotImplementedError()
        logit = x             # final logits shape is (N, 1)
        return logit


def train_model(model: MLP, X_train: torch.Tensor, y_train: torch.Tensor, num_epochs: int, batch_size: int = 32) -> list[float]:
    """Train the model and return the list of loss values per iteration."""

    # shape of X_train: (num_instances, 3)
    # shape of y_train: (num_instances, 1)

    # TODO: Replace None with an appropriate loss function (criterion) for multi-class classification,
    # and delete the NotImplementedError.
    raise NotImplementedError()
    criterion = None


    # TODO: Replace None with an appropriate Adam optimizer for the model,
    # and delete the NotImplementedError.
    raise NotImplementedError()
    optimizer = None

    # Wrap tensors into a dataset
    dataset = TensorDataset(X_train, y_train)


    # TODO: Insert a comment explaining why we use shuffle=True 
    # and exactly what effect it has.
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True)

    model.train()  # set model to training mode
    loss_history = []
    for _ in range(num_epochs):
        loss_total = 0.0
        num_instances = X_train.shape[0]
        for batch in loader:
            X_batch, y_batch = batch


            # Forward pass: compute predicted logits. Under the covers, 
            # Pytorch builds a computation graph, which will allow us to compute derivatives later.
            # TODO: Replace None with an appropriate computation,
            # and delete the NotImplementedError. Also write a comment indicating the shape of the logits tensor.
            raise NotImplementedError()
            logits = None


            # Calculate loss for this batch. PyTorch extends the computation graph, 
            # which will allow computation of the derivatives of 
            # the loss with respect to parameters when we do the backward pass later.
            # TODO: Replace None with an appropriate computation,
            # and delete the NotImplementedError. Also write a comment indicating the shape of each tensor.
            raise NotImplementedError()
            loss = None

            current_batch_size = X_batch.shape[0]
            loss_total += loss.item() * current_batch_size
            # Backward pass and optimization step
            # TODO: Implement the backward pass and optimization step. It's three lines of code.
            raise NotImplementedError()

        # average loss per instance
        loss_history.append(loss_total / num_instances)
    return loss_history


def validate_model(model: MLP, X_test: torch.Tensor, y_test: torch.Tensor, show_diagnostics: bool = True, class_to_id: dict[str, int] | None = None) -> float:
    """Evaluate on test data, print multi-class diagnostics, and return error rate."""

    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        logits = model(X_test)  # shape (num_instances, num_classes)
        # TODO: Replace None with an appropriate computation,
        # and delete the NotImplementedError. Also write a comment indicating the shape of 
        # the predictions tensor.
        raise NotImplementedError()
        predictions = None

        if show_diagnostics:
            generate_diagnostic_report(y_test, predictions, class_to_id=class_to_id)

        # TODO: Write a comment, explaining in your own words why the 
        # following line calculates the number of correct predictions.
        correct = (predictions == y_test).sum().item()
        
        total = y_test.shape[0]
        error_rate = 1.0 - correct / total
    return error_rate


def main():
    seed_torch(TORCH_SEED)
    df, X, y_ids, class_to_id = load_dry_bean_data()
    print_dataset_info(df, X, y_ids)
    print(f"class_to_id mapping: {class_to_id}")
    # num_instances = X.shape[0]


    # TODO: Replace None with an appropriate constructor.
    raise NotImplementedError()
    model = None
    
    print(model)
    # Split into train and test sets (NumPy arrays)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_ids.values, test_size=0.2, random_state=TORCH_SEED)

    # Scale features (still NumPy arrays)
    # TODO: Write a comment, explaining in your own words why  
    # we need the following three lines and what they do.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print("\nTraining model...")
    start_time = time.time()
    loss_history = train_model(
        model, X_train, y_train, num_epochs=EPOCHS, batch_size=BATCH_SIZE)

    elapsed_time = time.time() - start_time

    test_error = validate_model(model, X_test, y_test, class_to_id=class_to_id)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"epochs: {EPOCHS}")
    print(f"final training loss: {loss_history[-1]:.6f}")
    print(f"training time: {elapsed_time:.3f} seconds")
    print(f"test error: {test_error:.3f}")
    print("=" * 60)

    plot_training_comparison(loss_history)


if __name__ == "__main__":
    main()
