"""Train a single-neuron gym decision model with sigmoid and BCE using pytorch.

This script contrasts with gym_decision_neuron_BCE.py, because it uses PyTorch
to implement the artificial neuron and its training loop. The objective is to
understand how to use PyTorch for elementary neural network operations, after
previously understanding those same operations from first principles in the file
gym_decision_neuron_BCE.py (which did not employ PyTorch).
"""


import time

import torch
import torch.nn as nn

from gym_decision_shared import (
    InputInstance,
    Parameters,
    create_train_test_sets,
    print_final_summary,
)
from gym_decision_pytorch_shared import (
    seed_everything,
    configure_torch_determinism,
    instances_to_tensors,
)


SEED = 1234


class GymNeuron(nn.Module):
    def __init__(self):
        super().__init__()

        # one learnable neuron, 3 inputs → 4 parameters (3 weights + 1 bias).
        # The three weights correspond to the three features: hours at gym,
        # hours of prep needed, and energy level.
        # TODO: Replace None with appropriate code, and delete the raise statement.
        raise NotImplementedError()
        self.neuron = None

    def forward(self, x):              # x: shape (N, 3), where N is the batch size
        # TODO: Replace None with appropriate code, and delete the raise statement.
        raise NotImplementedError()
        logit = None                   # shape (N, 1)
        return logit                   # shape (N, 1)


def model_to_parameters(model: GymNeuron) -> Parameters:
    """Extract model weights/bias into shared Parameters format."""
    weight_values = model.neuron.weight.detach().cpu().numpy().reshape(-1)
    bias_value = float(model.neuron.bias.detach().cpu().numpy().reshape(-1)[0])
    return Parameters.from_values(
        v_wt=float(weight_values[0]),
        p_wt=float(weight_values[1]),
        e_wt=float(weight_values[2]),
        bias=bias_value,
    )

def train_model(model: GymNeuron, X_train: torch.Tensor, y_train: torch.Tensor, num_epochs: int) -> list[float]:
    """Train the model using SGD and return the list of loss values per iteration. 
    This approach does not use batching."""

    # shape of X_train: (num_instances, 3)
    # shape of y_train: (num_instances, 1)

    # TODO: Add a comment written entirely in your own words, explaining what the 
    # following loss function computes and why it is used in this context.
    criterion = nn.BCEWithLogitsLoss()  # combines sigmoid + binary cross-entropy

    # TODO: Replace None with appropriate code to create an SGD optimizer 
    # for the model parameters with a learning rate of 0.1., 
    # and delete the raise statement.
    raise NotImplementedError()
    optimizer = None

    loss_history = []
    # Ensure the model is in training mode so layers like dropout/batchnorm behave
    # correctly (even though this model doesn't use them, it's good practice).
    model.train()
    for _ in range(num_epochs):
        loss_total = 0.0
        num_instances = X_train.shape[0]
        for i in range(num_instances):
            # Forward pass: compute predicted logits
            # TODO: Replace None with appropriate code to compute the predicted logit 
            # for the i-th training instance, and delete the raise statement.
            raise NotImplementedError()   
            logit = None  # shape (1, 1)

            # Compute loss for one instance
            # TODO: Replace None with appropriate code to compute the loss 
            # for the i-th training instance, and delete the raise statement.
            raise NotImplementedError()
            loss = None  # shape (1,)

            loss_total += loss.item()

            # Backward pass and optimization step

            # TODO: Add a comment written entirely in your own words, explaining why the following line is needed.
            optimizer.zero_grad()

            # TODO: Add two lines of code needed to complete this iteration of optimization

        # average loss per instance
        loss_history.append(loss_total / num_instances)
    return loss_history


def validate_model(model: GymNeuron, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
    """Evaluate the model on the test set and return the error rate."""
    model.eval() # set model to evaluation mode
    # TODO: Add a comment written entirely in your own words, explaining why the following line is beneficial.
    with torch.no_grad():
        logits = model(X_test)  # shape (num_instances, 1)
        # shape (num_instances, 1)
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        correct = (predictions == y_test).float().sum().item()
        total = y_test.shape[0]
        error_rate = 1.0 - correct / total
    return error_rate


def train_and_summarize(
    model: GymNeuron,
    original_parameters: Parameters,
    num_epochs: int,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
) -> list[float]:
    loss_history = train_model(
        model,
        X_train,
        y_train,
        num_epochs,
    )

    print_final_summary(
        parameters=model_to_parameters(model),
        original_parameters=original_parameters,
        ground_truth=(
            InputInstance.gt_v_wt,
            InputInstance.gt_p_wt,
            InputInstance.gt_e_wt,
            InputInstance.gt_bias,
        )
    )
    return loss_history


def plot_loss_curve(loss_history: list[float]) -> None:
    """Plot average training loss versus epoch."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    epoch_history = list(range(1, len(loss_history) + 1))
    plt.plot(epoch_history, loss_history, linewidth=2, label="Training Loss")

    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Program entry point."""
    num_instances = 500
    num_epochs = 30
    use_deterministic_torch = True

    data_rng = seed_everything(SEED)
    print(f"seed: {SEED}")

    model = GymNeuron()

    test_set, training_set = create_train_test_sets(
        num_instances=num_instances,
        test_data_rng=data_rng,
        train_data_rng=data_rng,
    )

    X_train, y_train = instances_to_tensors(training_set)
    X_test, y_test = instances_to_tensors(test_set)

    print(f"Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")

    original_parameters = model_to_parameters(model)

    print("\nTraining with function: train_model (no batching, SGD)...")
    start_time = time.time()
    loss_history = train_and_summarize(
        model,
        original_parameters,
        num_epochs,
        X_train,
        y_train,
    )
    elapsed_time = time.time() - start_time
    test_error = validate_model(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"train_model: {elapsed_time:.3f} seconds, test error: {test_error:.3f}")
    print("=" * 60)

    plot_loss_curve(loss_history)


if __name__ == "__main__":
    main()
