"""Train a single-neuron gym decision model with sigmoid and binary cross-entropy from scratch.

The implementation is "raw" because we use raw Python rather than PyTorch.
The training loop and gradients are thus fully visible. This assists 
with understanding how an artificial neuron really works.
"""

import copy
import math
from typing import List
from gym_decision_shared import (
    DEFAULT_SEED_CONFIG,
    InputInstance,
    Parameters,
    compute_metrics,
    create_random_generators,
    create_train_test_sets,
    find_first_misclassified_instance,
    plot_loss_vs_iteration,
    print_final_summary,
    resolve_seed_config,
)

class SingleNeuronRaw:
    """Single-neuron sigmoid + BCE learner operating on shared Parameters."""

    # Hyperparameters used by SGD and BCE calculations.
    learning_rate = 0.002
    classification_threshold = 0.5
    loss_epsilon = 1e-12

    def __init__(self, parameters: Parameters) -> None:
        self.parameters = parameters

    def linear_output(self, instance: InputInstance) -> float:
        """Return the pre-sigmoid score (a.k.a. logit): w.x + b."""
        return self.parameters.v_wt * instance.v + \
            self.parameters.p_wt * instance.p + \
            self.parameters.e_wt * instance.e + \
            self.parameters.bias

    def sigmoid(self, logit: float) -> float:
        """Compute sigmoid of the logit value with a numerically stable branch."""
        if logit >= 0.0:
            return 1.0 / (1.0 + math.exp(-logit))
        # For large negative logits, this branch avoids overflow in exp(-logit).
        exp_logit = math.exp(logit)
        # TODO: uncomment and fill in question marks to complete the numerically stable sigmoid calculation
        # return exp_logit / (???????????????)
        raise NotImplementedError()

    def predict_probability(self, instance: InputInstance) -> float:
        """Predict P(y=1|x) for one instance."""
        logit = self.linear_output(instance)
        return self.sigmoid(logit)

    def classify(self, instance: InputInstance) -> float:
        """Convert probability into a hard class label using the classification 
        threshold."""
        predicted_probability = self.predict_probability(instance)
        if predicted_probability > SingleNeuronRaw.classification_threshold:
            return 1.0
        return 0.0

    def error_rate(self, instances: List[InputInstance]) -> float:
        """Return fraction of incorrectly classified instances."""
        assert len(instances) > 0
        num_correct = sum(
            1 for instance in instances
            if self.classify(instance) == instance.calc_target_value()
        )
        return 1.0 - num_correct / len(instances)

    def prediction_error(self, instance: InputInstance) -> float:
        """Return dL/dlogit for sigmoid + BCE: predicted_probability - target."""
        # TODO: fill in missing code.
        raise NotImplementedError()
        return -1 # placeholder to make the function syntactically valid
        

    def train_single_instance(self, instance: InputInstance) -> None:
        """Apply one SGD update using a single training instance."""
        # TODO: fill in missing code.
        raise NotImplementedError()

    def train_instances(self, instances: List[InputInstance]) -> None:
        """Train once over all provided instances in sequence."""
        for instance in instances:
            self.train_single_instance(instance)

    def loss_single_instance(self, instance: InputInstance) -> float:
        """Compute binary cross-entropy loss for one instance."""
        target = instance.calc_target_value()
        predicted_probability = self.predict_probability(instance)
        # Clip to avoid log(0) when probabilities saturate at 0 or 1.
        clipped_probability = min(max(
            predicted_probability, SingleNeuronRaw.loss_epsilon), 1.0 - SingleNeuronRaw.loss_epsilon)
        # TODO: fill in missing code in return statement
        raise NotImplementedError()
        return -1.0 # placeholder to make the function syntactically valid

    def loss(self, instances: List[InputInstance]) -> float:
        """Return the total BCE loss across a set of instances."""
        return sum(self.loss_single_instance(instance) for instance in instances)


def run_numerical_experiment(seed_config: dict[str, int] | None = None):
    """Run training and print periodic metrics plus a final parameter summary."""
    num_training_steps = 1_000_000
    metrics_interval = 10_000
    print_interval = 100_000
    num_instances = 1000

    seeds = resolve_seed_config(
        seed_config=seed_config,
        default_seed_config=DEFAULT_SEED_CONFIG,
    )

    test_data_rng, train_data_rng, parameter_init_rng, sgd_sampling_rng = create_random_generators(
        seeds)

    print("seed config", seeds)

    test_set, training_set = create_train_test_sets(
        num_instances=num_instances,
        test_data_rng=test_data_rng,
        train_data_rng=train_data_rng,
    )

    parameters = Parameters.random_init(parameter_init_rng)
    model = SingleNeuronRaw(parameters)
    original_parameters = copy.copy(parameters)
    iteration_history: List[int] = []
    loss_history: List[float] = []

    print(f"{'step':>8} {'loss':>10} {'train_err':>10} {'test_err':>10}")
    print(f"{'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10}")
    for i in range(1, num_training_steps+1):
        random_instance = sgd_sampling_rng.choice(training_set)
        model.train_single_instance(random_instance)

        first_or_last_step = (i == 1 or i == num_training_steps)
        should_capture_metrics = (
            i % metrics_interval == 0) or first_or_last_step
        if should_capture_metrics:
            train_loss, training_error, test_error = compute_metrics(
                model, training_set, test_set)
            iteration_history.append(i)
            loss_history.append(train_loss)

            should_print_row = (i % print_interval == 0) or first_or_last_step
            if should_print_row:
                print(
                    f"{i:8d} {train_loss:10.3f} {training_error:10.3f} {test_error:10.3f}")

    print_final_summary(
        model.parameters,
        original_parameters,
        ground_truth=(
            InputInstance.gt_v_wt,
            InputInstance.gt_p_wt,
            InputInstance.gt_e_wt,
            InputInstance.gt_bias,
        ),
    )
    # Uncomment one or both of following lines to see more details about a misclassification and the loss curve.
    # print_one_misclassification_details(parameters, test_set)
    # plot_loss_vs_iteration(iteration_history, loss_history)


def print_instance_loss_breakdown(model: SingleNeuronRaw, instance: InputInstance) -> None:
    """Print a step-by-step prediction and BCE loss breakdown for one instance."""
    target = instance.calc_target_value()
    predicted = model.classify(instance)
    a_val = model.linear_output(instance)
    predicted_probability = model.predict_probability(instance)
    err = model.prediction_error(instance)
    loss_val = model.loss_single_instance(instance)

    v_term = model.parameters.v_wt * instance.v
    p_term = model.parameters.p_wt * instance.p
    e_term = model.parameters.e_wt * instance.e

    print("misclassified instance details")
    print(
        f"  instance (v, p, e): ({instance.v:.3f}, {instance.p:.3f}, {instance.e:.3f})")
    print(f"  target: {target:.3f} predicted: {predicted:.3f}")
    print("  pre-activation a = v_wt*v + p_wt*p + e_wt*e + bias")
    print(
        f"                 = {v_term:.3f} + {p_term:.3f} + {e_term:.3f} + {model.parameters.bias:.3f}")
    print(f"                 = {a_val:.3f}")
    print(f"  activation output = sigmoid(a) = {predicted_probability:.3f}")
    print(f"  decision threshold: 0.500 -> predicted {predicted:.3f}")
    print(
        f"  BCE gradient term (output - target): {predicted_probability:.3f} - {target:.3f} = {err:.3f}")
    print("  binary cross-entropy loss = -(target * log(output) + (1-target) * log(1-output))")
    print(
        f"                            = -({target:.3f} * log({predicted_probability:.3f}) + {(1.0 - target):.3f} * log({1.0 - predicted_probability:.3f}))")
    print(f"                            = {loss_val:.3f}")
    print("  note: for a misclassification near the 0.500 threshold, BCE loss is often near 0.693 because -log(0.5) is about 0.693.")


def print_one_misclassification_details(model: SingleNeuronRaw, instances: List[InputInstance]) -> None:
    """Find one misclassified instance and print its detailed BCE breakdown."""
    misclassified_instance = find_first_misclassified_instance(
        model, instances)

    if misclassified_instance is None:
        print("No misclassified instance found in the provided set.")
        return

    print_instance_loss_breakdown(model, misclassified_instance)


def main():
    """Program entry point."""
    run_numerical_experiment()


if __name__ == "__main__":
    main()
