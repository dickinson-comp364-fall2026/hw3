"""Shared utilities for gym decision experiments.

This module contains framework-agnostic code that can be reused by multiple
implementations (for example, raw Python and PyTorch versions).
"""

from __future__ import annotations

import random
from typing import Iterable, List, Tuple

DEFAULT_SEED_CONFIG = {
    "train_data": 1234,
    "test_data": 5678,
    "parameter_init": 1357,
    "sgd_sampling": 2468,
}


class Parameters:
    """Store learnable parameter values for one gym-decision neuron.

    Example usage:
        rng = random.Random(1357)
        random_parameters = Parameters.random_init(rng)
        fixed_parameters = Parameters.from_values(
            v_wt=0.0,
            p_wt=0.0,
            e_wt=0.0,
            bias=0.0,
        )
    """

    def __init__(self, v_wt: float, p_wt: float, e_wt: float, bias: float) -> None:
        """Initialize model parameters from explicit weight and bias values."""
        self.v_wt = v_wt
        self.p_wt = p_wt
        self.e_wt = e_wt
        self.bias = bias

    @classmethod
    def random_init(cls, rng: random.Random) -> "Parameters":
        """Build parameters initialized randomly in [-1, 1]."""
        return cls(
            v_wt=rng.uniform(-1, 1),
            p_wt=rng.uniform(-1, 1),
            e_wt=rng.uniform(-1, 1),
            bias=rng.uniform(-1, 1),
        )

    @classmethod
    def from_values(cls, v_wt: float, p_wt: float, e_wt: float, bias: float) -> "Parameters":
        """Build parameters from provided values."""
        return cls(v_wt=v_wt, p_wt=p_wt, e_wt=e_wt, bias=bias)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        """Return rounded parameter values for readable logs."""
        return f"({self.v_wt:.3f}, {self.p_wt:.3f}, {self.e_wt:.3f}, {self.bias:.3f})"


class InputInstance:
    """Represent one example for the gym-decision task.

    Features:
    - v: hours already spent visiting gym
    - p: hours of race-preparation needed
    - e: energy level in [0, 1]
    """

    # Ground-truth linear rule used to generate labels.
    gt_v_wt = -4.0  # ground-truth weight for v feature
    gt_p_wt = 2.0  # ground-truth weight for p feature
    gt_e_wt = 3.0  # ground-truth weight for e feature
    gt_bias = -1.0  # ground-truth bias term

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        """Return a compact, rounded string for easy debug printing."""
        return f"({self.v:.3f}, {self.p:.3f}, {self.e:.3f})"

    def __init__(self, v, p, e) -> None:
        """Initialize one labeled input with the three task features."""
        # Hours already spent visiting the gym.
        self.v: float = v

        # Hours of preparation needed for an upcoming race.
        self.p: float = p

        # Current energy level in [0.0, 1.0].
        self.e: float = e

    def calc_target_value(self) -> float:
        """Compute the binary target label using the ground-truth rule."""
        val = (self.v * InputInstance.gt_v_wt +
               self.p * InputInstance.gt_p_wt +
               self.e * InputInstance.gt_e_wt +
               InputInstance.gt_bias)
        # The target is 1.0 (i.e. go to gym) if the linear combination is
        # closer to 1 than to 0, else 0.0 (i.e. don't go to gym).
        if abs(val - 1.0) < abs(val):
            return 1.0
        return 0.0
    


def resolve_seed_config(seed_config: dict[str, int] | None = None,
                        default_seed_config: dict[str, int] | None = None) -> dict[str, int]:
    """Return merged seed config where user values override defaults."""
    seeds = dict(default_seed_config or DEFAULT_SEED_CONFIG)
    if seed_config is not None:
        seeds.update(seed_config)
    return seeds


def create_random_generators(seeds: dict[str, int]) -> tuple[random.Random, random.Random, random.Random, random.Random]:
    """Build deterministic RNG objects for each experiment stage."""
    test_data_rng = random.Random(seeds["test_data"])
    train_data_rng = random.Random(seeds["train_data"])
    parameter_init_rng = random.Random(seeds["parameter_init"])
    sgd_sampling_rng = random.Random(seeds["sgd_sampling"])
    return test_data_rng, train_data_rng, parameter_init_rng, sgd_sampling_rng


def create_train_test_sets(num_instances: int,
                           test_data_rng: random.Random,
                           train_data_rng: random.Random) -> tuple[List[InputInstance], List[InputInstance]]:
    """Generate matching-sized synthetic train and test sets."""
    test_set = generate_numerical_data(num_instances=num_instances, rng=test_data_rng)
    training_set = generate_numerical_data(num_instances=num_instances, rng=train_data_rng)
    return test_set, training_set


def generate_numerical_data(num_instances: int = 1000,
                            rng: random.Random | None = None) -> List[InputInstance]:
    """Generate synthetic gym-decision data using simple non-uniform sampling."""
    instances: List[InputInstance] = []
    rng = rng or random.Random()

    # There is nothing special about the choice of parameters for this
    # synthetic data, other than producing a reasonably diverse set of
    # values to demonstrate the possible reasons for choosing to go to
    # the gym or stay home. We want some instances with zero hours at
    # the gym, some with zero prep needed, and some with zero or full
    # energy, but we also want a good number of instances with
    # non-zero values for all features to make the task non-trivial.
    for _ in range(num_instances):
        # 20% chance we haven't visited the gym, otherwise uniformly random
        # between zero and 5 hours.
        v = 0 if rng.random() < 0.2 else rng.uniform(0, 5)
        # 50% chance we need no extra prep, otherwise uniformly random
        # between zero and 4 hours.
        p = 0 if rng.random() < 0.5 else rng.uniform(0, 4)
        # 10% chance of no energy, 10% chance of full energy,
        # otherwise uniformly random in [0.0, 1.0].
        e = 0 if rng.random() < 0.1 else 1 if rng.random() < 0.1 else rng.uniform(0, 1)
        instances.append(InputInstance(v, p, e))

    return instances


def compute_metrics(parameters,
                    training_set: Iterable,
                    test_set: Iterable) -> tuple[float, float, float]:
    """Compute train loss, train error rate, and test error rate values."""
    train_loss = parameters.loss(training_set)
    training_error = parameters.error_rate(training_set)
    test_error = parameters.error_rate(test_set)
    return train_loss, training_error, test_error


def print_final_summary(parameters: Parameters,
                        original_parameters: Parameters,
                        ground_truth: Tuple[float, float, float, float]) -> None:
    """Print original/final parameters and normalized comparison to ground truth."""
    gt_v_wt, gt_p_wt, gt_e_wt, gt_bias = ground_truth

    print("original parameters", original_parameters)
    print("final parameters", parameters)
    if parameters.v_wt == 0.0:
        print("final parameters normalized: undefined because learned v_wt is 0.000")
    else:
        scale = gt_v_wt / parameters.v_wt
        normalized_bias = parameters.bias * scale
        print(
            f"final parameters normalized ({parameters.v_wt * scale:.3f}, {parameters.p_wt * scale:.3f}, {parameters.e_wt * scale:.3f}, {normalized_bias:.3f})")
    print(f"ground truth               ({gt_v_wt:.3f}, {gt_p_wt:.3f}, {gt_e_wt:.3f}, {gt_bias:.3f})")


def plot_loss_vs_iteration(iteration_history, loss_history) -> None:
    """Plot recorded loss values against iteration count."""
    # Import plotting backend only when needed to avoid test-time backend init.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(iteration_history, loss_history, color="tab:blue", linewidth=2)
    plt.title("Loss vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def find_first_misclassified_instance(parameters, instances):
    """Return the first misclassified instance, or None when all are correct."""
    for instance in instances:
        if parameters.classify(instance) != instance.calc_target_value():
            return instance
    return None
