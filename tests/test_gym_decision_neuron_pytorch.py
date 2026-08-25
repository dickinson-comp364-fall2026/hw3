import math
import random
import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
GYM_DECISION_DIR = ROOT / "gym_decision"

for path in (ROOT, GYM_DECISION_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from gym_decision.gym_decision_neuron_pytorch import (
    GymNeuron,
    instances_to_tensors,
    model_to_parameters,
    seed_everything,
    train_model,
    validate_model,
)
from gym_decision.gym_decision_shared import InputInstance


class GymDecisionPytorchTests(unittest.TestCase):
    def test_seed_everything_is_reproducible(self) -> None:
        rng_a = seed_everything(12345)
        sample_a = torch.randn(5)

        rng_b = seed_everything(12345)
        sample_b = torch.randn(5)

        self.assertEqual(rng_a.random(), rng_b.random())
        self.assertTrue(torch.equal(sample_a, sample_b))

    def test_instances_to_tensors_shapes_and_values(self) -> None:
        instances = [
            InputInstance(1.0, 2.0, 0.5),
            InputInstance(0.0, 1.0, 0.0),
        ]

        X, y = instances_to_tensors(instances)

        self.assertEqual(tuple(X.shape), (2, 3))
        self.assertEqual(tuple(y.shape), (2, 1))
        self.assertEqual(X.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)

        expected_X = torch.tensor([[1.0, 2.0, 0.5], [0.0, 1.0, 0.0]], dtype=torch.float32)
        expected_y = torch.tensor(
            [[instances[0].calc_target_value()], [instances[1].calc_target_value()]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(X, expected_X))
        self.assertTrue(torch.equal(y, expected_y))

    def test_train_model_is_reproducible_with_seeded_rng(self) -> None:
        training_instances = [
            InputInstance(0.0, 0.0, 0.0),
            InputInstance(1.0, 2.0, 0.5),
            InputInstance(2.0, 1.0, 1.0),
            InputInstance(0.2, 0.4, 0.8),
            InputInstance(3.0, 0.0, 0.2),
            InputInstance(0.0, 3.0, 0.9),
            InputInstance(1.5, 1.0, 0.1),
            InputInstance(0.5, 0.5, 0.5),
        ]
        X_train, y_train = instances_to_tensors(training_instances)

        seed_everything(2024)
        model_a = GymNeuron()
        history_a = train_model(
            model=model_a,
            X_train=X_train,
            y_train=y_train,
            num_epochs=4,
        )

        seed_everything(2024)
        model_b = GymNeuron()
        history_b = train_model(
            model=model_b,
            X_train=X_train,
            y_train=y_train,
            num_epochs=4,
        )

        self.assertEqual(len(history_a), 4)
        self.assertEqual(len(history_b), 4)
        for loss_a, loss_b in zip(history_a, history_b):
            self.assertTrue(math.isfinite(loss_a))
            self.assertTrue(math.isfinite(loss_b))
            self.assertAlmostEqual(loss_a, loss_b, places=7)

    def test_validate_model_returns_zero_when_predictions_match_labels(self) -> None:
        # Force predictions to all ones by making logits very positive.
        model = GymNeuron()
        with torch.no_grad():
            model.neuron.weight.fill_(0.0)
            model.neuron.bias.fill_(10.0)

        X_test = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0],
                [5.0, 1.0, 0.5],
            ],
            dtype=torch.float32,
        )
        y_test = torch.ones((3, 1), dtype=torch.float32)

        error_rate = validate_model(model, X_test, y_test)
        self.assertAlmostEqual(error_rate, 0.0)

    def test_model_to_parameters_extracts_expected_weights_and_bias(self) -> None:
        model = GymNeuron()
        with torch.no_grad():
            model.neuron.weight.copy_(torch.tensor([[1.25, -0.5, 3.75]], dtype=torch.float32))
            model.neuron.bias.copy_(torch.tensor([0.2], dtype=torch.float32))

        params = model_to_parameters(model)

        self.assertAlmostEqual(params.v_wt, 1.25, places=7)
        self.assertAlmostEqual(params.p_wt, -0.5, places=7)
        self.assertAlmostEqual(params.e_wt, 3.75, places=7)
        self.assertAlmostEqual(params.bias, 0.2, places=7)


if __name__ == "__main__":
    unittest.main()
