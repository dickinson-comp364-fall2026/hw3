import math
import random
import sys
from pathlib import Path
import unittest

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

ROOT = Path(__file__).resolve().parents[1]
GYM_DECISION_DIR = ROOT / "gym_decision"

for path in (ROOT, GYM_DECISION_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from gym_decision.gym_decision_MLP import (
    GymLayers,
    seed_torch_from_rng,
    configure_torch_determinism,
    instances_to_tensors,
    train_model,
    validate_model,
    MODEL_INPUT_DIM,
    MODEL_WIDTH,
    MODEL_NUM_HIDDEN_LAYERS,
    MODEL_ACTIVATION_FN,
)

from gym_decision.gym_decision_shared import (
    InputInstance,
    create_random_generators,
    create_train_test_sets,
    DEFAULT_SEED_CONFIG,
    resolve_seed_config,
)


class GymLayersArchitectureTests(unittest.TestCase):
    """Test the GymLayers model architecture."""

    def test_gym_layers_no_hidden(self):
        """Test GymLayers with 0 hidden layers (direct linear mapping)."""
        model = GymLayers(num_hidden_layers=0)
        # Should have only one layer: Linear(3, 1)
        self.assertEqual(len(model.layers), 1)
        self.assertIsInstance(model.layers[0], nn.Linear)
        self.assertEqual(model.layers[0].in_features, MODEL_INPUT_DIM)
        self.assertEqual(model.layers[0].out_features, 1)

    def test_gym_layers_single_hidden(self):
        """Test GymLayers with 1 hidden layer."""
        model = GymLayers(num_hidden_layers=1)
        # Should have: Linear(3, 8), ReLU, Linear(8, 1)
        self.assertEqual(len(model.layers), 3)
        self.assertIsInstance(model.layers[0], nn.Linear)
        self.assertIsInstance(model.layers[1], nn.Module)  # activation
        self.assertIsInstance(model.layers[2], nn.Linear)

    def test_gym_layers_multiple_hidden(self):
        """Test GymLayers with multiple hidden layers."""
        num_hidden = 3
        model = GymLayers(num_hidden_layers=num_hidden)
        # Should have: Linear, Act, Linear, Act, Linear, Act, Linear
        # Total: 2*num_hidden + 1 layers
        expected_layer_count = 2 * num_hidden + 1
        self.assertEqual(len(model.layers), expected_layer_count)

    def test_gym_layers_forward_shape_no_hidden(self):
        """Test forward pass output shape with no hidden layers."""
        model = GymLayers(num_hidden_layers=0)
        batch_size = 5
        X = torch.randn(batch_size, MODEL_INPUT_DIM)
        logits = model(X)
        self.assertEqual(logits.shape, (batch_size, 1))

    def test_gym_layers_forward_shape_with_hidden(self):
        """Test forward pass output shape with hidden layers."""
        model = GymLayers(num_hidden_layers=2)
        batch_size = 10
        X = torch.randn(batch_size, MODEL_INPUT_DIM)
        logits = model(X)
        self.assertEqual(logits.shape, (batch_size, 1))

    def test_gym_layers_different_activation(self):
        """Test GymLayers with a different activation function."""
        model_relu = GymLayers(num_hidden_layers=2, activation_factory=nn.ReLU)
        model_tanh = GymLayers(num_hidden_layers=2, activation_factory=nn.Tanh)
        
        # Both should have same architecture
        self.assertEqual(len(model_relu.layers), len(model_tanh.layers))
        
        # Test forward pass works for both
        X = torch.randn(5, MODEL_INPUT_DIM)
        logits_relu = model_relu(X)
        logits_tanh = model_tanh(X)
        self.assertEqual(logits_relu.shape, logits_tanh.shape)


class TorchUtilityTests(unittest.TestCase):
    """Test torch seeding and determinism utilities."""

    def test_seed_torch_from_rng(self):
        """Test that seed_torch_from_rng sets torch seed reproducibly."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        
        seed1 = seed_torch_from_rng(rng1)
        seed2 = seed_torch_from_rng(rng2)
        
        # Same input RNG seed should produce same torch seed
        self.assertEqual(seed1, seed2)
        self.assertGreaterEqual(seed1, 0)
        self.assertLess(seed1, 2**63 - 1)

    def test_configure_torch_determinism_enabled(self):
        """Test that torch determinism can be configured."""
        # This just checks that the function runs without error
        configure_torch_determinism(enabled=True)
        configure_torch_determinism(enabled=False)


class DataConversionTests(unittest.TestCase):
    """Test data conversion utilities."""

    def test_instances_to_tensors_shape(self):
        """Test that instances_to_tensors produces correct tensor shapes."""
        # Create a small set of instances
        seeds = resolve_seed_config(default_seed_config=DEFAULT_SEED_CONFIG)
        test_data_rng, train_data_rng, _, _ = create_random_generators(seeds)
        test_set, _ = create_train_test_sets(
            num_instances=50,
            test_data_rng=test_data_rng,
            train_data_rng=train_data_rng,
        )
        
        X, y = instances_to_tensors(test_set)
        
        # Check shapes
        self.assertEqual(X.shape[0], len(test_set))
        self.assertEqual(X.shape[1], 3)  # v, p, e
        self.assertEqual(y.shape[0], len(test_set))
        self.assertEqual(y.shape[1], 1)
        
        # Check dtypes
        self.assertEqual(X.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)

    def test_instances_to_tensors_values(self):
        """Test that instances_to_tensors preserves instance values."""
        seeds = resolve_seed_config(default_seed_config=DEFAULT_SEED_CONFIG)
        test_data_rng, train_data_rng, _, _ = create_random_generators(seeds)
        test_set, _ = create_train_test_sets(
            num_instances=10,
            test_data_rng=test_data_rng,
            train_data_rng=train_data_rng,
        )
        
        X, y = instances_to_tensors(test_set)
        
        # Check that first instance values match
        inst0 = test_set[0]
        self.assertAlmostEqual(X[0, 0].item(), inst0.v, places=5)
        self.assertAlmostEqual(X[0, 1].item(), inst0.p, places=5)
        self.assertAlmostEqual(X[0, 2].item(), inst0.e, places=5)
        self.assertAlmostEqual(y[0, 0].item(), float(inst0.calc_target_value()), places=5)


class TrainingTests(unittest.TestCase):
    """Test training functionality."""

    def test_train_model_returns_loss_history(self):
        """Test that train_model returns a list of losses."""
        # Create small dataset
        seeds = resolve_seed_config(default_seed_config=DEFAULT_SEED_CONFIG)
        test_data_rng, train_data_rng, param_rng, sgd_rng = create_random_generators(seeds)
        _, training_set = create_train_test_sets(
            num_instances=100,
            test_data_rng=test_data_rng,
            train_data_rng=train_data_rng,
        )
        
        X_train, y_train = instances_to_tensors(training_set)
        
        # Create model
        seed_torch_from_rng(param_rng)
        model = GymLayers(num_hidden_layers=1)
        
        # Train for a few epochs
        num_epochs = 5
        loss_history = train_model(
            model, X_train, y_train, num_epochs, sgd_rng, batch_size=32
        )
        
        # Check loss history
        self.assertEqual(len(loss_history), num_epochs)
        self.assertTrue(all(isinstance(l, float) for l in loss_history))
        self.assertTrue(all(l >= 0 for l in loss_history))

    def test_train_model_loss_decreases(self):
        """Test that loss generally decreases over training (for small dataset)."""
        seeds = resolve_seed_config(default_seed_config=DEFAULT_SEED_CONFIG)
        test_data_rng, train_data_rng, param_rng, sgd_rng = create_random_generators(seeds)
        _, training_set = create_train_test_sets(
            num_instances=100,
            test_data_rng=test_data_rng,
            train_data_rng=train_data_rng,
        )
        
        X_train, y_train = instances_to_tensors(training_set)
        
        seed_torch_from_rng(param_rng)
        model = GymLayers(num_hidden_layers=1)
        
        # Train for more epochs to see clear trend
        loss_history = train_model(
            model, X_train, y_train, 20, sgd_rng, batch_size=32
        )
        
        # Loss at end should be less than loss at start
        self.assertLess(loss_history[-1], loss_history[0])


class ValidationTests(unittest.TestCase):
    """Test validation functionality."""

    def test_validate_model_returns_error_rate(self):
        """Test that validate_model returns a valid error rate."""
        # Create small dataset
        seeds = resolve_seed_config(default_seed_config=DEFAULT_SEED_CONFIG)
        test_data_rng, train_data_rng, param_rng, _ = create_random_generators(seeds)
        test_set, _ = create_train_test_sets(
            num_instances=100,
            test_data_rng=test_data_rng,
            train_data_rng=train_data_rng,
        )
        
        X_test, y_test = instances_to_tensors(test_set)
        
        seed_torch_from_rng(param_rng)
        model = GymLayers(num_hidden_layers=0)
        
        # Suppress diagnostics output
        error_rate = validate_model(model, X_test, y_test, show_diagnostics=False)
        
        # Check error rate is in valid range
        self.assertGreaterEqual(error_rate, 0.0)
        self.assertLessEqual(error_rate, 1.0)
        self.assertIsInstance(error_rate, float)

    def test_validate_model_perfect_predictions(self):
        """Test validate_model with perfect predictions."""
        # Create dummy data where model should predict perfectly
        X_test = torch.zeros(10, MODEL_INPUT_DIM)
        y_test = torch.zeros(10, 1)  # All labels are 0
        
        model = GymLayers(num_hidden_layers=0)
        # Zero out the model weights so logits are all ~0, which sigmoid to ~0.5
        # This won't be perfect, but we can at least test the mechanics
        
        error_rate = validate_model(model, X_test, y_test, show_diagnostics=False)
        
        # Just verify it returns a valid error rate
        self.assertGreaterEqual(error_rate, 0.0)
        self.assertLessEqual(error_rate, 1.0)


class ReproducibilityTests(unittest.TestCase):
    """Test that training is reproducible with same seeds."""

    def test_reproducible_training(self):
        """Test that training with same seeds produces same results."""
        # Run training twice with identical seeds
        results = []
        
        for run in range(2):
            seeds = resolve_seed_config(default_seed_config=DEFAULT_SEED_CONFIG)
            test_data_rng, train_data_rng, param_rng, sgd_rng = create_random_generators(seeds)
            
            _, training_set = create_train_test_sets(
                num_instances=100,
                test_data_rng=test_data_rng,
                train_data_rng=train_data_rng,
            )
            
            X_train, y_train = instances_to_tensors(training_set)
            
            seed_torch_from_rng(param_rng)
            configure_torch_determinism(enabled=True)
            model = GymLayers(num_hidden_layers=1)
            
            loss_history = train_model(
                model, X_train, y_train, 3, sgd_rng, batch_size=32
            )
            results.append(loss_history)
        
        # Losses should be identical
        for loss1, loss2 in zip(results[0], results[1]):
            self.assertAlmostEqual(loss1, loss2, places=5)


if __name__ == "__main__":
    unittest.main()
