import math
import sys
from pathlib import Path
import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

ROOT = Path(__file__).resolve().parents[1]
GYM_DECISION_DIR = ROOT / "gym_decision"

for path in (ROOT, GYM_DECISION_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from gym_decision.gym_decision_neuron_raw import SingleNeuronRaw

from gym_decision.gym_decision_shared import (
    DEFAULT_SEED_CONFIG,
    InputInstance,
    Parameters,
    create_random_generators,
    create_train_test_sets,
    resolve_seed_config,
)


class GymDecisionDataSetupTests(unittest.TestCase):
    def test_create_train_test_sets_is_deterministic(self) -> None:
        seeds = resolve_seed_config(default_seed_config=DEFAULT_SEED_CONFIG)

        first_test_rng, first_train_rng, _, _ = create_random_generators(seeds)
        second_test_rng, second_train_rng, _, _ = create_random_generators(seeds)

        first_test_set, first_training_set = create_train_test_sets(
            num_instances=7,
            test_data_rng=first_test_rng,
            train_data_rng=first_train_rng,
        )
        second_test_set, second_training_set = create_train_test_sets(
            num_instances=7,
            test_data_rng=second_test_rng,
            train_data_rng=second_train_rng,
        )

        self.assertEqual(self._serialize_instances(first_test_set), self._serialize_instances(second_test_set))
        self.assertEqual(self._serialize_instances(first_training_set), self._serialize_instances(second_training_set))
        self.assertEqual(len(first_test_set), 7)
        self.assertEqual(len(first_training_set), 7)

    def _serialize_instances(self, instances):
        return [(instance.v, instance.p, instance.e) for instance in instances]


class GymDecisionBceMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SingleNeuronRaw(Parameters.from_values(
            v_wt=0.0,
            p_wt=0.0,
            e_wt=0.0,
            bias=0.0,
        ))

    def test_sigmoid_handles_large_magnitudes(self) -> None:
        self.assertAlmostEqual(self.model.sigmoid(0.0), 0.5)
        self.assertAlmostEqual(self.model.sigmoid(1000.0), 1.0)
        self.assertGreater(self.model.sigmoid(-100.0), 0.0)
        self.assertLess(self.model.sigmoid(-100.0), 1e-40)

    def test_prediction_error_and_loss_at_zero_logit(self) -> None:
        instance = InputInstance(0.0, 0.0, 0.0)

        self.assertEqual(instance.calc_target_value(), 0.0)
        self.assertAlmostEqual(self.model.predict_probability(instance), 0.5)
        self.assertAlmostEqual(self.model.prediction_error(instance), 0.5)
        self.assertAlmostEqual(self.model.loss_single_instance(instance), math.log(2.0))

    def test_single_training_step_updates_parameters(self) -> None:
        instance = InputInstance(0.0, 1.0, 0.0)

        self.assertEqual(instance.calc_target_value(), 1.0)
        self.model.train_single_instance(instance)

        self.assertAlmostEqual(self.model.parameters.v_wt, 0.0)
        self.assertAlmostEqual(self.model.parameters.p_wt, 0.001)
        self.assertAlmostEqual(self.model.parameters.e_wt, 0.0)
        self.assertAlmostEqual(self.model.parameters.bias, 0.001)


if __name__ == "__main__":
    unittest.main()
