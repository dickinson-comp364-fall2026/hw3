# PA3: Neural Networks

Programming assignment exploring multilayer neural networks.

_Everything below this point was AI-generated, based on some code that was written originally to perform experiments to be published in general audience book. The code is being repurposed for a programming assignment. Therefore many of the descriptions below describe things that were in the book experiments but will not be needed for the programming assignment. Nevertheless I am creating an initial README that will help to remind me of was in the original book-experiment version of this code._

## Problem

Given three numerical features:
- **v** — hours per week the gym was visited
- **e** — effort level (0–1)
- **w** — pounds overweight (0–4)

Classify whether the person should go to the gym (binary output: 0 or 1).

## Project Structure

| File | Description |
|------|-------------|
| `gymDecisionVanilla.py` | From-scratch implementation: single-neuron network with manual forward pass, gradient descent, and leaky ReLU — no frameworks |
| `main.py` | Keras-based experiments building up to the final multilayer model |

## Experiments (in `main.py`)

| Function | Architecture | Task |
|----------|-------------|------|
| `proof_of_concept` | 2 inputs → 1 node | Regression: learn `3x + 5y + 7` |
| `proof_of_concept_functional` | 2 inputs → 1 node | Same, using Keras functional API |
| `proof_of_concept_binary` | 2 inputs → 1 node | Binary: classify `3x + 5y > 4` |
| `gym_dec_3input_1node` | 3 inputs → 1 node | Gym decision — regression |
| `gym_dec_3input_1node_binary` | 3 inputs → 1 node | Gym decision — binary classifier |
| `gym_dec_3input_2node` | 3 inputs → 2 nodes → 1 | Attempts to mimic the decision-tree structure (V separate from W/E) |
| `gym_dec_3input_2node_binary` | 3 inputs → 2 nodes → 1 | Same, binary classifier |
| `gym_dec_3input_3layers_fullyconn` | 3 → 3 (hidden) → 1 | **Final model**: fully-connected 3-layer binary classifier |

The final `gym_dec_3input_3layers_fullyconn` model is called from `main()` and is the one used in the textbook figure.


## Usage

```bash
python main.py
```

To run a different experiment, uncomment the corresponding call in `main()` at the bottom of `main.py`.

## Notes

- Training data is generated procedurally by `gymDecisionVanilla.generate_numerical_data()` with a fixed random seed for reproducibility.
- The "mimic decision tree" experiments (`gym_dec_3input_2node*`) did not converge reliably and are not used in the final figures.

## Useful commands

To run tests from the root directory:

```bash
py -m unittest discover -s tests -p "test_*.py"
```

