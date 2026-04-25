# Fashion-MNIST MLP From Scratch

This project implements the HW1 requirements using only `NumPy` for numerical computation. It includes:

- a custom array-based autograd engine
- a multi-layer perceptron with configurable hidden dimensions
- SGD, learning-rate decay, cross-entropy loss, and L2 regularization
- validation-based checkpoint saving
- hyper-parameter search
- test-set evaluation with confusion matrix
- report assets for training curves, first-layer weight visualization, and error analysis

## Submission Info

- Student: `谢唯`
- Student ID: `23307130044`
- GitHub Repo: `https://github.com/twothsnd/hw1_fashion_mnist`
- Model Weights: `https://pan.baidu.com/s/1lCKtDl2byjsPVCNrDmNHRA`
- Extraction Code: `zytd`

## Project Layout

```text
hw1_fashion_mnist/
├── artifacts/
├── data/
├── reports/
├── scripts/
│   ├── evaluate.py
│   ├── search.py
│   ├── smoke_test.py
│   └── train.py
└── src/
```

## Environment

```bash
python -m pip install -r requirements.txt
```

## Train A Baseline Model

```bash
python scripts/train.py \
  --output-dir artifacts/baseline \
  --hidden-dims 128 \
  --activation relu \
  --epochs 20 \
  --batch-size 128 \
  --learning-rate 0.1 \
  --lr-scheduler step \
  --lr-gamma 0.5 \
  --lr-step-size 5 \
  --weight-decay 0.0001
```

Artifacts written to the run directory:

- `best_model.npz`
- `config.json`
- `history.json`
- `summary.json`
- `training_curves.png`

## Run Hyper-Parameter Search

```bash
python scripts/search.py \
  --mode grid \
  --output-dir artifacts/search \
  --learning-rates 0.1,0.05,0.01 \
  --hidden-dim-grid "128;256;256,128" \
  --weight-decays 0.0001,0.0005,0.001 \
  --activations relu,tanh
```

The search script stores one artifact directory per trial and writes a top-level `search_results.csv`.

## Evaluate On The Test Set

```bash
python scripts/evaluate.py --artifact-dir artifacts/baseline
```

Evaluation outputs:

- `test_metrics.json`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `first_layer_weights.png`
- `error_examples.png`

## Smoke Test

The smoke test does not download Fashion-MNIST. It creates a synthetic IDX-format dataset, checks the cross-entropy gradient numerically, and verifies that the training loop can fit the data.

```bash
python scripts/smoke_test.py
```

## Notes For The Report

- `training_curves.png` contains the required train/validation loss curves and validation accuracy curve.
- `first_layer_weights.png` visualizes the first hidden layer weights reshaped to `28x28`.
- `error_examples.png` supports the required error analysis section.
- `confusion_matrix.csv` and `confusion_matrix.png` provide the test-set confusion matrix.
