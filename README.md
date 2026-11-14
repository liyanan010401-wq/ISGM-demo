# ISGM: Image-Supported Graph Memory — Research Demo

A small, CPU-runnable demonstration of the interfaces proposed in **Image-Supported Graph Memory for Chest X-Ray Report Generation**.

**Release scope:** this repository contains source code, configuration, unit tests, and self-generated geometric sample data. It is a mechanism demonstration, not a complete reproduction of the manuscript experiments. It contains no chest radiographs, real patient reports, clinical model weights, or clinical evaluation results. Generated text and errors are retained for inspection and have no clinical use.

## Quick start

Tested with Python 3.12, PyTorch 2.10.0 (CPU), and NumPy 2.3.5. A GPU is not required.

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.txt
python run_demo.py --output outputs
python -m unittest discover -s tests -v
```

Run these commands from the repository root. The demo also accepts an absolute script path from another working directory. Input resources are resolved relative to the script; an explicit relative `--output` path is resolved relative to the current working directory.

To regenerate the included synthetic fixture from the repository root:

```bash
python -m src.data
```

The generator seed is 4404; the model seed is 1404. CPU execution uses one thread. The development split is reserved but is not used to calibrate thresholds in this demo.

## What runs

1. **Stage A:** train tiny image/text alignment, masked assertion classification, a patient-excluding clause-matching surrogate, and an image-conditioned causal decoder.
2. **Stage B:** freeze the encoders and candidate cache; train a support/conflict/insufficient-evidence gate and the decoder using graph-retrieved clauses.
3. **Stage C:** generate a discrete draft, train a memory correction adapter and auxiliary relation heads through the frozen decoder, then produce a revised sequence.

The graph admits only the training split. It removes all records for the query's synthetic patient identifier before building the eligible retrieval transition. Test references are appended for evaluation only after inference returns. Inference retrieves once, generates twice, and does not update model parameters.

## Files and outputs

| Path | Contents |
| --- | --- |
| `src/core.py` | Source graph, PPR retrieval, gate, tiny model, and training/inference interfaces |
| `src/data.py` | Deterministic geometric sample generator |
| `data/sample.json` | 15 synthetic records: 10 train, 2 reserved development, 3 test |
| `config.json` | Seeds, dimensions, learning rate, and training budgets |
| `run_demo.py` | Three-stage training and two-pass inference |
| `tests/test_core.py` | Contract and edge-case tests |
| `examples/` | Unfiltered JSON outputs from the verified synthetic run |
| `METHOD_MAP.md` | Detailed manuscript-to-demo mapping and limitations (Chinese) |
| `VALIDATION.md` | Publication-run environment and verification results |
| `DATA_AND_CODE_AVAILABILITY.md` | Availability statement and original data-source links |

A run writes `predictions.json`, `metrics.json`, and `tiny_model.pt`. The checkpoint is generated locally and is not required to start the demo. `metrics.json` records training losses, adapter gradients, retrieval counts, frozen-test-state checks, and the data hash; it does **not** report RadGraph/CheXbert performance. Its elapsed time includes training and initialization and is not per-examination inference latency.

## Differences from the full proposed method

- Single 8×8 geometric inputs, 8-dimensional encoders, and a tiny GRU replace the full radiology backbones and multihead decoder. The gate has 90 input features rather than 1330.
- Facts are supplied by a synthetic schema; no RadGraph parser or official DART/HippoRAG model is reproduced.
- Gate thresholds are fixed demo values, not clinically calibrated values. The simplified weak-label helper can conflate cross-side presence/absence with a contradiction; it is unsuitable for real report supervision without redesign and adjudication.
- The fixture has 15 made-up patient identifiers and one examination per identifier. These are bookkeeping keys, not real people or evidence of medical generalization.
- No MIMIC-CXR/MS-CXR loader, full benchmark suite, trained clinical weights, expert review, or manuscript experimental results are included.

Read [the data card](data/DATA_CARD.md), [the method mapping](METHOD_MAP.md), and [the Chinese introduction](README_zh.md) before extending the demo.

