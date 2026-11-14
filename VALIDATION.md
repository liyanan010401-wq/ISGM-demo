# Verified synthetic demonstration

Publication check: 2026-11-14 (Asia/Shanghai).

Environment: Python 3.12.14, PyTorch 2.10.0+cpu, NumPy 2.3.5, Windows, CPU single-thread execution. The dependency versions are pinned in requirements.txt.

Commands executed successfully:

    python -m unittest discover -s tests -v
    python run_demo.py --output outputs/publication_validation

The unit-test command was run from the release root; the full demo was also started by absolute script path from an external working directory.

- 17 unit tests passed.
- Stages A/B/C completed 5/3/2 epochs.
- All three held-out synthetic records were processed, with three retrieval calls in total.
- Model parameters remained unchanged during test inference.
- The correction adapter received nonzero gradients through the frozen decoder.
- data/sample.json was byte-identical to a fresh seed-4404 generator result.
- Predictions exactly matched the previously verified local run; the timing field is machine-dependent.
- Unfiltered outputs from this check are in examples/predictions.json and examples/metrics.json.

The public copy normalizes the fixture to LF and makes the generator explicitly write LF newlines. This serialization-only change preserves all sample values and model predictions. The original local manuscript and demo are retained separately.

The tests cover training-only memory, patient exclusion before PPR normalization, conflict priority and composite candidates, projected-value attenuation and zero-key masking, frozen-decoder gradients, two-pass inference, complete-clause preservation, ambiguous relation attributes, unmentioned states, stale caches, and Stage A agreement/empty-set cases.

No real clinical data were loaded or evaluated. These checks establish that the tiny demonstration runs and satisfies the tested software contracts. They do not validate real-world medical semantics, clinical performance, full paper reproduction, or the simplified weak-label helper.
