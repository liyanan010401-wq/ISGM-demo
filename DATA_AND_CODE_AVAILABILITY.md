# Data and Code Availability

The ISGM demonstration code, configuration files, unit tests, and synthetic example data are publicly available at https://github.com/liyanan010401-wq/ISGM-demo. The repository provides a runnable demonstration of the proposed method interfaces; it does not include the full experimental implementation, trained clinical model weights, or real patient data. The synthetic examples are provided solely for demonstration and are not evidence of clinical performance.

## Data included in this release

`data/sample.json` contains self-generated 8×8 geometric arrays and rule-based fact strings. All patient and examination identifiers are invented. The generator is `src/data.py`; no external data download or account is needed to run the demo. See [the data card](data/DATA_CARD.md).

## Original resources for the proposed clinical evaluation

The following resources are **not used by the included synthetic demo and are not redistributed here**:

- [MIMIC-CXR-JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- [MS-CXR v1.1.0](https://physionet.org/content/ms-cxr/1.1.0/)

Both are distributed by PhysioNet under credentialed-access conditions. Researchers must obtain access directly from PhysioNet and comply with the applicable training, data-use agreements, and usage terms. Access to this repository does not grant access to those datasets. This release makes no claim that the manuscript's real-data experiments have been completed.

