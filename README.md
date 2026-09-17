# Al-Mg-Si alloy HGNN

Core reference implementation of an element-phase-process HGNN for UTS/EL
prediction and T5/T6 route screening. Training data and trained weights are not included.


## Installation

Python 3.12:

```bash
python -m pip install -r requirements.txt
```

## Training

```bash
python phase_path_hgnn.py --target all --data private_data/training.csv --element-properties private_data/element_properties.csv --output outputs
```

Grouped five-fold validation is the default. `source_id` identifies a paper,
melt or production batch. Selected descriptors and search bounds are defined
in `phase_path_hgnn.py`.

## Prediction and screening

Generate candidates, then add phase descriptors calculated using external CALPHAD software:

```bash
python generate_candidates.py --output outputs/candidates_for_calphad.csv
python predict.py --ensemble-dir outputs/group/full/UTS/ensemble --calibration outputs/group/full/UTS/cross_validation/lcb_calibration.json --data private_data/enriched_candidates.csv --element-properties private_data/element_properties.csv --enforce-search-bounds --output outputs/uts_predictions.csv
```

Repeat prediction with the EL ensemble and calibration, saving `outputs/el_predictions.csv`, then run:

```bash
python screen_candidates.py --uts outputs/uts_predictions.csv --el outputs/el_predictions.csv --output outputs/screening
```

`benchmark_models.py` compares tabular models. `explain_model.py` provides
descriptor attribution and node/edge removal analyses. Use `--help` for each script's arguments.
