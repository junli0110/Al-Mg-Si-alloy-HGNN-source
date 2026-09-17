"""Ensemble prediction and lower-bound scoring."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from phase_path_hgnn import load_phase_checkpoint, prepare_phase_dataset, predict_checkpoint
from phase_path_hgnn import check_search_bounds

def ensemble_predict(checkpoints, data, element_properties, calibration, enforce_bounds=False):
    paths = sorted(map(Path, checkpoints))
    if len(paths) < 2: raise ValueError('At least two checkpoints are required for ensemble dispersion.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, schema, _ = load_phase_checkpoint(paths[0], device)
    df, _, _, graphs = prepare_phase_dataset(Path(data), Path(element_properties), schema.target,
        schema.element_property_names, phase_attributes=schema.phase_attributes,
        phase_threshold=schema.phase_threshold, require_target=False, ablation=schema.ablation)
    if enforce_bounds: check_search_bounds(df)
    predictions=[]
    for path in paths:
        _, _, other, _ = load_phase_checkpoint(path, device)
        if asdict(other) != asdict(schema): raise ValueError('Ensemble checkpoint schemas differ.')
        predictions.append(predict_checkpoint(path, graphs, device=device))
    cal = json.loads(Path(calibration).read_text(encoding='utf-8'))
    if cal['target'] != schema.target or cal['kappa'] != 1.0:
        raise ValueError('Calibration target or kappa does not match the model.')
    residual_scale = float(cal['residual_rmse'])
    if not np.isfinite(residual_scale) or residual_scale < 0: raise ValueError('Invalid residual RMSE.')
    pred = np.array(predictions)
    result = df.copy();target=schema.target
    result[f'mean_{target}'] = pred.mean(axis=0)
    result[f'std_{target}'] = pred.std(axis=0, ddof=1)
    result[f'sigma_cal_{target}'] = np.hypot(result[f'std_{target}'], residual_scale)
    result[f'LCB_{target}'] = result[f'mean_{target}']-result[f'sigma_cal_{target}']
    for i, member in enumerate(pred): result[f'member_{i+1:02d}_{target}'] = member
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ensemble-dir',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--element-properties',type=Path,required=True)
    p.add_argument('--enforce-search-bounds',action='store_true')
    p.add_argument('--output',type=Path,default=Path('outputs/predictions.csv'))
    a=p.parse_args();torch.set_num_threads(1)
    result=ensemble_predict(a.ensemble_dir.glob('ensemble_*.pt'),a.data,a.element_properties,
                           a.calibration,a.enforce_search_bounds)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.output,index=False)
    print(f'Local predictions: {a.output}')

if __name__=='__main__':main()
