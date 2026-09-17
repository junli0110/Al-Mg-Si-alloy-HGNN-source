"""Generate composition and process candidates for CALPHAD enrichment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from phase_path_hgnn import BOUNDS

def generate(n_compositions=1000,n_schedules=128,seed=3407):
    if n_compositions<1 or n_schedules<1:raise ValueError('Positive candidate counts required.')
    rng=np.random.default_rng(seed);rows=[]
    elements=['Mg','Si','Mn','Cu','Zn','Cr','Zr','Fe']
    for i in range(n_compositions):
        comp={c:round(float(rng.uniform(*BOUNDS[c])),4) for c in elements}
        for j in range(n_schedules):
            process={c:round(float(rng.uniform(*BOUNDS[c])),3) for c in ['TEXT','vEXT','REXT','TSS','tSS','TAA','tAA']}
            for route in ['T5','T6']:
                row={'sample_no':f'C{i:04d}_P{j:03d}_{route}','composition_id':f'C{i:04d}',
                     'schedule_pair':j,'route_label':route,**comp,**process}
                if route=='T5':row.update(TSS=np.nan,tSS=np.nan)
                rows.append(row)
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compositions',type=int,default=1000);p.add_argument('--schedules',type=int,default=128)
    p.add_argument('--seed',type=int,default=3407);p.add_argument('--output',type=Path,default=Path('outputs/candidates_for_calphad.csv'))
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    generate(a.compositions,a.schedules,a.seed).to_csv(a.output,index=False)
    print('Generated process/composition coordinates only. Calculate actual CALPHAD descriptors externally before prediction.')

if __name__=='__main__':main()
