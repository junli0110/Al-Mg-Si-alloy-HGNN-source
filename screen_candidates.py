"""Pareto screening and composition-wise route comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from phase_path_hgnn import check_search_bounds

def pareto_mask(values):
    """Exact 2-objective maximization in O(n log n); preserve equal duplicates."""
    v=np.asarray(values,dtype=float)
    if v.ndim!=2 or v.shape[1]!=2 or not np.isfinite(v).all():
        raise ValueError('Pareto input must be a finite n x 2 matrix.')
    mask=np.zeros(len(v),dtype=bool)
    order=np.lexsort((-v[:,1],-v[:,0]));best=-np.inf;i=0
    while i<len(order):
        j=i+1
        while j<len(order) and v[order[j],0]==v[order[i],0]:j+=1
        group=order[i:j];top=v[group,1].max()
        if top>best:mask[group[v[group,1]==top]]=True
        best=max(best,top);i=j
    return mask

def merge_predictions(uts,el):
    for table in [uts,el]:
        if table.sample_no.isna().any() or table.sample_no.astype(str).duplicated().any():
            raise ValueError('Prediction sample_no values must be unique and nonempty.')
    if set(uts.sample_no.astype(str))!=set(el.sample_no.astype(str)):
        raise ValueError('UTS and EL candidate IDs differ.')
    uts=uts.copy();el=el.copy();uts['sample_no']=uts.sample_no.astype(str);el['sample_no']=el.sample_no.astype(str)
    el=el.set_index('sample_no').loc[uts.sample_no].reset_index()
    metadata=['route_label','composition_id','Mg','Si','Mn','Cu','Zn','Cr','Zr','Fe',
              'Al','Ti','V','Ni','Sc','Ag','Er','Y',
              'TEXT','vEXT','REXT','TSS','tSS','TAA','tAA',
              'EXS(m/min)','EXR','SS','SS-t','AA','AA-t']
    predictions=('mean_','std_','sigma_cal_','LCB_','member_')
    metadata=list(dict.fromkeys(metadata+[c for c in set(uts.columns)&set(el.columns)
        if not c.startswith(predictions) and c not in ['UTS','EL']]))
    for col in metadata:
        if (col in uts)!=(col in el):raise ValueError(f'Metadata missing from one prediction file: {col}')
        if col in uts:
            x,y=uts[col].reset_index(drop=True),el[col].reset_index(drop=True)
            if not x.fillna('<NA>').astype(str).eq(y.fillna('<NA>').astype(str)).all():
                raise ValueError(f'UTS/EL metadata mismatch: {col}')
    out=uts.copy()
    for col in el:
        if col.endswith('_EL'):out[col]=el[col].to_numpy()
    return out

def screen(df, enforce_bounds=True):
    df=df.copy().reset_index(drop=True)
    required=['sample_no','composition_id','route_label','mean_UTS','mean_EL','LCB_UTS','LCB_EL']
    if any(c not in df for c in required):raise ValueError(f'Required columns: {required}')
    if df.empty or df.composition_id.isna().any() or not df.route_label.isin(['T5','T6']).all():
        raise ValueError('Nonempty data, composition IDs and T5/T6 routes are required.')
    if not np.isfinite(df[['mean_UTS','mean_EL','LCB_UTS','LCB_EL']].to_numpy(dtype=float)).all():
        raise ValueError('Prediction objectives must be finite.')
    if df.sample_no.astype(str).duplicated().any():raise ValueError('Duplicate candidate IDs.')
    if enforce_bounds:check_search_bounds(df)
    chemistry=[c for c in ['Al','Mg','Si','Mn','Cu','Fe','Cr','Zr','Ti','Zn','V','Ni','Sc','Ag','Er','Y'] if c in df]
    if chemistry and (df.groupby('composition_id')[chemistry].nunique(dropna=False)>1).any().any():
        raise ValueError('One composition_id maps to different compositions.')
    manifest={'weights':{'UTS':.6,'EL':.4},'standardization':'all candidates within each route; population SD',
              'kappa':1.0,'route_preference':'sign of mean UTS difference at route-specific max S_LCB schedules',
              'z_parameters':{}}
    df['pareto']=False;df['S_LCB']=np.nan
    pooled = df[['LCB_UTS','LCB_EL']].to_numpy(dtype=float)
    pooled_mean = pooled.mean(axis=0)
    pooled_scale = np.where(pooled.std(axis=0)>0, pooled.std(axis=0), 1.)
    df['S_LCB_common'] = ((pooled-pooled_mean)/pooled_scale)@np.array([.6,.4])
    manifest['common_z_parameters'] = {'mean':pooled_mean.tolist(),'scale':pooled_scale.tolist()}
    for route, part in df.groupby('route_label'):
        values=part[['LCB_UTS','LCB_EL']].to_numpy(dtype=float)
        mean=values.mean(axis=0);scale=values.std(axis=0);scale=np.where(scale>0,scale,1.)
        df.loc[part.index,'S_LCB']=((values-mean)/scale)@np.array([.6,.4])
        df.loc[part.index,'pareto']=pareto_mask(values)
        manifest['z_parameters'][route]={'mean':mean.tolist(),'scale':scale.tolist()}
    best=df.sort_values(['S_LCB','sample_no'],ascending=[False,True]).drop_duplicates(['composition_id','route_label'])
    pairs=[]
    for composition,part in best.groupby('composition_id',sort=False):
        if set(part.route_label)!= {'T5','T6'}:raise ValueError(f'Both routes required: {composition}')
        a=part.set_index('route_label').loc['T5'];b=part.set_index('route_label').loc['T6']
        delta=float(a.mean_UTS-b.mean_UTS)
        delta_score=float(a.S_LCB_common-b.S_LCB_common)
        pairs.append({'composition_id':composition,'T5_sample_no':a.sample_no,'T6_sample_no':b.sample_no,
                      'delta_UTS_mean':delta,'delta_UTS_LCB':float(a.LCB_UTS-b.LCB_UTS),
                      'delta_EL_LCB':float(a.LCB_EL-b.LCB_EL),
                      'delta_S_LCB_common':delta_score,
                      'preference':'T5' if delta>0 else 'T6' if delta<0 else 'tie',
                      'composite_preference':'T5' if delta_score>0 else 'T6' if delta_score<0 else 'tie'})
    front=df.loc[df.pareto].sort_values(['route_label','S_LCB'],ascending=[True,False]).copy()
    front['rank_within_route']=front.groupby('route_label').cumcount()+1
    return df,front,pd.DataFrame(pairs),manifest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uts',type=Path,required=True);p.add_argument('--el',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/screening'))
    a=p.parse_args();df=merge_predictions(pd.read_csv(a.uts),pd.read_csv(a.el))
    all_rows,front,routes,manifest=screen(df)
    a.output.mkdir(parents=True,exist_ok=True)
    for name,table in [('all_candidates',all_rows),('pareto_fronts',front),('route_preferences',routes)]:
        table.to_csv(a.output/f'{name}.csv',index=False)
    (a.output/'screening_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':main()
