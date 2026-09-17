"""Node and edge removal analyses with permutation-based Shapley estimates."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from phase_path_hgnn import (load_phase_checkpoint,prepare_phase_dataset,to_heterodata,
                            _evaluate,PHASE_TYPES,STAGE_TYPES,PHASE_DESCRIPTORS)
from torch_geometric.loader import DataLoader

def evaluator(model,preprocessor,device):
    def predict(graphs):
        data=[to_heterodata(g,preprocessor,i) for i,g in enumerate(graphs)]
        scaled,_,indices=_evaluate(model,DataLoader(data,batch_size=128,shuffle=False),device,with_truth=False)
        out=np.empty(len(graphs));out[indices]=preprocessor.target.inverse_transform(scaled).ravel()
        return out
    return predict

def remove_node(graph,kind,index):
    g=copy.deepcopy(graph)
    matrix=getattr(g,f'{kind}_x')
    if len(matrix)<=1:raise ValueError('Cannot evaluate an empty node type with this readout.')
    setattr(g,f'{kind}_x',np.delete(matrix,index,axis=0))
    getattr(g,f'{kind}_names').pop(index)
    if kind=='phase':g.phase_stages.pop(index)
    for relation,edges in g.edges.items():
        keep=np.ones(edges.shape[1],dtype=bool)
        if relation[0]==kind:keep &= edges[0]!=index
        if relation[2]==kind:keep &= edges[1]!=index
        edges=edges[:,keep].copy()
        if relation[0]==kind:edges[0]-=(edges[0]>index)
        if relation[2]==kind:edges[1]-=(edges[1]>index)
        g.edges[relation]=edges
    return g

def removal_attribution(graph,predict):
    original=float(predict([graph])[0]);rows=[];changed=[]
    for kind in ['element','phase','stage']:
        names=getattr(graph,f'{kind}_names')
        for i,name in enumerate(names):
            if len(names)<=1:continue
            label=f'{graph.phase_stages[i]}@{name}' if kind=='phase' else name
            rows.append({'kind':'node','component':f'{kind}:{label}'})
            changed.append(remove_node(graph,kind,i))
    for relation,edges in graph.edges.items():
        for i,(source,target) in enumerate(edges.T):
            g=copy.deepcopy(graph);g.edges[relation]=np.delete(edges,i,axis=1);changed.append(g)
            rows.append({'kind':'edge','component':f'{relation}:{source}->{target}'})
    values=predict(changed)
    for row,value in zip(rows,values):
        row.update(original_prediction=original,removed_prediction=float(value),signed_change=original-float(value),
                   importance=abs(original-float(value)))
    return pd.DataFrame(rows)

def feature_groups(graph,schema,background):
    groups=[];offset=len(PHASE_TYPES)+len(STAGE_TYPES)
    for j,name in enumerate(schema.element_property_names):
        col=2+j;value=float(np.median(np.concatenate([g.element_x[:,col] for g in background])))
        groups.append((name,'element',list(range(len(graph.element_x))),col,value))
    for j,name in enumerate(schema.phase_attributes):
        stage,phase,_=PHASE_DESCRIPTORS[name]
        indices=[i for i,pair in enumerate(zip(graph.phase_stages,graph.phase_names)) if pair==(stage,phase)]
        if not indices:continue
        vals=[g.phase_x[i,offset+j] for g in background for i,pair in enumerate(zip(g.phase_stages,g.phase_names)) if pair==(stage,phase)]
        if not vals:raise ValueError(f'Background lacks {stage}/{phase}')
        groups.append((name,'phase',indices,offset+j,float(np.median(vals))))
    specs=[('TEXT','EXT',3),('vEXT','EXT',4),('REXT','EXT',5),('TSS','SS',3),('tSS','SS',6),('TAA','AA',3),('tAA','AA',7)]
    for name,stage,col in specs:
        if stage not in graph.stage_names:continue
        vals=[g.stage_x[g.stage_names.index(stage),col] for g in background if stage in g.stage_names]
        if not vals:raise ValueError(f'Background lacks stage {stage}')
        groups.append((name,'stage',[graph.stage_names.index(stage)],col,float(np.median(vals))))
    return groups

def permutation_shapley(graph,schema,background,predict,n_permutations=64,seed=3407):
    if n_permutations<2:raise ValueError('Use at least two permutations to estimate Monte Carlo error.')
    groups=feature_groups(graph,schema,background)
    baseline=copy.deepcopy(graph)
    for _,kind,indices,col,value in groups:getattr(baseline,f'{kind}_x')[indices,col]=value
    baseline_value,original=predict([baseline,graph]);rng=np.random.default_rng(seed)
    contributions=np.zeros((n_permutations,len(groups)))
    for repeat in range(n_permutations):
        order=rng.permutation(len(groups));current=copy.deepcopy(baseline);states=[]
        for k in order:
            _,kind,indices,col,_=groups[k]
            getattr(current,f'{kind}_x')[indices,col]=getattr(graph,f'{kind}_x')[indices,col]
            states.append(copy.deepcopy(current))
        values=np.r_[baseline_value,predict(states)]
        contributions[repeat,order]=np.diff(values)
    means=contributions.mean(axis=0)
    if not np.isclose(means.sum(),original-baseline_value,atol=1e-3):raise RuntimeError('Shapley additivity check failed.')
    return pd.DataFrame({'feature':[g[0] for g in groups],'shapley':means,
        'monte_carlo_se':contributions.std(axis=0,ddof=1)/np.sqrt(n_permutations),
        'baseline_prediction':baseline_value,'prediction':original})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--background-data',type=Path,required=True);p.add_argument('--element-properties',type=Path,required=True)
    p.add_argument('--mode',choices=['removal','shapley'],default='removal')
    p.add_argument('--sample-no',nargs='*',help='Omit to explain all supplied records.')
    p.add_argument('--permutations',type=int,default=64);p.add_argument('--output',type=Path,default=Path('outputs/interpretation'))
    a=p.parse_args();torch.set_num_threads(1);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,pre,schema,_=load_phase_checkpoint(a.checkpoint,device)
    if schema.ablation!='full':raise ValueError('Interpretation CLI currently requires a full-model checkpoint.')
    def load(path):
        return prepare_phase_dataset(path,a.element_properties,schema.target,schema.element_property_names,
            phase_attributes=schema.phase_attributes,require_target=False)[3]
    graphs=load(a.data);background=load(a.background_data);predict=evaluator(model,pre,device)
    if a.sample_no:
        if set(a.sample_no)-{g.sample_no for g in graphs}:raise ValueError('Unknown requested sample_no.')
        graphs=[g for g in graphs if g.sample_no in a.sample_no]
    rows=[]
    for graph in graphs:
        result=removal_attribution(graph,predict) if a.mode=='removal' else permutation_shapley(graph,schema,background,predict,a.permutations)
        result.insert(0,'sample_no',graph.sample_no);result.insert(1,'route',graph.route);rows.append(result)
    result=pd.concat(rows,ignore_index=True);a.output.mkdir(parents=True,exist_ok=True)
    result.to_csv(a.output/f'{schema.target}_{a.mode}.csv',index=False)
    if a.mode=='shapley':
        summary=result.assign(abs_shapley=result.shapley.abs()).groupby(['route','feature']).agg(
            mean_signed=('shapley','mean'),mean_absolute=('abs_shapley','mean'),n=('shapley','size'))
        summary.to_csv(a.output/f'{schema.target}_shapley_summary.csv')
    (a.output/'interpretation_manifest.json').write_text(json.dumps({
        'mode':a.mode,'n_explained':len(graphs),'background_records':len(background),
        'baseline':'Median feature values from explicit background; topology and composition held fixed.',
        'interpretation':'Model attribution, not causal evidence; single directed edges removed individually.'},indent=2),encoding='utf-8')

if __name__=='__main__':main()
