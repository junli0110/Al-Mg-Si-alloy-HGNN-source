"""Tabular benchmarks with the HGNN's saved outer folds and equivalent inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV,KFold,GroupKFold
from xgboost import XGBRegressor
from phase_path_hgnn import (ALL_ELEMENTS, ELEMENT_FEATURES, PHASE_TYPES, STAGE_TYPES,
    leakage_groups, prepare_phase_dataset, regression_metrics, validate_group_metadata)

def flatten(graph):
    """Map graph features to fixed tabular slots."""
    element=np.zeros((len(ALL_ELEMENTS),graph.element_x.shape[1]+1))
    for i,name in enumerate(graph.element_names):element[ALL_ELEMENTS.index(name)]=np.r_[graph.element_x[i],1]
    phase=np.zeros((len(STAGE_TYPES)*len(PHASE_TYPES),graph.phase_x.shape[1]+1))
    for i,(stage,name) in enumerate(zip(graph.phase_stages,graph.phase_names)):
        phase[STAGE_TYPES.index(stage)*len(PHASE_TYPES)+PHASE_TYPES.index(name)]=np.r_[graph.phase_x[i],1]
    stage=np.zeros((len(STAGE_TYPES),graph.stage_x.shape[1]+1))
    for i,name in enumerate(graph.stage_names):stage[STAGE_TYPES.index(name)]=np.r_[graph.stage_x[i],1]
    return np.r_[element.ravel(),phase.ravel(),stage.ravel()]

def estimators(seed):
    return {
        'XGBoost':(XGBRegressor(n_estimators=200,max_depth=3,learning_rate=.05,random_state=seed,n_jobs=1),{'max_depth':[2,4],'n_estimators':[100,300]}),
        'SVR':(SVR(),{'C':[1.,10.,100.],'gamma':['scale',.01]}),
        'GBR':(GradientBoostingRegressor(random_state=seed),{'n_estimators':[100,300],'max_depth':[2,3]}),
        'RFR':(RandomForestRegressor(n_estimators=200,random_state=seed,n_jobs=1),{'max_features':[.5,1.],'min_samples_leaf':[1,3]}),
        'BPNN':(MLPRegressor(hidden_layer_sizes=(128,32),max_iter=500,early_stopping=False,random_state=seed),{'alpha':[1e-4,1e-2],'hidden_layer_sizes':[(64,32),(128,32)]}),
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--element-properties',type=Path,required=True)
    p.add_argument('--target',choices=['UTS','EL'],required=True)
    p.add_argument('--folds',type=Path,required=True,help='HGNN cross_validation/oof_predictions.csv')
    p.add_argument('--split',choices=['random','group'],default='group')
    p.add_argument('--tune',action='store_true',help='Inner-CV hyperparameter search.')
    p.add_argument('--variant',choices=['full','base'],default='full')
    p.add_argument('--output',type=Path,default=Path('outputs/benchmarks'))
    a=p.parse_args()
    _,_,_,graphs=prepare_phase_dataset(a.data,a.element_properties,a.target,ELEMENT_FEATURES[a.target],ablation=a.variant)
    reference=pd.read_csv(a.folds,dtype={'sample_no':str})
    ids=[g.sample_no for g in graphs]
    if reference.sample_no.duplicated().any() or set(reference.sample_no)!=set(ids):raise ValueError('Fold IDs do not match training records.')
    reference=reference.set_index('sample_no').loc[ids]
    if a.split=='group':validate_group_metadata(graphs)
    y=np.array([g.y[0] for g in graphs]);x=np.stack([flatten(g) for g in graphs]);groups=leakage_groups(graphs)
    if not np.allclose(y,reference[f'true_{a.target}']):raise ValueError('Targets differ from the reference HGNN run.')
    folds=reference['fold'].to_numpy();summary=[];params={}
    a.output.mkdir(parents=True,exist_ok=True)
    for name,(estimator,grid) in estimators(3407).items():
        prediction=np.empty(len(y));params[name]={}
        for fold in sorted(set(folds)):
            train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
            if a.split=='group' and set(groups[train]) & set(groups[test]):raise ValueError('Reference folds are not group-disjoint.')
            pipeline=Pipeline([('scale',StandardScaler()),('regressor',TransformedTargetRegressor(regressor=estimator,transformer=StandardScaler()))])
            if a.tune:
                inner=GroupKFold(3) if a.split=='group' else KFold(3,shuffle=True,random_state=3407+int(fold))
                search=GridSearchCV(pipeline,{f'regressor__regressor__{key}':v for key,v in grid.items()},cv=inner,scoring='neg_mean_squared_error',n_jobs=1)
                search.fit(x[train],y[train],**({'groups':groups[train]} if a.split=='group' else {}))
                fitted=search.best_estimator_;params[name][str(fold)]=search.best_params_
            else:
                fitted=pipeline.fit(x[train],y[train]);params[name][str(fold)]={'tuned':False}
            prediction[test]=fitted.predict(x[test])
        rows=pd.DataFrame({'sample_no':ids,'route':[g.route for g in graphs],'fold':folds,'true':y,'pred':prediction})
        rows.to_csv(a.output/f'{a.target}_{a.variant}_{name}_oof.csv',index=False)
        for scope,part in [('all',rows),*list(rows.groupby('route'))]:
            summary.append({'model':name,'scope':scope,'variant':a.variant,**regression_metrics(part.true.to_numpy(),part.pred.to_numpy())})
    pd.DataFrame(summary).to_csv(a.output/f'{a.target}_{a.variant}_metrics.csv',index=False)
    (a.output/f'{a.target}_{a.variant}_parameters.json').write_text(json.dumps(params,indent=2),encoding='utf-8')

if __name__=='__main__':main()
