"""A locked final evaluation, independent of the exploratory notebook."""
import json,hashlib
from pathlib import Path
import numpy as np
from labcore import sigmoid,retrieve


def evaluate():
    seed=20261119;rng=np.random.default_rng(seed)
    x=rng.normal(size=(400,2));y=(x[:,0]+.7*x[:,1]+rng.normal(0,.6,400)>0).astype(int)
    X=np.column_stack([np.ones(len(x)),x]);w=np.zeros(3)
    # Fixed split and optimizer before looking at final test outcomes.
    for _ in range(400):w-=.1*X[:240].T@(sigmoid(X[:240]@w)-y[:240])/240
    candidates=np.arange(.2,.81,.05);pv=sigmoid(X[240:320]@w)
    def cost(labels,pred):return int(3*np.sum((labels==1)&~pred)+np.sum((labels==0)&pred))
    chosen=float(min(candidates,key=lambda t:cost(y[240:320],pv>=t)))
    test_y=y[320:];pred=sigmoid(X[320:]@w)>=chosen
    tp=int(np.sum(pred&(test_y==1)));fp=int(np.sum(pred&(test_y==0)));fn=int(np.sum(~pred&(test_y==1)));tn=int(np.sum(~pred&(test_y==0)))
    evaluation=[('missing values in tabular arrays','DATA'),('training preprocessing test set','ML'),('step budget tool permissions review','SAFE'),('monitor validation latency drift','OPS')]
    hits=sum(gold in [r[0] for r in retrieve(q,2)] for q,gold in evaluation)
    artifact={'version':'synthetic-logistic-v1','weights':w.tolist(),'threshold':chosen,'training_rows':240,'validation_rows':80,'test_rows':80,'seed':seed}
    fingerprint=hashlib.sha256(json.dumps(artifact,sort_keys=True).encode()).hexdigest()
    report={'artifact_sha256':fingerprint,'test':{'tp':tp,'fp':fp,'fn':fn,'tn':tn,'accuracy':(tp+tn)/80,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'cost_units':3*fn+fp},'retrieval':{'hits':hits,'n':len(evaluation),'k':2},'limitation':'Synthetic fixed-split evidence only. Repeatedly tuning against these results invalidates final-test status.'}
    return artifact,report


if __name__=='__main__':
    artifact,report=evaluate()
    out=Path('artifacts');out.mkdir(exist_ok=True)
    (out/'model.json').write_text(json.dumps(artifact,indent=2));(out/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
