"""Original, deterministic numerical teaching kernels. No network calls or private data."""
import math
import random
from collections import Counter
import re
import numpy as np


def activity_data(seed=17, n=120, missing=0.12):
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0, 10, n)
    practice = rng.integers(0, 12, n)
    scores = np.clip(25 + 4 * hours + 2 * practice + rng.normal(0, 9, n), 0, 100)
    groups = np.where(np.arange(n) % 2, 'A', 'B')
    dirty = hours.copy()
    dirty[rng.random(n) < missing] = np.nan
    return dict(hours=dirty, practice=practice, score=scores, group=groups)


def regression(epochs=100, lr=0.05, degree=1, seed=11):
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, 70)
    y = 0.3 + 0.6 * x + 0.35 * x ** 2 + rng.normal(0, 0.12, len(x))
    train = np.arange(len(x)) % 3 != 0
    X = np.stack([x ** i for i in range(degree + 1)], axis=1)
    w = np.zeros(degree + 1)
    losses = []
    for _ in range(epochs):
        e = X[train] @ w - y[train]
        w -= lr * 2 * X[train].T @ e / train.sum()
        losses.append(float(np.mean(e * e)))
    pred = X @ w
    return x, y, pred, losses, float(np.mean((pred[train]-y[train])**2)), float(np.mean((pred[~train]-y[~train])**2))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def classification(seed=17, threshold=0.5, shift=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(240, 2))
    y = (x[:, 0] + 0.7 * x[:, 1] + rng.normal(0, 0.6, 240) > 0).astype(int)
    X = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(3)
    for _ in range(300):
        p = sigmoid(X[:150] @ w)
        w -= 0.12 * X[:150].T @ (p-y[:150]) / 150
    xt = X[150:].copy(); xt[:, 1] += shift
    probs = sigmoid(xt @ w); yt = y[150:]; yh = probs >= threshold
    tp = int(np.sum(yh & (yt == 1))); fp = int(np.sum(yh & (yt == 0)))
    fn = int(np.sum(~yh & (yt == 1))); tn = int(np.sum(~yh & (yt == 0)))
    return {'x': x[150:], 'y': yt, 'probs': probs, 'pred': yh, 'w': w, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'accuracy': (tp+tn)/len(yt), 'precision': tp/max(tp+fp,1), 'recall': tp/max(tp+fn,1)}


def train_xor(epochs=500, lr=0.8, hidden=4, seed=4):
    rng = np.random.default_rng(seed)
    x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([[0.], [1.], [1.], [0.]])
    w1 = rng.normal(0, 1, (2, hidden)); b1 = np.zeros((1, hidden))
    w2 = rng.normal(0, 1, (hidden, 1)); b2 = np.zeros((1, 1)); losses=[]
    for _ in range(epochs):
        h = np.tanh(x @ w1 + b1); p = sigmoid(h @ w2 + b2)
        losses.append(float(-np.mean(y*np.log(p+1e-9)+(1-y)*np.log(1-p+1e-9))))
        dz=(p-y)/4; dw2=h.T@dz; db2=dz.sum(axis=0,keepdims=True)
        dh=(dz@w2.T)*(1-h*h)
        dw1=x.T@dh; db1=dh.sum(axis=0,keepdims=True)
        w2-=lr*dw2; b2-=lr*db2; w1-=lr*dw1; b1-=lr*db1
    h=np.tanh(x@w1+b1); p=sigmoid(h@w2+b2)
    return losses, p.ravel(), w1, w2


def conv2(image, kernel):
    image=np.asarray(image,dtype=float); kernel=np.asarray(kernel,dtype=float)
    h,w=image.shape; kh,kw=kernel.shape
    return np.array([[np.sum(image[i:i+kh,j:j+kw]*kernel) for j in range(w-kw+1)] for i in range(h-kh+1)])


def attention(tokens=5, temperature=1.0, causal=True):
    rng=np.random.default_rng(8); x=rng.normal(size=(tokens,4))
    q=x@rng.normal(size=(4,3)); k=x@rng.normal(size=(4,3)); v=x@rng.normal(size=(4,3))
    scores=q@k.T/math.sqrt(3)/temperature
    if causal: scores[np.triu_indices(tokens,1)] = -1e9
    scores-=scores.max(axis=1,keepdims=True); a=np.exp(scores); a/=a.sum(axis=1,keepdims=True)
    return a,a@v


def recurrent(memory=0.6, length=12):
    inputs=np.sin(np.arange(length)*0.8); h=0.; states=[]
    for x in inputs:
        h=math.tanh(memory*h+float(x)); states.append(h)
    return inputs,np.array(states)


def autoencoder(epochs=200, latent=2, seed=5):
    rng=np.random.default_rng(seed); t=np.linspace(0,2*np.pi,80)
    X=np.column_stack([np.sin(t),np.cos(t),np.sin(2*t),np.cos(2*t)])
    w1=rng.normal(0,.3,(4,latent)); w2=rng.normal(0,.3,(latent,4)); losses=[]
    for _ in range(epochs):
        z=np.tanh(X@w1); out=z@w2; err=out-X
        losses.append(float(np.mean(err**2)))
        dw2=z.T@err*2/X.size; dz=(err@w2.T)*(1-z*z)*2/X.size
        dw1=X.T@dz; w1-=.25*dw1; w2-=.25*dw2
    return X,np.tanh(X@w1)@w2,losses


def denoise(noise=0.4, smooth=5, seed=9):
    rng=np.random.default_rng(seed); x=np.linspace(0,2*np.pi,80); clean=np.sin(x)
    noisy=clean+rng.normal(0,noise,80)
    pad=smooth//2; padded=np.pad(noisy,(pad,pad),mode='edge')
    estimate=np.convolve(padded,np.ones(smooth)/smooth,mode='valid')[:80]
    return clean,noisy,estimate


def temperature_distribution(temperature=1.0):
    logits=np.array([2.8,2.,1.2,.3,-.2]); p=np.exp((logits-logits.max())/temperature); p/=p.sum()
    entropy=float(-np.sum(p*np.log2(p)))
    return ['learn','build','test','guess','skip'],p,entropy


def paired_alignment(seed=6, noise=0.2):
    rng=np.random.default_rng(seed); image=rng.normal(size=(5,4)); text=image+rng.normal(0,noise,(5,4))
    image/=np.linalg.norm(image,axis=1,keepdims=True); text/=np.linalg.norm(text,axis=1,keepdims=True)
    return image@text.T


CORPUS=[
 ('PY','Python collections include lists, dictionaries and sets. Functions make data checks reusable.'),
 ('DATA','NumPy supports numeric arrays. pandas can detect missing values and aggregate tabular records.'),
 ('ML','Fit preprocessing on training data only. Keep a final test set separate from model selection.'),
 ('DL','Neural networks learn weights with gradients. Convolution uses shared filters over local windows.'),
 ('GEN','An autoencoder reconstructs its input. Denoising alone does not make a model Stable Diffusion.'),
 ('RAG','Retrieval selects evidence before answering. A grounded answer cites its source and may abstain.'),
 ('OPS','Monitor latency, validation errors and data drift. Drift alone does not prove accuracy has fallen.'),
 ('SAFE','An agent should have a step budget, explicit tool permissions and human review for external actions.')]


def tokenize(text):
    return re.findall(r'[a-z]+',text.lower())


def retrieve(query, topk=3, corpus=None):
    corpus=CORPUS if corpus is None else corpus
    docs=[Counter(tokenize(t)) for _,t in corpus]; q=Counter(tokenize(query)); n=len(docs)
    vocabulary=set(q)
    for d in docs: vocabulary.update(d)
    idf={t:math.log((1+n)/(1+sum(t in d for d in docs)))+1 for t in vocabulary}
    def vec(c): return {t:v*idf[t] for t,v in c.items()}
    qv=vec(q); qn=math.sqrt(sum(v*v for v in qv.values())); ranked=[]
    for (key,text),d in zip(corpus,docs):
        dv=vec(d); dn=math.sqrt(sum(v*v for v in dv.values()))
        score=sum(v*dv.get(t,0) for t,v in qv.items())/max(qn*dn,1e-12)
        ranked.append((key,text,float(score)))
    return sorted(ranked,key=lambda r:-r[2])[:topk]


def answer(query, topk=2, min_score=.1):
    rows=retrieve(query,topk)
    kept=[r for r in rows if r[2]>=min_score]
    return (' '.join(f'[{key}] {text}' for key,text,_ in kept) if kept else 'Not enough evidence in this tiny corpus.'),rows


def agent_trace(budget=5, review=True, approved=False):
    stages=['intake','retrieve','draft','review','final'] if review else ['intake','retrieve','draft','final']
    out=[]
    for i,stage in enumerate(stages):
        if i>=budget: return out+['STOP: step budget reached']
        out.append(stage)
        if stage=='review' and not approved: return out+['WAIT: human approval required']
    return out


def psi(reference,current,bins=8):
    lo=min(min(reference),min(current)); hi=max(max(reference),max(current))
    edges=np.linspace(lo-1e-6,hi+1e-6,bins+1)
    a=np.histogram(reference,bins=edges)[0].astype(float)+.5
    b=np.histogram(current,bins=edges)[0].astype(float)+.5
    a/=a.sum(); b/=b.sum()
    return float(np.sum((b-a)*np.log(b/a))),a,b


def serving(payload,model=None):
    if not isinstance(payload,dict) or set(payload)!={'hours','practice'}:
        raise ValueError('Expected exactly hours and practice')
    for name,bound in [('hours',24),('practice',100)]:
        v=payload[name]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=bound:
            raise ValueError('Invalid '+name)
    model=model or {'intercept':20.,'hours':4.,'practice':2.,'version':'teaching-v1'}
    score=min(100.,max(0.,model['intercept']+model['hours']*payload['hours']+model['practice']*payload['practice']))
    return {'score':round(score,2),'model_version':model['version'],'synthetic_demo':True}


def rollout(old_error=.16,new_error=.14,latency=120,budget=200):
    return new_error<=old_error and latency<=budget
