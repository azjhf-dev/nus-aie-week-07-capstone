# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo==0.25.2", "numpy>=2.0,<3", "pandas>=2.2,<3"]
# ///
# Original independent self-study material. Not an official NUS course.

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Build and defend an end-to-end project",
)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import json

    return mo, np, pd


@app.cell
def _(np):
    import math
    import random
    from collections import Counter
    import re


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

    return answer, classification


@app.cell
def _(np):
    import html as html_lib

    def chart_wrap(title, svg):
        return '<section style="margin:12px 0;padding:14px;background:#f6f8f6;border-radius:12px"><p style="font:600 15px/1.4 system-ui;margin:0 0 10px;color:#20332e">'+html_lib.escape(title)+'</p>'+svg+'</section>'

    def line_graph(series, labels, title, width=400, height=250):
        colors=['#12675c','#b56420','#7353a6','#3465a4']
        all_values=[float(v) for s in series for v in s]
        low=min(all_values+[0]); high=max(all_values+[0]); span=max(high-low,1e-6)
        n=max(len(s) for s in series); left=48; right=width-15; top=14; bottom=height-48
        pieces=[f'<svg role="img" aria-label="{html_lib.escape(title)}" viewBox="0 0 {width} {height}" style="width:100%;height:auto"><title>{html_lib.escape(title)}</title>']
        for frac in [0,.5,1]:
            y=bottom-frac*(bottom-top); val=low+frac*span
            pieces.append(f'<path d="M{left},{y} H{right}" stroke="#dce5df"/><text x="1" y="{y+4}" font-size="14" fill="#435a50">{val:.2f}</text>')
        for i,(s,label) in enumerate(zip(series,labels)):
            pts=' '.join(f'{left+j/max(n-1,1)*(right-left):.2f},{bottom-(float(v)-low)/span*(bottom-top):.2f}' for j,v in enumerate(s))
            pieces.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i%4]}" stroke-width="2.5"/>')
        pieces.append('</svg>')
        legend='<p style="font:13px/1.7 system-ui;margin:2px 0 0">'+' &nbsp; '.join(f'<span style="color:{colors[i%4]}">&#9679; {html_lib.escape(label)}</span>' for i,label in enumerate(labels))+'</p>'
        return chart_wrap(title,''.join(pieces)+legend)

    def bars(values,labels,title):
        colors=['#12675c','#b56420','#7353a6','#3465a4']; n=len(values); maxv=max([float(v) for v in values]+[1e-8]); h=12+44*n
        s=[f'<svg role="img" aria-label="{html_lib.escape(title)}" viewBox="0 0 400 {h}" style="width:100%;height:auto"><title>{html_lib.escape(title)}</title>']
        for i,(v,label) in enumerate(zip(values,labels)):
            y=4+44*i; w=max(0,float(v)/maxv*210)
            s.append(f'<text x="0" y="{y+18}" font-size="14" fill="#20332e">{html_lib.escape(str(label)[:16])}</text><rect x="132" y="{y}" width="{w:.2f}" height="25" rx="4" fill="{colors[i%4]}"/><text x="{137+w:.2f}" y="{y+18}" font-size="14" fill="#20332e">{float(v):.3f}</text>')
        return chart_wrap(title,''.join(s)+'</svg>')

    def heat(matrix,title):
        a=np.asarray(matrix); rows,cols=a.shape; mn=float(a.min()); mx=float(a.max()); span=max(mx-mn,1e-9)
        cw=min(65,380/max(cols,1)); ch=min(55,300/max(rows,1)); w=cols*cw+14; h=rows*ch+14
        parts=[f'<svg role="img" aria-label="{html_lib.escape(title)}" viewBox="0 0 {w} {h}" style="width:100%;max-width:500px;height:auto"><title>{html_lib.escape(title)}</title>']
        for i in range(rows):
            for j in range(cols):
                v=float(a[i,j]); t=(v-mn)/span; color=f'rgb({int(236-210*t)},{int(244-130*t)},{int(239-135*t)})'
                x=7+j*cw;y=7+i*ch
                parts.append(f'<rect x="{x}" y="{y}" width="{cw-2}" height="{ch-2}" rx="3" fill="{color}"/><text x="{x+cw/2}" y="{y+ch*.62}" text-anchor="middle" fill="{"white" if t>.58 else "#20332e"}" font-size="16">{v:.2f}</text>')
        return chart_wrap(title,''.join(parts)+'</svg>')

    def scatter(real,sample,title):
        arrays=[np.asarray(real),np.asarray(sample)]
        allp=np.concatenate(arrays);low=allp.min(axis=0);high=allp.max(axis=0);span=np.maximum(high-low,1e-6)
        pieces=['<svg role="img" aria-label="Original and generated point clouds" viewBox="0 0 400 300" style="width:100%;height:auto">']
        for a,color in zip(arrays,['#12675c','#b56420']):
            for xy in a:
                x,y=(xy-low)/span;pieces.append(f'<circle cx="{20+x*360:.2f}" cy="{275-y*250:.2f}" r="2.2" fill="{color}" opacity=".5"/>')
        pieces.append('</svg><p style="font:13px system-ui;color:#435a50">Green: training points. Orange: newly generated points. Axes share the same scale within this plot.</p>')
        return chart_wrap(title,''.join(pieces))

    def flow(nodes,enabled=True):
        parts=['<style>@keyframes labPulse{0%,100%{opacity:.35}50%{opacity:1}}@media(prefers-reduced-motion:reduce){.lab-pulse{animation:none!important}}</style>', '<svg role="img" aria-label="Animated conceptual data flow" viewBox="0 0 400 235" style="width:100%;height:auto"><title>Conceptual data flow. Motion shows order, not measured time.</title>']
        for i,name in enumerate(nodes):
            col=i%2;row=i//2;x=8+col*195;y=12+row*98
            sty=f'animation:labPulse 3s {i*.45}s infinite' if enabled else ''
            parts.append(f'<g class="lab-pulse" style="{sty}"><rect x="{x}" y="{y}" width="180" height="70" rx="11" fill="#12675c"/><text x="{x+90}" y="{y+42}" text-anchor="middle" fill="white" font-size="19">{i+1}. {html_lib.escape(name)}</text></g>')
        parts.append('</svg><p style="font:13px/1.4 system-ui;color:#567068">Conceptual sequence, not a measured timeline. Pause with the checkbox above.</p>')
        return ''.join(parts)

    return bars, flow


@app.cell
def _(mo):
    mo.md("""
    # Week 7: Build and defend an end-to-end project

    A full week of self-study, mapped to **Capstone Project for AI Engineering**. Estimated **20 active hours**, not an official contact-hour claim.

    **Independent reconstruction, not NUS-issued teaching material.** Published objectives are the anchor. Exercises, datasets and rubric are original.

    - Frame a problem and measurable acceptance criteria
    - Combine a predictor and retrieval helper with source tracing
    - Run hold-out tests, inspect subgroup errors and present honest limits

    [Official objectives](https://www.comp.nus.edu.sg/executive-education/course/sctp-capstone-project-for-ai-engineering/) | [Repository](https://github.com/azjhf-dev/nus-aie-week-07-capstone)

    **Boundary:** Assessment weights, expected capstone size, official datasets and marking rubrics are unpublished in the evidence inspected. The rubric supplied here is an original self-study rubric, not NUS's.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Plan your week

    - Day 1: scope, dataset card and evaluation plan (3h)
    - Day 2: baseline pipeline and user interface (4h)
    - Day 3: predictor plus retrieval integration (4h)
    - Day 4: evaluation, incident drill and fairness caveats (4h)
    - Day 5: clean rerun, report and five-minute demo (5h)

    Start each lab by predicting the effect of a control. Change one variable, record the observation, then explain why. All default experiments need no key, no paid service and no private data. The hosted WASM viewer needs an internet connection to fetch its runtime and packages.
    """)
    return


@app.cell
def _(mo):
    motion=mo.ui.checkbox(value=True,label="Play conceptual motion graphic")
    motion
    return (motion,)


@app.cell
def _(flow, mo, motion):
    mo.Html(flow(['Frame', 'Build', 'Test', 'Defend'],motion.value))
    return


@app.cell
def _(mo):
    mo.md("""
    ## 01 | Define the acceptance criterion first

    The predictor uses a synthetic binary task. It is not a system for deciding what real pupils deserve. This integrated capstone combines numerical prediction, explicit error costs and an authored retrieval corpus.
    """)
    return


@app.cell
def _(mo):
    cap_threshold=mo.ui.slider(0.1,0.9,step=0.05,value=0.5,label="Intervention threshold")
    cap_min_recall=mo.ui.slider(0.5,1,step=0.05,value=0.8,label="Proposed minimum recall")
    mo.vstack([cap_threshold,cap_min_recall])
    return cap_min_recall, cap_threshold


@app.cell
def _(bars, cap_min_recall, cap_threshold, classification, mo):
    cap_result=classification(seed=91,threshold=cap_threshold.value)
    _recall=cap_result['recall']; _passes=_recall>=cap_min_recall.value
    mo.vstack([mo.Html(bars([cap_result['precision'],_recall,cap_result['accuracy']],["precision","recall","accuracy"],"Capstone evaluation on a separate seeded synthetic set")),mo.md(f"Recall gate: **{'PASS' if _passes else 'HOLD'}**. Interactive threshold tuning makes this a validation set. A final test must use a new locked seed and no further tuning.")])
    return (cap_result,)


@app.cell
def _(mo):
    mo.md("""
    ## 02 | Inspect errors across synthetic groups

    Groups are alternating row labels, not protected attributes. The gap is a sampling illustration only. Small denominators can make a large-looking difference unstable; equal aggregate accuracy is not proof of fairness.
    """)
    return


@app.cell
def _(mo):
    min_group=mo.ui.slider(5,40,value=20,label="Minimum group sample before reporting")
    min_group
    return (min_group,)


@app.cell
def _(cap_result, min_group, mo, np, pd):
    _rows=[]
    for _g in [0,1]:
        _mask=np.arange(len(cap_result['y']))%2==_g; _n=int(_mask.sum())
        _acc=float(np.mean(cap_result['pred'][_mask]==cap_result['y'][_mask]))
        _rows.append({'synthetic_group':str(_g),'n':_n,'accuracy':round(_acc,3) if _n>=min_group.value else 'not enough records'})
    mo.ui.table(pd.DataFrame(_rows),selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 03 | Add evidence, not invented advice

    The retrieval helper cites an authored source ID. An extractive answer may be awkward but its evidence is inspectable. A real educational product would need privacy, safeguarding and pedagogical review.
    """)
    return


@app.cell
def _(mo):
    cap_question=mo.ui.text(value="How can I validate training and test data?",label="Study-resource question",full_width=True)
    cap_question
    return (cap_question,)


@app.cell
def _(answer, cap_question, mo, pd):
    _response,_sources=answer(cap_question.value,2,.1)
    mo.vstack([mo.md('**Resource helper:** '+_response),mo.ui.table(pd.DataFrame(_sources,columns=['id','source text','score']),selection=None)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 04 | Prepare a release decision

    A checklist is evidence bookkeeping, not automated certification. Only check an item when its artifact exists. The original self-study rubric is in ASSESSMENT.md, not an official NUS marking guide.
    """)
    return


@app.cell
def _(mo):
    data_card=mo.ui.checkbox(label="Dataset card and source permissions documented")
    test_lock=mo.ui.checkbox(label="Final test locked before evaluation")
    limits_doc=mo.ui.checkbox(label="Failure cases and limitations written")
    runbook_done=mo.ui.checkbox(label="Clean rerun and rollback rehearsal recorded")
    mo.vstack([data_card,test_lock,limits_doc,runbook_done])
    return data_card, limits_doc, runbook_done, test_lock


@app.cell
def _(data_card, limits_doc, mo, runbook_done, test_lock):
    _flags=[data_card.value,test_lock.value,limits_doc.value,runbook_done.value]
    mo.md(f"**Release evidence: {sum(_flags)}/{len(_flags)} items.** "+('Ready for a human review, not automatic real-world use.' if all(_flags) else 'Still a learning demonstrator. Record the missing artifacts before presenting it as complete.'))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Build something, then defend it

    **This week's deliverable:** Study-support demonstrator, using only synthetic records and a tiny authored corpus.

    Create a dated lab log with: question, prediction, changed input, observed output, explanation, failure case and next test. Keep the final evaluation separate from the data used for choosing settings.
    """)
    return


@app.cell
def _(mo):
    mo.accordion({'Can this capstone be used to rank real pupils?': 'No. The data and task are synthetic, the evaluation is small, and the system lacks the necessary privacy, safeguarding, pedagogical and impact review.', 'What is the difference between a demo and evidence?': 'A demo shows a path can work; evidence includes held-out tests, failed cases, repeatability, provenance and limitations.', 'Build task': 'Produce DATA_CARD.md, MODEL_CARD.md, EVALUATION.md, DEMO.md and an incident runbook. Present one success and two failures, then explain what would change the release decision.'})
    return


@app.cell
def _(mo):
    mo.md("""
    ## Evidence and next depth

    These are **other NUS-ISS courses and projects**, not submissions from this exact SCTP programme. A public repository does not establish permission to copy everything in it. The research CSV records license and evidence depth.

    - `anastasiapatricks/JobAId`
    - `NUS-AIS-Practice-Modules/IRS-PM-2025-08-30-AIS11FT-GRP-NeverStopCoding-WhatsEat`
    - `atsui888/MTech-AI-Capstone-Project`
    - `IRS-PM/Workshop-Project-Submission-Template`

    See `CURRICULUM.md`, `ASSESSMENT.md`, `SOURCES.md` and the portfolio files for more work. Real pretrained model training and live-service integrations are optional extensions, not things this notebook silently performs.

    Checked 19 November 2026. Record your own results; published student results are not reproduced here.
    """)
    return


if __name__ == "__main__":
    app.run()
