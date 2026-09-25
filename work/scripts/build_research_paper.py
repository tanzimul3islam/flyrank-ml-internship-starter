"""Build the public paper and charts from aggregate capstone receipts only."""
from pathlib import Path
import hashlib
import html
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/outputs/capstone_cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = ROOT / 'work/outputs'
PAPER = ROOT / 'work/paper'
FIGURES = ROOT / 'work/figures'
REPO = 'https://github.com/tanzimul3islam/flyrank-ml-internship-starter'
BASE = REPO + '/blob/main/'
TITLE = 'Does a higher top-20 score mean a better content queue?'


def build():
    s = json.loads((OUT / 'capstone_metrics.json').read_text())
    p = s['june_primary']; m, b = p['model'], p['baseline']
    march = s['march_comparison']; secondary = s['june_secondary_all_clients']
    (PAPER / 'assets').mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#c3cfce',
                         'text.color':'#142b35','axes.labelcolor':'#142b35','xtick.color':'#52616a',
                         'ytick.color':'#52616a','figure.facecolor':'#fffdf9','axes.facecolor':'#fffdf9',
                         'savefig.facecolor':'#fffdf9','svg.fonttype':'none'})
    def save(fig, name):
        for ext in ['svg','png']:
            path=FIGURES/f'{name}.{ext}'
            fig.savefig(path,bbox_inches='tight',dpi=190)
            shutil.copyfile(path,PAPER/'assets'/path.name)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,4.6))
    x=np.arange(2);width=.25
    aa=[march['baseline']['precision_at_k'],b['precision_at_k']]
    bb=[march['model']['precision_at_k'],m['precision_at_k']]
    cc=[march['baseline']['base_rate'],b['base_rate']]
    for delta,vals,color,label in [(-width,aa,'#b8522e','Volume baseline'),(0,bb,'#166a84','Frozen tree'),(width,cc,'#a1afab','Decline prevalence')]:
        bars=ax.bar(x+delta,np.array(vals)*100,width=.22,color=color,label=label)
        ax.bar_label(bars,labels=[f'{v:.0%}' for v in vals],padding=5,fontsize=10)
    ax.set(xticks=x,xticklabels=['March · reused comparison','June · final client holdout'],ylim=(0,105),ylabel='Declining pages in top 20 (%)')
    ax.legend(loc='upper left',frameon=False,ncol=3,fontsize=9);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    save(fig,'top20-comparison')
    groups=s['per_client_primary'];fig,ax=plt.subplots(figsize=(9,4.8))
    diffs=[100*(g['model_p_at_k']-g['baseline_p_at_k']) for g in groups]
    ys=np.arange(len(groups)); ax.barh(ys,diffs,color=['#b8522e' if v<0 else '#a1afab' for v in diffs],height=.6)
    ax.set_yticks(ys,[f"{g['anonymous_client']} · n={g['n']:,}" for g in groups]);ax.invert_yaxis()
    ax.axvline(0,color='#142b35',linewidth=1)
    for y,d,g in zip(ys,diffs,groups):ax.text(d-1 if d<0 else 1,y,f'{d:+.0f} pp'+(' · all 17 pages' if g['k']<20 else ''),va='center',ha='right' if d<0 else 'left',fontsize=10)
    ax.set_xlim(-60,28);ax.set_xlabel('Tree minus baseline precision within each client (percentage points)')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    save(fig,'client-level-lift')
    imp=sorted(s['permutation_interpretation'],key=lambda d:d['mean_auc_drop'])
    labels={'mean_daily_impressions':'Past exposure','ctr_pct':'Past CTR','impression_weighted_position':'Past position','active_day_share':'Active-day share','daily_impression_cv':'Daily variability'}
    fig,ax=plt.subplots(figsize=(9,4.4))
    ax.barh([labels[r['feature']] for r in imp],[r['mean_auc_drop'] for r in imp],xerr=[r['std_auc_drop'] for r in imp],color='#166a84',height=.6,capsize=3)
    ax.axvline(0,color='#52616a',lw=.8);ax.set_xlabel('June ROC-AUC decrease after shuffling one input (5 repeats)');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    save(fig,'feature-interpretation')
    lo,hi=s['bootstrap']['lift_interval_95']
    abstract=(
        'Can a small search-data model help an editor choose which pages deserve inspection for a possible content refresh? '
        f'We studied pinned March and June 2026 FlyRank warehouse partitions, training on {s["train_pages"]:,} pages from {s["train_clients"]} clients and reserving later outcomes from untrained clients. '
        'A frozen depth-three decision tree used five past-window features and competed with a volume-only baseline under the same eligibility rules and decline proxy. '
        f'On {m["n"]:,} June pages across {m["clients"]} held-out clients, the tree found {m["top_k_declines"]} declines in its top 20 versus {b["top_k_declines"]} for the baseline, but it performed worse within all five clients large enough for a top-20 selection. '
        'The result supports a historical inspection pilot with explicit guardrails, while broad adoption and any claim of refresh benefit remain unsupported.')
    sections=[]
    def section(id,title,body):sections.append((id,title,body))
    section('abstract','Abstract',f'<p class="abstract">{abstract}</p>')
    section('problem','An editor has twenty review slots', '''<p>A content editor has thousands of visible pages and enough time to inspect only twenty. A useful queue should improve the use of those slots: identify outdated facts, missing coverage or an intent mismatch, while recognizing that technical problems and seasonal demand may call for a different response.</p>
<p>This is <strong>Refresh / Content Opportunity Scoring</strong>, the lane locked in Week 4. The output is a ranked inspection queue with reasons, not an automatic instruction to rewrite. A false positive wastes review time and may disturb successful content; a false negative delays attention. We do not have independent editorial judgments or measured refresh effects, so this study evaluates an observable traffic-decline proxy.</p>
<p>The question is therefore narrower than “does ML improve content?”: does a transparent model rank later-declining pages better than an exposure-first rule, and does that result hold within clients?</p>''')
    section('data','Two months, with the windows kept apart', f'''<p>The source is the gated <a href="https://huggingface.co/datasets/FlyRank/internship-warehouse">FlyRank internship warehouse</a>, release build v20260703, pinned to revision <code>{s['dataset_revision']}</code>. We read only March and June partitions of <code>fact_content_daily_performance</code>. No starter-CSV observations were substituted.</p>
<div class="table-wrap"><table><thead><tr><th>Measure</th><th>March</th><th>June</th></tr></thead><tbody>
<tr><td>Source page-day rows</td><td>{s['march_source']['source_rows']:,}</td><td>{s['june_source']['source_rows']:,}</td></tr>
<tr><td>Exact duplicate rows removed</td><td>0</td><td>{s['june_source']['identical_extra_rows_removed']:,}</td></tr>
<tr><td>Past-eligible pages</td><td>{s['march_eligible_pages']:,}</td><td>{s['june_eligible_pages']:,}</td></tr>
<tr><td>Pages with complete labels</td><td>{s['march_labeled_pages']:,}</td><td>{s['june_labeled_pages']:,}</td></tr></tbody></table></div>
<div class="timeline" aria-label="Feature and outcome windows within each month"><div class="time-block"><strong>Days 1–14</strong>Past features · 14 valid GSC days</div><div class="time-block buffer"><strong>Days 15–16</strong>Reporting buffer</div><div class="time-block outcome"><strong>Days 17–30</strong>Later outcome · decision on day 17</div></div>
<p>One source row is a client–page–day; one scored row is a client–page at a decision date. Pages need at least 100 impressions and 14 valid past days. Availability is checked with <code>gsc_data_available IS TRUE</code>; unavailable or NULL measurements are not interpreted as zero traffic. Both windows need complete coverage to assign a label, but unknown outcomes do not remove pages from the decision-time queue.</p>
<p>The first June grain check found <strong>6,390 identical extra rows and zero conflicting versions</strong> across all selected fields. The run stopped before model scoring. An auditable <a href="{BASE}work/capstone_data_amendment.json">data-only amendment</a> removes those copies; conflicting duplicates still cause failure. June has {s['june_source']['clean_page_day_rows']:,} page-days after cleaning.</p>
<p>We exclude client identities, domains, page URLs, private queries, GA4 measurements, mutable content metadata and the overlapping fixed-window query table. Hash IDs are used only for grouping and deterministic ties. The final-month <code>_sample</code> table was not used for development.</p>''')
    section('method','A small model and a frozen comparison', f'''<p>The baseline, <code>volume_first_v1</code>, ranks eligible pages by past impressions. Week 4 found mixed evidence for both volume and volatility, so it dropped a proposed volatility bonus and retained exposure as a simple review-priority policy.</p>
<p>The model is a decision tree with <strong>maximum depth 3, minimum leaf size 100 and seed 42</strong>. Its five features are defined below. Median imputation is fitted only on training clients. Daily position values below 1 are treated as missing: this corrects the anomaly identified in Week 4 without changing row eligibility.</p>
<div class="table-wrap"><table><thead><tr><th>Input</th><th>Definition, using days 1–14 only</th></tr></thead><tbody>
<tr><td>Daily exposure</td><td>Past impressions / 14</td></tr><tr><td>CTR</td><td>100 × past clicks / past impressions</td></tr>
<tr><td>Position</td><td>Impression-weighted mean of valid daily positions</td></tr><tr><td>Active-day share</td><td>Days with positive impressions / 14</td></tr>
<tr><td>Daily variability</td><td>Population SD of daily impressions / daily mean</td></tr></tbody></table></div>
<p>The binary proxy is <code>later_14d_impressions &lt; 0.8 × past_14d_impressions</code>. The model never receives future counts, the proxy label, client IDs or product flags. Report-date separation is verified; the two-day buffer is an availability assumption, not proof of historical ingestion time.</p>
<p>We retain Week 4’s client split: {s['train_pages']:,} March training pages from {s['train_clients']} clients, plus a previously inspected comparison set of {s['march_comparison_pages']:,} pages from {s['march_comparison_clients']} other clients. Five-fold client-grouped validation uses only training clients, with no hyperparameter search. The March comparison is explicitly reused, not advertised as fresh.</p>
<p>The specification was committed in <a href="{REPO}/commit/01a5c7b">01a5c7b before the first June read</a>. The primary final cohort contains June pages from the original March holdout clients; six of those clients meet June eligibility. June was not used to refit or tune. <strong>Precision@20</strong> is primary; prevalence, ROC-AUC, average precision, client-specific rankings and a 500-resample paired client bootstrap provide context. Reproducibility reruns retain the same frozen choices.</p>''')
    rows=''
    for name,r in [('March · reused comparison',march),('June · primary final cohort',p),('June · all clients (secondary)',secondary)]:
        for model_name in ['baseline','model']:
            v=r[model_name]
            rows+=f'<tr><td>{name}</td><td>{"Volume baseline" if model_name=="baseline" else "Frozen tree"}</td><td>{v["precision_at_k"]:.0%}</td><td>{v["roc_auc"]:.3f}</td><td>{v["average_precision"]:.3f}</td><td>{v["base_rate"]:.2%}</td></tr>'
    section('results','The pooled gain does not survive the client check', f'''<p>On the primary June cohort, the model’s first 20 contain <strong>{m['top_k_declines']} declining pages ({m['precision_at_k']:.0%})</strong>, versus <strong>{b['top_k_declines']} ({b['precision_at_k']:.0%})</strong> for the volume rule. That is two additional proxy-positive pages, with a cohort decline prevalence of {b['base_rate']:.2%}.</p>
<figure><img src="assets/top20-comparison.svg" width="900" height="460" alt="Grouped bars: March baseline 45 percent, tree 70 percent; June baseline 75 percent, tree 85 percent; prevalence is shown for each cohort."><figcaption><span class="caption-label">Figure 1.</span> The tree leads on pooled precision@20, but the June lift is only two pages. March is a reused comparison set; June is the final temporal/client evaluation.</figcaption></figure>
<div class="table-wrap"><table class="numeric"><thead><tr><th>Cohort</th><th>Method</th><th>P@20</th><th>ROC-AUC</th><th>Avg. precision</th><th>Prevalence</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>Other checks temper that headline. The model’s June ROC-AUC is <strong>{m['roc_auc']:.3f}</strong>, below the baseline’s <strong>{b['roc_auc']:.3f}</strong>. Its global top 20 cover only {m['top_k_clients']} clients, versus {b['top_k_clients']} for the baseline. The primary cohort contains {m['n']:,} labeled pages, but these come from only {m['clients']} client groups.</p>
<figure><img src="assets/client-level-lift.svg" width="900" height="480" alt="Client-specific precision differences: model is worse by 15, 10, 45, 35 and 15 percentage points in five clients; the sixth has only 17 pages and both methods select all of them."><figcaption><span class="caption-label">Figure 2.</span> Within every client with more than 20 pages, the model is worse. The 17-page group is a tie because both methods include every page; it does not test ranking quality.</figcaption></figure>
<div class="callout caution"><p><strong>No reliable overall win.</strong> The paired client-bootstrap interval for the pooled precision difference spans <strong>{lo*100:+.0f} to {hi*100:+.0f} percentage points</strong>. With only six clusters, this is a descriptive uncertainty check, not a significance claim. We would not promote the model broadly based on this result.</p></div>
<p>The all-client June cohort is secondary and mixes previously seen and unseen clients. There the model achieves {secondary['model']['precision_at_k']:.0%} precision@20 versus {secondary['baseline']['precision_at_k']:.0%} for the baseline, reinforcing that the result depends on the pool being ranked. No tuning follows this observation.</p>
<figure><img src="assets/feature-interpretation.svg" width="900" height="440" alt="Permutation interpretation: past exposure and CTR account for most of the tree's small ROC-AUC discrimination; other features have smaller or negligible contributions."><figcaption><span class="caption-label">Figure 3.</span> Shuffling past exposure or CTR reduces the frozen model’s June AUC most. This explains reliance by this model, not a causal effect of those signals on search performance.</figcaption></figure>''')
    section('limitations','What the experiment cannot establish', '''<ul>
<li><strong>Decline is not refresh benefit.</strong> Neither editorial actionability nor traffic recovered after an edit was observed. Three of the model’s primary top 20 did not decline, and even a true decline can have a non-content cause.</li>
<li><strong>Client composition matters.</strong> A stronger global top 20 coexists with weaker within-client rankings and lower AUC. Client quotas or operational allocation policies would require a separately declared evaluation.</li>
<li><strong>Few independent groups.</strong> Tens of thousands of pages do not replace independent clients. The final six-client sample and wide bootstrap interval limit generalization.</li>
<li><strong>Coverage selects the labeled sample.</strong> Requiring later data can exclude disappearing pages or clients. Unknown outcomes are not treated as negative labels.</li>
<li><strong>Two snapshots are not a production replay.</strong> We do not know exact reporting arrival or later revisions, and one later month does not establish seasonal robustness. June recommendations are historical, not current September instructions.</li>
<li><strong>Coarse scores and imperfect measurements.</strong> The shallow tree produces tied leaf frequencies, not calibrated probabilities of editorial success. Position needed a documented quality repair. IDs resolve ties for reproducibility only.</li>
<li><strong>Review remains hypothetical.</strong> No page text, private query intent or technical configuration was inspected. Recommendations below are an inspection playbook, with low editorial confidence.</li></ul>''')
    recs=''
    for r in s['public_recommendations']:
        recs+=f'''<div class="recommendation"><div class="rank">{r['rank']:02d}</div><span class="tag">{html.escape(r['reason_code'])}</span><h3>{html.escape(r['action'])}</h3><p>{html.escape(r['why'])}</p><p><strong>Could be wrong if:</strong> {html.escape(r['wrong_if'])}</p><p class="small">Leaf score {r['model_score']:.3f} · Low editorial confidence</p></div>'''
    section('recommendations','A ranked inspection playbook, with a stop before editing', f'''<p>The predeclared global metric selects the model for this <strong>historical June 17 illustration</strong>; the adverse client-level audit argues against broad deployment. The queue contains {s['historical_queue_rows']:,} past-eligible pages from the original held-out clients, including {s['june_primary_unlabeled_pages']:,} with unknown later outcomes. The full pseudonymized queue stays local and is regenerated by code.</p>
<p>These ten public review slots retain their actual rank but omit page/client identifiers and raw measurements. An authorized editor can map them to the local CSV. Every slot calls for inspection; refreshing, rewriting, pruning or merging requires independent evidence. Similar leaf scores and reason codes are expected from a shallow tree and are a limitation of its resolution.</p>
<div class="recommendation-grid">{recs}</div>
<div class="callout"><p><strong>Operational guardrail:</strong> verify reporting coverage, compare query intent and seasonality, inspect content and technical health, and log the editor’s decision. Do not automate destructive content changes. Collect independent editorial precision@20 before claiming this system saves useful review time.</p></div>''')
    links=''.join(f'<li><a href="{BASE}work/notebooks/{f}.ipynb">{label}</a></li>' for f,label in [
        ('w01_research_question','Week 1 · research question'),('w02_ml_task_framing','Week 2 · task framing'),('w03_data_contract','Week 3 · data contract and leakage trap'),('w04_baseline_score','Week 4 · signal checks and frozen baseline'),('w05_model','Week 5 · model and grouped validation'),('w06_validation_audit','Week 6 · final evaluation and claim audit'),('w07_action_playbook','Week 7 · action playbook'),('capstone','Capstone · executable paper companion')])
    section('reproducibility','Follow the result back to an executed notebook', f'''<p>All research artifacts live in <a href="{REPO}">the public repository</a>. The code reads pinned gated data locally; no tokens or raw exports are published. Seed 42, the <a href="{BASE}work/capstone_protocol.json">frozen protocol</a>, the <a href="{BASE}work/capstone_data_amendment.json">data-quality amendment</a>, the <a href="{BASE}work/scripts/capstone_pipeline.py">pipeline</a>, and <a href="{BASE}work/outputs/capstone_metrics.json">aggregate metrics receipt</a> make the comparison inspectable. Split fingerprints verify the March cohort against Week 4.</p>
<pre><code>python -m pip install -r work/requirements-capstone.txt
# Accept warehouse access; authenticate locally with hf auth login.
python work/scripts/capstone_pipeline.py
python work/scripts/build_research_paper.py
python work/scripts/execute_capstone_notebooks.py</code></pre>
<p>Tested with Python {s['python']}, pandas {s['versions']['pandas']}, DuckDB {s['versions']['duckdb']} and scikit-learn {s['versions']['scikit-learn']}. The notebook runner executes the Week 5–7 notebooks and the capstone companion. Reruns may use the pinned local cache; no feature or model selection occurs after the final result.</p>
<ul class="refs">{links}</ul>
<p class="small">Chart downloads: <a href="assets/top20-comparison.png">global comparison</a>, <a href="assets/client-level-lift.png">client audit</a>, <a href="assets/feature-interpretation.png">model interpretation</a>. This page uses local assets and can be printed as a paper.</p>''')
    section('credit','Acknowledgments & data credit', '''<p><a href="https://flyrank.ai">Built on the FlyRank ML Internship dataset</a>.</p><p>Thanks to the FlyRank internship team for the warehouse release, starter materials and research framing. AI assistance supported code, execution checks, visualization and drafting. The ten recommendation slots have not received an independent human content review; no such review or causal experiment is claimed.</p>''')
    nav=''.join(f'<a href="#{id}">{i:02d} · {html.escape(title if len(title)<32 else {"problem":"Problem","data":"Data & windows","method":"Methodology","results":"Results","limitations":"Limitations","recommendations":"Ranked recommendations","reproducibility":"Reproducibility","credit":"Data credit"}.get(id,title))}</a>' for i,(id,title,_) in enumerate(sections,1))
    bodies=''.join(f'<section id="{id}" aria-labelledby="heading-{id}"><span class="section-no">{i:02d} / RESEARCH NOTE</span><h2 id="heading-{id}">{title}</h2>{body}</section>' for i,(id,title,body) in enumerate(sections,1))
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE} — FlyRank capstone</title><meta name="description" content="A frozen search-data model gains two declines in a global top-20 queue but loses within clients. A reproducible FlyRank capstone with honest validation."><meta name="theme-color" content="#f6f3ec"><link rel="stylesheet" href="assets/style.css"></head><body><a class="skip" href="#abstract">Skip to paper</a><div class="wrap"><header class="topbar"><span>Search intelligence / Research paper 01</span><a href="{REPO}">Open research repo ↗</a></header><div class="hero"><p class="eyebrow">FlyRank ML Internship · Refresh opportunity scoring</p><h1>{TITLE}</h1><p class="dek">A frozen tree finds two more declines at the top of the global queue. The client-level audit tells a less encouraging story.</p><div class="metadata"><span><strong>By</strong> tanzimul3islam</span><span><strong>Data</strong> March & June 2026</span><span><strong>Research date</strong> September 2026</span><span><strong>Reading time</strong> 10 minutes</span></div><div class="numbers"><div class="number"><strong>{m['top_k_declines']}/20</strong><span>declines in the model’s June top 20</span></div><div class="number"><strong>{b['top_k_declines']}/20</strong><span>declines in the volume baseline’s top 20</span></div><div class="number"><strong>5/5</strong><span>rankable clients where the model did worse</span></div></div></div><div class="paper-layout"><nav class="toc" aria-label="Paper sections"><strong>In this paper</strong>{nav}</nav><article>{bodies}</article></div><footer class="footer"><span>Observed patterns · Decision support · No causal refresh claim</span><a href="#">Back to top ↑</a></footer></div></body></html>'''
    (PAPER/'index.html').write_text(page)
    metadata={'title':TITLE,'protocol_sha256':s['protocol_sha256'],'metrics_sha256':hashlib.sha256((OUT/'capstone_metrics.json').read_bytes()).hexdigest(),'required_sections':9}
    (PAPER/'publication.json').write_text(json.dumps(metadata,indent=2)+'\n')
    # Human-readable companion with the same claims and data links; HTML is the designed publication.
    from bs4 import BeautifulSoup
    report=['# '+TITLE, '\n**Author:** tanzimul3islam · **Lane:** Refresh / Content Opportunity Scoring\n']
    for _,title,body in sections:report.append('## '+title+'\n\n'+BeautifulSoup(body,'html.parser').get_text(' ',strip=True)+'\n')
    report.append(f'\nRepository: {REPO}\nData credit: [Built on the FlyRank ML Internship dataset](https://flyrank.ai)\n')
    (ROOT/'work/capstone_report.md').write_text('\n'.join(report))
    print('Built nine-section paper, three charts (SVG/PNG), report, and publication fingerprint.')
    return metadata


if __name__=='__main__':build()
