"""Author the four capstone notebooks from verified scripts and aggregate receipts."""
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[2]
BOOT='''from pathlib import Path
import sys, json
import pandas as pd
import numpy as np
from IPython.display import display, Markdown, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "skills/README.md").exists())
sys.path.insert(0, str(ROOT / "work/scripts"))
import capstone_pipeline as cp
SUMMARY_PATH = ROOT / "work/outputs/capstone_metrics.json"
if not SUMMARY_PATH.exists():
    cp.execute()
s = json.loads(SUMMARY_PATH.read_text())
print("Dataset revision:", s["dataset_revision"])
print("Protocol:", s["protocol_version"], s["protocol_sha256"])
'''
META={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
def save(name,items):
 cells=[nb.v4.new_markdown_cell(t.strip()) if kind=='md' else nb.v4.new_code_cell(t.strip()) for kind,t in items]
 n=nb.v4.new_notebook(cells=cells,metadata=META)
 for c in cells:
  if c.cell_type=='code':compile(c.source,name,'exec')
 nb.validate(n);nb.write(n,ROOT/'work/notebooks'/f'{name}.ipynb')

save('w05_model',[
('md','''# ML-08 — A frozen tree for the refresh-review queue

Lane 2 remains locked. This notebook trains a small decision tree on real March warehouse features and compares it with the frozen Week 4 volume rule on exactly the same client split. It does not select parameters using June. See [the frozen protocol](../capstone_protocol.json) and [capstone pipeline](../scripts/capstone_pipeline.py).

Run from a cloned repo with `work/requirements-capstone.txt` installed and Hugging Face access. The final aggregate receipt is committed; the pipeline rebuilds it from pinned data. No private page text or raw query strings are used.'''),
('md','''## 1. Method choice and why

A maximum-depth-3 tree is readable and can represent simple interactions without an extensive parameter search. `min_samples_leaf=100` avoids tiny training leaves; seed 42 fixes randomness. These choices were frozen before June was accessed, rather than selected on final-test performance.

The five features are daily mean impressions, CTR percent, impression-weighted position, active-day share and daily impression CV, all from March 1–14. Nonpositive and sub-one daily positions are unknown; training-only median imputation handles missing position. This quality change follows the Week 4 review. No IDs, future counts, label proxies or product flags are features.

The label is a >20% drop in impressions between the equal-length March 1–14 and March 17–30 windows. It describes visibility movement, not editorial actionability or refresh benefit.'''),
('code',BOOT+'''
raw, source_audit = cp.load_month("2026-03")
all_pages, X_all = cp.prepare(raw, "2026-03")
known = all_pages.label_known
pages = all_pages.loc[known].reset_index(drop=True)
X = X_all.loc[known].reset_index(drop=True)
y = pages.declined.astype(int)
display(pd.DataFrame([source_audit]))
display(X.head(5))
print(f"{len(pages):,} labeled pages; {X.shape[1]} past-only features.")'''),
('md','''## 2. Split design

Use `GroupShuffleSplit(test_size=0.25, random_state=42)` by client, preserving the Week 4 cohort. Within the 24 development clients, five-fold GroupKFold provides an internal diagnostic. There is no grid search, refit on holdout clients or selection among many models. The March comparison has already been inspected; the separate June evaluation is documented in Week 6.'''),
('code','''from sklearn.model_selection import GroupShuffleSplit, GroupKFold
train, test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(X,y,groups=pages.client_hash_id))
assert set(pages.iloc[train].client_hash_id).isdisjoint(set(pages.iloc[test].client_hash_id))
baseline_receipt = json.loads((ROOT / "work/outputs/w04_baseline_metrics.json").read_text())
assert cp.fingerprint(pages.iloc[test]) == baseline_receipt["comparison_keys_sha256"]
assert cp.fingerprint(pages.iloc[train]) == baseline_receipt["development_keys_sha256"]
fold_results=[]
for fold,(a,b) in enumerate(GroupKFold(n_splits=5).split(X.iloc[train],y.iloc[train],pages.iloc[train].client_hash_id),1):
    tr,va=train[a],train[b]
    fitted=cp.make_model().fit(X.iloc[tr],y.iloc[tr])
    validation=pages.iloc[va].copy()
    validation["baseline_score"]=validation.past_impressions
    validation["model_score"]=fitted.predict_proba(X.iloc[va])[:,1]
    results=cp.comparison(validation)
    fold_results.append({"fold":fold,"validation_pages":len(va),"tree_auc":results["model"]["roc_auc"],
                         "tree_p20":results["model"]["precision_at_k"],"baseline_p20":results["baseline"]["precision_at_k"]})
display(pd.DataFrame(fold_results))
print(f"Train: {len(train):,} pages; comparison: {len(test):,}; no client overlap.")'''),
('md','''## 3. Train and compare with the frozen baseline

The baseline score is past impressions; the tree's score is its uncalibrated positive-leaf frequency. Both use the same eligibility and the same held-out page IDs. Ties are deterministic using pseudonymous keys, which are never learned features. The primary task metric is precision@20, with prevalence shown as the expected value for a random queue. ROC-AUC and average precision assess more of the ranking.'''),
('code','''model=cp.make_model().fit(X.iloc[train],y.iloc[train])
comparison=pages.iloc[test].copy()
comparison["baseline_score"]=comparison.past_impressions
comparison["model_score"]=model.predict_proba(X.iloc[test])[:,1]
results=cp.comparison(comparison)
display(pd.DataFrame(results).T)
assert results==s["march_comparison"]
print("Same split confirmed; March results match the aggregate capstone receipt.")'''),
('md','''## 4. Errors and interpretation

The tree improves March pooled precision@20 from 45% to 70%, but six of its first twenty do not decline. Its top twenty span only two clients, versus three for the baseline. These are proxy errors and concentration warnings; neither a positive nor a negative label establishes the correct editorial action.

Read the actual tree below. Its thresholds are associations learned from past measurements, not a reconstruction of Google's algorithm. June's permutation interpretation and adverse client-level results appear in the validation notebook and paper; no retraining follows them.'''),
('code','''from sklearn.tree import export_text
print(export_text(model[-1],feature_names=cp.FEATURES,decimals=3))
top=cp.ordered(comparison,"model_score").head(20)
display(pd.DataFrame({"measure":["Top-20 declines","Top-20 non-declines","Top-20 clients"],
                     "value":[int(top.declined.sum()),int((top.declined==0).sum()),top.client_hash_id.nunique()]}))
assert model.n_features_in_==5
assert list(X.columns)==cp.FEATURES'''),
('md','''## 5. Self-check

- [x] One fixed model, training-only imputation, no parameter search.
- [x] Five past-only features and a grouped split verified against Week 4 fingerprints.
- [x] Baseline, model and prevalence compared on identical pages.
- [x] Errors and client concentration inspected; no causal or editorial-success claims.
- [x] Notebook executed with outputs; code and aggregate receipts support reproduction.

AI assistance drafted and executed this notebook. The final June result is evaluated separately, not used to retune this model.''')])

save('w06_validation_audit',[
('md','''# ML-09 — Temporal validation and research claim audit

This capstone audits its own two measured findings. No external FlyRank research paper was supplied, so no finding is attributed to an unseen paper. The June evaluation, raw-data quality amendment and code were recorded in the repository. The original method was committed before the first June read.'''),
('md','''## 1. Two findings and the questions they raise

**Finding 1:** On the final June cohort, pooled precision@20 is 85% for the tree versus 75% for the volume baseline. Are these twenty pages independent, and is the model simply concentrating on particular clients?

**Finding 2:** The model performs worse within all five clients with more than twenty eligible pages. Does a global ranking metric support the operational decision if review time is allocated by client?

The proxy comes from later impressions, not an editor's decision or a refresh experiment. Both findings concern visibility decline only. The sixth client has seventeen pages, so selecting all seventeen cannot test ranking quality.'''),
('code',BOOT+'''
display(pd.DataFrame(s["june_primary"]).T)
display(pd.DataFrame(s["per_client_primary"]))'''),
('md','''## 2. Before/after: reused March comparison to June final evaluation

The model is fixed. March's comparison clients were held out of training but their results had been seen in previous weeks. June's primary cohort uses only those untrained clients at a later decision date. Six of the original eight have eligible labeled pages. No June label informed fitting, feature selection or parameter tuning.

A data-only amendment removed 6,390 exact duplicate June page-day copies after the grain check failed and before model scoring. No conflicting records were collapsed. This change is explicitly documented, rather than hidden as a perfect preregistered data pipeline.

The 500-resample paired client bootstrap reranks the global top twenty within each resampled client pool. Its -60 to +20 percentage-point interval is descriptive and wide; six clusters do not justify a precise generalization claim.'''),
('code','''rows=[]
for cohort in ["march_comparison","june_primary","june_secondary_all_clients"]:
    for name,metrics in s[cohort].items():rows.append({"cohort":cohort,"method":name,**metrics})
display(pd.DataFrame(rows))
display(pd.DataFrame([s["bootstrap"]]))
display(pd.DataFrame([s["march_source"],s["june_source"]]))
print("Data-only amendment:")
display(json.loads((ROOT/"work/capstone_data_amendment.json").read_text()))'''),
('md','''## 3. Leakage audit and feature stability

The executable check below mutates all future-window counts and coverage in a copy of June's aggregated data. Past eligibility and all five feature values must remain unchanged. The target and label availability may change, but those outputs are never model inputs.

Additional safeguards: disjoint March training/final clients, train-only imputation, no hash IDs or target-derived columns in the feature list, no fixed-window query table, and June held out until the model was fixed. These checks establish code boundaries; they cannot establish when upstream source values first became available.'''),
('code','''raw, _ = cp.load_month("2026-06")
original, X = cp.prepare(raw,"2026-06")
perturbed=raw.copy()
perturbed["future_impressions"]=0
perturbed["future_valid_days"]=0
changed, X_changed=cp.prepare(perturbed,"2026-06")
pd.testing.assert_frame_equal(X,X_changed)
pd.testing.assert_frame_equal(original[cp.KEYS],changed[cp.KEYS])
assert changed.declined.isna().all()
assert all(s["checks"].values())
assert not any("future" in f or "declin" in f or "hash" in f for f in cp.FEATURES)
print("PASS: changing outcomes does not change score inputs or past eligibility.")
display(pd.DataFrame(s["feature_summary"]))
display(pd.DataFrame(s["permutation_interpretation"]))'''),
('md','''## 4. Rewrite the claim

**Too strong:** “The model improves content recommendations and will recover lost traffic.”

**Supported:** “In a frozen retrospective June test on six untrained clients, the model found two more declining pages in a pooled top twenty than the volume baseline. Its within-client rankings were worse in every rankable client, its AUC was lower, and the client-bootstrap interval was wide. The result does not establish editorial usefulness or causal refresh benefit.”

We would not broadly deploy the tree based on this evaluation. Collect independent editorial judgments and prospectively evaluate the actual review-allocation policy first. Preserve the frozen results as a negative/qualified finding rather than tuning the June test into a success.'''),
('code','''groups=pd.DataFrame(s["per_client_primary"])
rankable=groups.loc[groups.n.gt(groups.k)]
assert (rankable.model_p_at_k < rankable.baseline_p_at_k).all()
print(f"Model loses within all {len(rankable)} rankable clients.")
print("Unknown final outcomes:",s["june_primary_unlabeled_pages"])
print("No claim of causal refresh impact or confirmed editorial value.")'''),
('md','''## 5. Self-check

- [x] Distinguishes reused March evidence from frozen June evaluation.
- [x] Reports identical-cohort baselines, prevalence, adverse client results and uncertainty.
- [x] Documents the data-quality amendment before outcome scoring.
- [x] Executes an outcome-mutation leakage check.
- [x] Rewrites claims around the observed limits; no external paper findings invented.
- [x] Executed outputs saved and backed by the committed aggregate receipt.''')])

save('w07_action_playbook',[
('md','''# ML-10 — Historical content inspection playbook

The analytical output is a June 17, 2026 queue, not current instructions. It connects a visibility-risk ranking to explicit human inspection, while retaining the negative validation findings.'''),
('md','''## 1. Ranked actions and reason codes

The predeclared global precision criterion selects the model for a historical illustration, but the within-client losses argue against broad operational promotion. Every recommendation is **inspect before refresh**. Reasons distinguish variable visibility from general visibility context. No reason code authorizes automatic rewriting, pruning or merging.

The full queue retains pseudonymous routing keys locally. The public top-ten output below strips those keys and raw measurements; authorized users can join the rank back to the locally regenerated CSV.'''),
('code',BOOT+'''
queue_path=ROOT/"work/outputs/capstone_recommendations.csv"
if not queue_path.exists(): cp.execute()
queue=pd.read_csv(queue_path)
assert len(queue)==s["historical_queue_rows"]
assert not any("future" in c or "declined" in c for c in queue.columns)
display(pd.DataFrame(s["public_recommendations"])[["rank","candidate","action","reason_code","model_score","wrong_if"]])
print(f"Local historical queue: {len(queue):,} pages; public display: ten anonymous review slots.")'''),
('md','''## 2. Intended use and limits

A content editor first verifies tracking, query intent, seasonality and technical health, then inspects for outdated facts or coverage gaps. A refresh is appropriate only if that independent review finds a content issue. Otherwise monitor or route a technical investigation.

Leaf scores are uncalibrated training frequencies; neither they nor the decline proxy are probabilities that editing will help. Tied scores and repeated reason codes are expected from a shallow tree. The evaluation does not justify deploying this model across clients; the historical queue is an inspectable research artifact.'''),
('code','''print("Illustrated recommender:",s["selected_recommender"])
print("Queue date:",queue.decision_date.unique().tolist())
print("Unknown outcomes retained:",s["june_primary_unlabeled_pages"])
print("Distinct tree scores in queue:",queue.model_score.nunique())
print("Model's June global top-20 clients:",s["june_primary"]["model"]["top_k_clients"])'''),
('md','''## 3. Human review and the no-go list

For every slot: verify measurement coverage; inspect the page and search context; identify a concrete issue; record the decision and rationale. A volatile pattern can reflect demand or instrumentation, and a stable pattern can belong to a healthy page.

Do not automatically rewrite, merge, prune, redirect or delete. Do not infer private identities from hashes. Do not count a missing editorial judgment as a negative. Do not claim ranking-algorithm discovery or causal refresh recovery. These ten notes are AI-assisted metric-based hypotheses; no human page-content inspection is claimed.'''),
('code','''for r in s["public_recommendations"]:
    display(Markdown(f"**{r['rank']}. {r['candidate']} — {r['action']}.** {r['why']} "
                     f"Wrong if: {r['wrong_if']} Confidence: low; independent editorial review required."))
assert len(s["public_recommendations"])==10'''),
('md','''## 4. Monitoring and reassessment triggers

For a future pilot, recompute source grain and availability each run and stop on conflicting duplicates, missing keys or invalid time windows. Track feature drift and client representation, collect blind editorial judgments for both methods' review queues, and report editorial precision@20 with missing-label counts.

As a proposed policy, pause promotion if paired review cycles show the model below the baseline, if any targeted client repeatedly loses, or if input coverage changes materially. Review after each cycle; investigate and preregister changes rather than automatically retraining. No production monitor or retraining service is implemented in this research repo.'''),
('code','''monitor_plan=pd.DataFrame([
 {"check":"Source grain and availability","cadence":"Every data load","response":"Stop on conflicts or unavailable evidence"},
 {"check":"Client representation and feature drift","cadence":"Every review cycle","response":"Investigate distribution changes before use"},
 {"check":"Paired editorial precision@20","cadence":"After complete blind review labels","response":"Do not promote a model that trails its comparator"},
 {"check":"Client-specific review value","cadence":"Each cycle, with enough reviewed cases","response":"Pause for repeatedly harmed clients"},
])
display(monitor_plan)'''),
('md','''## 5. Exports for the paper

The full CSV remains ignored. The committed aggregate metrics and public recommendation summaries feed the paper's charts and top-ten playbook. All chart and page content is generated from the same receipt.'''),
('code','''from build_research_paper import build
publication=build()
display(publication)
assert (ROOT/"work/paper/index.html").exists()
print("Paper assets regenerated; raw queue stays outside Git.")'''),
('md','''## 6. Self-check

- [x] Ranked historical actions, reason codes and ten wrong-if notes are visible.
- [x] Intended user and concrete action are explicit, with a human gate before editing.
- [x] Adverse validation and uncertain editorial value remain visible.
- [x] Proposed monitoring is labeled as a policy, not an implemented service.
- [x] Public artifacts are generated without raw exports or client identities.
- [x] Notebook executed with saved outputs.''')])

save('capstone',[
('md','''# Does a higher top-20 score mean a better content queue?

**Lane 2 — Refresh / Content Opportunity Scoring.** This executable companion mirrors the nine-section public paper. It reruns the frozen pipeline, writes aggregate receipts, regenerates the historical local queue and builds the static paper. Reproduction does not retune the model after June.

**Five-sentence abstract:** Can a small search-data model help an editor choose pages to inspect for a possible refresh? We trained on 14,137 March pages from 24 clients and tested later outcomes from untrained clients. A frozen depth-three tree used five past-window features and competed with a volume-only rule on the same eligibility and decline proxy. On 35,588 June pages across six held-out clients, it found 17 declines in its top twenty versus 15 for the baseline, but lost within every client with more than twenty eligible pages. The result supports a guarded historical inspection exercise, while broad deployment and refresh-benefit claims remain unsupported.'''),
('md','''## 1. Question and decision

Does a simple learned ranking improve a twenty-slot content-review queue over a transparent exposure rule, and does any improvement hold within clients? The owner is an editor; the action is inspection before deciding whether content needs a refresh. A false positive wastes a slot or prompts an unnecessary edit; a false negative delays attention. The model estimates a traffic-decline proxy, not the effect of editing.'''),
('code',BOOT+'''
# Full rerun from pinned monthly data/caches; no hyperparameter selection.
s=cp.execute()
print("Historical decision date: 2026-06-17; primary metric: decline-proxy precision@20.")'''),
('md','''## 2. Data

Use only March and June partitions of the pinned FlyRank warehouse release. One source row is a page-day and one scored row is a page at a decision date. Features cover days 1–14; days 15–16 are a reporting buffer; outcomes cover days 17–30. Require GSC availability IS TRUE, valid counts, complete fourteen-day past coverage and at least 100 past impressions. Complete later coverage is needed only to observe the label.

The June grain check stopped before scoring. Exactly 6,390 identical duplicate copies were removed under a committed data-only amendment; conflicts would still fail. Daily position below 1 is treated as missing, following Week 4's review. Private identities, URLs, queries, product flags and overlapping query-table data are excluded.'''),
('code','''display(pd.DataFrame([s["march_source"],s["june_source"]]))
display(pd.DataFrame({"cohort":["March training","March reused comparison","June final client cohort"],
    "pages":[s["train_pages"],s["march_comparison_pages"],s["june_primary"]["model"]["n"]],
    "clients":[s["train_clients"],s["march_comparison_clients"],s["june_primary"]["model"]["clients"]]}))'''),
('md','''## 3. Methodology

The frozen tree has depth 3, minimum leaf size 100, and seed 42. Its inputs are past exposure, CTR, impression-weighted position, active-day share and daily variability. A train-only median imputer handles missing positions. The proxy is later impressions <80% of past impressions. The baseline remains Week 4's volume-first rule.

Five-fold client-grouped validation uses March development clients. The reused March comparison is reported transparently. June's primary test uses the same held-out client identities at a later date; no June outcomes inform fitting or selection. A paired client bootstrap, per-client rankings and secondary all-client June evaluation qualify the pooled result. Source code and the pre-June protocol commit make this boundary auditable.'''),
('code','''display(json.loads((ROOT/"work/capstone_protocol.json").read_text()))
assert all(s["checks"].values())
display(pd.DataFrame(s["feature_summary"]))
print((ROOT/"work/outputs/capstone_tree.txt").read_text())'''),
('md','''## 4. Results against the baseline

The final pooled result is 85% precision@20 versus 75%, with 58.90% decline prevalence. Yet the model's AUC is lower (0.575 vs 0.599) and it ranks worse inside every client with more than twenty pages. The all-client June comparison also favors the baseline at top twenty. The -60 to +20 percentage-point client-bootstrap interval is wide; this is a qualified finding, not an established operational win.'''),
('code','''rows=[]
for cohort in ["march_comparison","june_primary","june_secondary_all_clients"]:
    for name,result in s[cohort].items():rows.append({"cohort":cohort,"method":name,**result})
display(pd.DataFrame(rows))
display(pd.DataFrame(s["per_client_primary"]))
display(pd.DataFrame([s["bootstrap"]]))'''),
('md','''## 5. Limitations and honest framing

Visibility decline is not refresh benefit. Six final client clusters, retrospective coverage selection, missing ingestion timestamps, tied leaf scores, uninspected page context and one later month limit the claim. Historical June recommendations are not current content instructions. Broad deployment is not justified by the global metric gain because the client-level audit is adverse.'''),
('code','''print("June unknown outcomes retained in queue:",s["june_primary_unlabeled_pages"])
g=pd.DataFrame(s["per_client_primary"])
rankable=g[g.n>g.k]
assert (rankable.model_p_at_k < rankable.baseline_p_at_k).all()
print("Rankable clients with worse model precision:",len(rankable))'''),
('md','''## 6. Ranked recommendations

Show the ten highest-ranked historical review slots from the selected model illustration. Full pseudonymous keys remain only in the ignored CSV; public slots omit raw measurements and identifiers. Every action is inspection before editing, with low editorial confidence. Keep the baseline as the operational reference until a prospectively evaluated client-allocation policy and independent editorial judgments justify a change.'''),
('code','''display(pd.DataFrame(s["public_recommendations"])[["rank","candidate","action","reason_code","wrong_if","confidence"]])
print("Historical full queue rows:",s["historical_queue_rows"])'''),
('md','''## 7. Paper artifacts and reproducibility

The build script produces the nine required sections, three reproducible charts and a machine-readable publication fingerprint. Install `work/requirements-capstone.txt`, authenticate to the gated warehouse, run `python work/scripts/capstone_pipeline.py`, then `python work/scripts/build_research_paper.py` and the notebook runner. The repo includes pinned source revision, seeds, protocol, amendment, code and aggregate receipts.'''),
('code','''from build_research_paper import build
publication=build()
for name in ["top20-comparison","client-level-lift","feature-interpretation"]:
    display(Image(filename=str(ROOT/"work/figures"/(name+".png"))))
display(publication)'''),
('md','''## 8. Acknowledgments & data credit

[Built on the FlyRank ML Internship dataset](https://flyrank.ai).

Thanks to the internship team for the release and starter materials. AI assistance supported code, checks, charts and drafting. No independent human content review or causal refresh experiment is claimed.

## 9. Demo and audience summaries

**Five-minute demo:** 0:00–0:45 explain the editor's twenty-slot decision; 0:45–1:30 show the two date windows and grouped split; 1:30–2:30 compare the frozen baseline and tree; 2:30–3:30 reveal the adverse client-level audit and uncertainty; 3:30–4:30 walk through one historical recommendation and its wrong-if condition; 4:30–5:00 open the reproducibility links and state what the work cannot claim.

**Social-post cut:** A global metric can hide worse client-level recommendations. My frozen FlyRank search-data tree found 17 declines in a June top twenty versus a volume rule's 15—but ranked worse within every rankable client. The public paper shows the grouped/time-separated method, uncertainty and complete code; no refresh-benefit claim. Share the verified URL from `submission/paper_url.txt` once deployed.

**Employer summary:** I built a reproducible search-content inspection pipeline on real FlyRank warehouse data, including a frozen baseline, a small tree and a public research paper. I trained on 14,137 March pages from 24 clients and evaluated a later 35,588-page cohort from six untrained clients. The analysis exposed a global top-twenty gain that failed client-level checks, demonstrating why operational validation matters more than one favorable headline metric.

## 10. Self-check

- [x] Repeatable real-data pipeline, frozen baseline, grouped and later-window evaluation.
- [x] Results, uncertainty, adverse checks, ranked inspection playbook and reproducibility links.
- [x] Executed notebook outputs and three generated research charts.
- [x] Nine-section paper with five-sentence abstract and linked FlyRank credit.
- [x] Demo outline, social cut and three-sentence employer summary.
- [ ] Public deployment verified and its exact URL written to `submission/paper_url.txt`.

The publishing script checks the deployed publication fingerprint before recording the mandatory URL. A local build alone is not a live publication.''')])
print('Authored w05, w06, w07 and capstone notebooks.')
