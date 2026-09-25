"""Repeatable, public-safe capstone analysis. Raw/page-level outputs remain ignored."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform

import duckdb
import numpy as np
import pandas as pd
from huggingface_hub import get_token, hf_hub_download
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work/outputs'
CACHE = OUT / 'capstone_cache'
HF_CACHE = OUT / 'hf_cache'
PROTOCOL_FILE = ROOT / 'work/capstone_protocol.json'
PROTOCOL = json.loads(PROTOCOL_FILE.read_text())
FEATURES = PROTOCOL['features']
KEYS = ['client_hash_id', 'content_hash_id']
REPO_ID = 'FlyRank/internship-warehouse'
REVISION = PROTOCOL['dataset_revision']
SEED = 42
K = 20


def fingerprint(frame):
    rows = frame[KEYS].sort_values(KEYS).itertuples(index=False, name=None)
    return hashlib.sha256('\n'.join('|'.join(map(str, row)) for row in rows).encode()).hexdigest()


def load_month(month):
    """One pinned month only; aggregate locally and cache derived page-level data."""
    CACHE.mkdir(parents=True, exist_ok=True)
    sql = (ROOT / 'work/scripts/capstone_features.sql').read_text().format(month=month)
    key = hashlib.sha256(('exact_dedup_v1' + REVISION + sql).encode()).hexdigest()[:16]
    cache_file = CACHE / f'{month}-{key}.parquet'
    audit_file = CACHE / f'{month}-{key}.json'
    if cache_file.exists() and audit_file.exists():
        return pd.read_parquet(cache_file), json.loads(audit_file.read_text())
    partition = f'fact_content_daily_performance/month={month}/data_0.parquet'
    try:
        file = hf_hub_download(REPO_ID, partition, repo_type='dataset', revision=REVISION,
                               cache_dir=HF_CACHE, local_files_only=True)
    except Exception:
        token = os.environ.get('HF_TOKEN') or get_token()
        if not token:
            try:
                from google.colab import userdata
                token = userdata.get('HF_TOKEN')
            except Exception:
                pass
        if not token:
            raise RuntimeError('Hugging Face login required. Use saved login or HF_TOKEN, never notebook literals.') from None
        try:
            file = hf_hub_download(REPO_ID, partition, repo_type='dataset', revision=REVISION,
                                   cache_dir=HF_CACHE, token=token)
        except Exception:
            raise RuntimeError('Cannot read the gated monthly partition. Check access and network.') from None
    con = duckdb.connect()
    con.execute("SET threads = 2")
    con.execute("SET memory_limit = '2GB'")
    columns = 'report_date, client_hash_id, content_hash_id, gsc_data_available, gsc_impressions, gsc_clicks, gsc_avg_position'
    con.read_parquet(file).project(columns).create_view('march')  # SQL relation name, not a date filter.
    audit = con.sql('''SELECT COUNT(*) AS source_rows, MIN(report_date) AS first_date,
        MAX(report_date) AS last_date, COUNT(DISTINCT client_hash_id) AS clients,
        COUNT(*) FILTER (WHERE gsc_data_available IS TRUE) AS gsc_available_rows,
        COUNT(*) FILTER (WHERE gsc_data_available IS FALSE) AS gsc_false_rows,
        COUNT(*) FILTER (WHERE gsc_data_available IS NULL) AS gsc_null_rows,
        COUNT(*) FILTER (WHERE gsc_data_available IS TRUE AND gsc_avg_position > 0
                          AND gsc_avg_position < 1) AS subone_position_rows
        FROM march''').df().iloc[0].to_dict()
    violations = con.sql('''SELECT COUNT(*) FROM (
        SELECT report_date, client_hash_id, content_hash_id FROM march
        GROUP BY ALL HAVING COUNT(*) > 1 OR report_date IS NULL
            OR client_hash_id IS NULL OR content_hash_id IS NULL)''').fetchone()[0]
    # Data-only amendment: June has duplicate keys, but all used fields are identical.
    # No outcome metric had been computed when this defect was found.
    if violations:
        con.execute("CREATE TEMP TABLE duplicate_keys AS SELECT report_date, client_hash_id, content_hash_id, COUNT(*) AS n FROM march GROUP BY 1,2,3 HAVING COUNT(*) > 1")
        conflicts = con.sql("""SELECT COUNT(*) FROM (
            SELECT m.report_date, m.client_hash_id, m.content_hash_id
            FROM march m JOIN duplicate_keys USING (report_date, client_hash_id, content_hash_id)
            GROUP BY 1,2,3
            HAVING COUNT(DISTINCT ROW(gsc_data_available, gsc_impressions, gsc_clicks, gsc_avg_position)) > 1
        )""").fetchone()[0]
        null_keys = con.sql("SELECT COUNT(*) FROM march WHERE report_date IS NULL OR client_hash_id IS NULL OR content_hash_id IS NULL").fetchone()[0]
        assert conflicts == 0 and null_keys == 0, 'Conflicting duplicates or null keys require investigation.'
        removed = int(con.sql("SELECT SUM(n-1) FROM duplicate_keys").fetchone()[0])
        con.read_parquet(file).project(columns).create_view('source_raw')
        con.sql('SELECT DISTINCT * FROM source_raw').create_view('march', replace=True)
        audit.update(duplicate_keys=int(violations), identical_extra_rows_removed=removed, conflicting_keys=0)
    else:
        audit.update(duplicate_keys=0, identical_extra_rows_removed=0, conflicting_keys=0)
    clean_violations = con.sql("SELECT COUNT(*) FROM (SELECT report_date,client_hash_id,content_hash_id FROM march GROUP BY 1,2,3 HAVING COUNT(*)>1)").fetchone()[0]
    assert clean_violations == 0, 'Grain still invalid after exact deduplication'
    audit['clean_page_day_rows'] = int(con.sql('SELECT COUNT(*) FROM march').fetchone()[0])
    assert str(audit['first_date'])[:7] == str(audit['last_date'])[:7] == month
    page = con.sql(sql).df().sort_values(KEYS).reset_index(drop=True)
    con.close()
    audit = {k: str(v)[:10] if k.endswith('date') else int(v) for k, v in audit.items()}
    audit.update(partition=partition, source_grain_violations_before_cleaning=violations, source_grain_violations_after_cleaning=clean_violations, aggregate_rows=len(page))
    page.to_parquet(cache_file, index=False)
    audit_file.write_text(json.dumps(audit, indent=2) + '\n')
    return page, audit


def prepare(page, month):
    page = page.loc[page.past_valid_days.eq(14) & page.past_impressions.ge(100)].copy()
    page = page.sort_values(KEYS).reset_index(drop=True)
    assert not page.duplicated(KEYS).any()
    assert pd.to_datetime(page.past_last).max() < pd.Timestamp(f'{month}-17')
    X = pd.DataFrame({
        'mean_daily_impressions': page.past_impressions / 14,
        'ctr_pct': 100 * page.past_clicks / page.past_impressions,
        'impression_weighted_position': page.weighted_position_sum / page.position_impressions.replace(0, np.nan),
        'active_day_share': page.past_active_days / 14,
        'daily_impression_cv': page.past_impression_sd / (page.past_impressions / 14),
    })[FEATURES]
    assert len(FEATURES) == 5 and not np.isinf(X.to_numpy()).any()
    assert not any('future' in col or 'declin' in col or 'hash' in col for col in X)
    page['label_known'] = page.future_valid_days.eq(14)
    page['declined'] = pd.Series(pd.NA, index=page.index, dtype='Int64')
    known = page.label_known
    page.loc[known, 'declined'] = (page.loc[known, 'future_impressions'] < .8 * page.loc[known, 'past_impressions']).astype(int)
    assert page.loc[~known, 'declined'].isna().all()
    page['decision_date'] = f'{month}-17'
    return page, X


def make_model():
    return make_pipeline(SimpleImputer(strategy='median'),
                         DecisionTreeClassifier(max_depth=3, min_samples_leaf=100, random_state=SEED))


def ordered(frame, score):
    return frame.sort_values([score, *KEYS], ascending=[False, True, True], kind='stable')


def scores(frame, score, k=K):
    assert frame.declined.notna().all() and len(frame) >= k
    y = frame.declined.astype(int)
    top = ordered(frame, score).head(k)
    return {'n': len(frame), 'clients': int(frame.client_hash_id.nunique()),
            'base_rate': float(y.mean()), 'k': k, 'top_k_declines': int(top.declined.sum()),
            'precision_at_k': float(top.declined.mean()),
            'roc_auc': float(roc_auc_score(y, frame[score])) if y.nunique() == 2 else None,
            'average_precision': float(average_precision_score(y, frame[score])) if y.nunique() == 2 else None,
            'top_k_clients': int(top.client_hash_id.nunique())}


def comparison(frame):
    return {name: scores(frame, col) for name, col in [('baseline', 'baseline_score'), ('model', 'model_score')]}


def boot_lift(frame, repeats=500):
    rng = np.random.default_rng(SEED)
    groups = [g.copy() for _, g in frame.groupby('client_hash_id', sort=True)]
    lifts = []
    for _ in range(repeats):
        sample = pd.concat([groups[i] for i in rng.integers(len(groups), size=len(groups))], ignore_index=True)
        lifts.append(float(ordered(sample, 'model_score').head(K).declined.mean()
                           - ordered(sample, 'baseline_score').head(K).declined.mean()))
    return {'method': 'paired client bootstrap; rerank global top20; no model refitting',
            'resamples': repeats, 'clusters': len(groups),
            'lift_interval_95': [float(v) for v in np.quantile(lifts, [.025, .975])],
            'caution': 'Descriptive percentile interval with few client clusters; not proof of significance.'}


def execute():
    print('Build March frame and verify frozen baseline split.', flush=True)
    raw, march_audit = load_month('2026-03')
    march_all, Xm_all = prepare(raw, '2026-03')
    march = march_all.loc[march_all.label_known].copy().reset_index(drop=True)
    Xm = Xm_all.loc[march_all.label_known].reset_index(drop=True)
    y = march.declined.astype(int)
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED)
                       .split(Xm, y, groups=march.client_hash_id))
    train_clients = set(march.iloc[train].client_hash_id)
    holdout_clients = set(march.iloc[test].client_hash_id)
    assert train_clients.isdisjoint(holdout_clients)
    baseline = json.loads((OUT / 'w04_baseline_metrics.json').read_text())
    assert fingerprint(march.iloc[test]) == baseline['comparison_keys_sha256']
    assert fingerprint(march.iloc[train]) == baseline['development_keys_sha256']
    assert Xm.iloc[train].notna().any().all()
    folds = []
    for fold, (a, b) in enumerate(GroupKFold(n_splits=5).split(Xm.iloc[train], y.iloc[train], march.iloc[train].client_hash_id), 1):
        tr, va = train[a], train[b]
        assert set(march.iloc[tr].client_hash_id).isdisjoint(set(march.iloc[va].client_hash_id))
        model = make_model().fit(Xm.iloc[tr], y.iloc[tr])
        f = march.iloc[va].copy()
        f['baseline_score'] = f.past_impressions
        f['model_score'] = model.predict_proba(Xm.iloc[va])[:, 1]
        folds.append({'fold': fold, **comparison(f)})
    model = make_model().fit(Xm.iloc[train], y.iloc[train])
    march['baseline_score'] = march.past_impressions
    march['model_score'] = model.predict_proba(Xm)[:, 1]
    march_result = comparison(march.iloc[test])
    assert march_result['baseline']['precision_at_k'] == baseline['proxy_precision_at_k']
    tree_text = export_text(model[-1], feature_names=FEATURES, decimals=3)
    (OUT / 'capstone_tree.txt').write_text(tree_text)
    print('March method fixed. Read final June partition; no tuning after this point.', flush=True)
    raw_june, june_audit = load_month('2026-06')
    june, Xj = prepare(raw_june, '2026-06')
    june['baseline_score'] = june.past_impressions
    june['model_score'] = model.predict_proba(Xj)[:, 1]
    final_mask = june.label_known & june.client_hash_id.isin(holdout_clients)
    final = june.loc[final_mask].copy()
    assert len(final) >= K and not set(final.client_hash_id) & train_clients
    assert final.declined.nunique() == 2
    final_result = comparison(final)
    secondary_result = comparison(june.loc[june.label_known])
    delta = final_result['model']['precision_at_k'] - final_result['baseline']['precision_at_k']
    selected = 'model' if delta > 0 else 'baseline'
    primary_bootstrap = boot_lift(final)
    per_client = []
    for i, (_, group) in enumerate(final.groupby('client_hash_id', sort=True), 1):
        k = min(K, len(group))
        a, b = scores(group, 'baseline_score', k), scores(group, 'model_score', k)
        per_client.append({'anonymous_client': f'Group {i}', 'n': len(group), 'k': k,
                           'base_rate': a['base_rate'], 'baseline_p_at_k': a['precision_at_k'],
                           'model_p_at_k': b['precision_at_k']})
    # Permutation importance is interpretation of the frozen model, never used to refit/select.
    auc_original = final_result['model']['roc_auc']
    rng = np.random.default_rng(SEED)
    interpretation = []
    X_final = Xj.loc[final_mask].copy()
    for feature in FEATURES:
        drops = []
        for _ in range(5):
            shuffled = X_final.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            drops.append(auc_original - roc_auc_score(final.declined.astype(int), model.predict_proba(shuffled)[:, 1]))
        interpretation.append({'feature': feature, 'mean_auc_drop': float(np.mean(drops)), 'std_auc_drop': float(np.std(drops))})
    queue_mask = june.client_hash_id.isin(holdout_clients)
    context_columns = KEYS + ['decision_date', 'past_impressions', 'past_clicks', 'model_score', 'baseline_score']
    queue = june.loc[queue_mask, context_columns].copy()
    queue['daily_impression_cv'] = Xj.loc[queue_mask, 'daily_impression_cv']
    queue['position'] = Xj.loc[queue_mask, 'impression_weighted_position']
    queue['reason_code'] = np.where(queue.daily_impression_cv.ge(1), 'inspect_variable_visibility', 'inspect_visibility_context')
    queue['action_label'] = 'inspect_before_refresh'
    queue = ordered(queue, selected + '_score').reset_index(drop=True)
    queue.insert(0, 'rank', np.arange(1, len(queue) + 1))
    queue.to_csv(OUT / 'capstone_recommendations.csv', index=False, float_format='%.10g')
    public_recommendations = []
    for row in queue.head(10).itertuples(index=False):
        public_recommendations.append({
            'rank': row.rank, 'candidate': f'Review {row.rank:02d}',
            'action': 'Inspect before refreshing', 'reason_code': row.reason_code,
            'model_score': round(row.model_score, 3), 'score_caution': 'Uncalibrated leaf frequency, not probability of refresh benefit',
            'why': ('Variable past visibility; inspect demand changes and measurement coverage.' if row.daily_impression_cv >= 1
                    else 'Ranked by the selected method; inspect intent, missing coverage and technical context.'),
            'wrong_if': ('A temporary demand spike or measurement change explains the pattern.' if row.daily_impression_cv >= 1
                         else 'The page already meets intent, or visibility loss has a technical or seasonal cause.'),
            'confidence': 'Low editorial confidence; page text and query intent were not inspected.'})
    march.to_parquet(CACHE / 'march_predictions.parquet', index=False)
    june.to_parquet(CACHE / 'june_predictions.parquet', index=False)
    # A benchmark for each missingness/distribution check, with no page-level data in the public receipt.
    feature_summary = []
    for feature in FEATURES:
        feature_summary.append({'feature': feature, 'train_median': float(Xm.iloc[train][feature].median()),
                                'final_median': float(X_final[feature].median()),
                                'train_missing_fraction': float(Xm.iloc[train][feature].isna().mean()),
                                'final_missing_fraction': float(X_final[feature].isna().mean())})
    final_top = ordered(final, 'model_score').head(K)
    summary = {
        'protocol_version': PROTOCOL['protocol_version'],
        'protocol_sha256': hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest(),
        'dataset_revision': REVISION, 'march_source': march_audit, 'june_source': june_audit,
        'march_eligible_pages': len(march_all), 'march_labeled_pages': len(march),
        'train_pages': len(train), 'train_clients': len(train_clients),
        'march_comparison_pages': len(test), 'march_comparison_clients': len(holdout_clients),
        'march_comparison_keys_sha256': fingerprint(march.iloc[test]),
        'june_eligible_pages': len(june), 'june_labeled_pages': int(june.label_known.sum()),
        'june_primary_keys_sha256': fingerprint(final),
        'june_primary_unlabeled_pages': int((queue_mask & ~june.label_known).sum()),
        'features': FEATURES, 'cross_validation': folds, 'march_comparison': march_result,
        'june_primary': final_result, 'june_secondary_all_clients': secondary_result,
        'primary_precision_lift_absolute': delta, 'bootstrap': primary_bootstrap,
        'per_client_primary': per_client, 'permutation_interpretation': interpretation,
        'feature_summary': feature_summary,
        'model_top20_non_declines': int((final_top.declined == 0).sum()),
        'selected_recommender': selected, 'historical_queue_rows': len(queue),
        'public_recommendations': public_recommendations,
        'checks': {'disjoint_training_and_primary_clients': True, 'baseline_split_matches_week4': True,
                   'feature_window_precedes_outcome': True, 'score_features_exclude_ids_and_labels': True,
                   'unknown_outcomes_not_negative': True, 'subone_position_values_excluded': True},
        'versions': {p: importlib.metadata.version(p) for p in ['pandas', 'numpy', 'duckdb', 'scikit-learn', 'huggingface_hub']},
        'python': platform.python_version(),
        'scope': 'Historical March/June 2026 analysis, not current live recommendations; no causal effect claims.'}
    (OUT / 'capstone_metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'march': march_result, 'june_primary': final_result,
                      'precision_lift': delta, 'bootstrap': primary_bootstrap, 'selected': selected}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    execute()
