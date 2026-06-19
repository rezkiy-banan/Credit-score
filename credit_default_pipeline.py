# -------------------------------------------------------------------------
# Прогнозирование выхода клиента в дефолт по истории кредитных продуктов
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# 0. Настройка окружения
# -------------------------------------------------------------------------

import os, gc, time, warnings, random
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams['figure.dpi'] = 110

def _try(name):
    try:
        __import__(name); return True
    except Exception:
        return False

import joblib

HAS_LGB  = _try('lightgbm')
HAS_XGB  = _try('xgboost')
HAS_CAT  = _try('catboost')
HAS_OPT  = _try('optuna')
HAS_TORCH= _try('torch')
HAS_POLARS=_try('polars')

if HAS_LGB:  import lightgbm as lgb
if HAS_XGB:  import xgboost as xgb
if HAS_CAT:  from catboost import CatBoostClassifier
if HAS_OPT:  import optuna
if HAS_POLARS: import polars as pl
if HAS_TORCH:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

print('pandas', pd.__version__, '| numpy', np.__version__)
print('LightGBM:', HAS_LGB, '| XGBoost:', HAS_XGB, '| CatBoost:', HAS_CAT,
      '| Optuna:', HAS_OPT, '| PyTorch:', HAS_TORCH, '| Polars:', HAS_POLARS)
if HAS_TORCH:
    print('CUDA доступна:', torch.cuda.is_available(),
          '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))

class CFG:
    DATA_DIR        = '.'
    TRAIN_DATA      = 'train_data.parquet'
    TEST_DATA       = 'test_data.parquet'
    TRAIN_TARGET    = 'train_target.csv'
    SAMPLE_SUB      = 'sample_submission.csv'
    SUB_OUT         = 'submission.csv'

    ID_COL          = 'id'
    SEQ_COL         = 'rn'
    TARGET          = 'flag'

    SEED            = 42
    N_FOLDS         = 5

    ARTIFACTS_DIR   = 'artifacts'
    USE_CACHE       = True
    USE_POLARS      = False

    CORR_THRESHOLD  = 0.97
    IMP_TOP_K       = None

    RUN_NN          = True
    RUN_OPTUNA      = True
    OPTUNA_TRIALS   = 20

    NN_ENCODER      = 'gru'
    NN_MAX_LEN      = None
    NN_EPOCHS       = 8
    NN_BATCH        = 2048
    NN_LR           = 2e-3
    NN_HIDDEN       = 128

def seed_everything(seed=CFG.SEED):
    random.seed(seed); np.random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    if HAS_TORCH:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_everything()
DEVICE = 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu'
path = lambda f: os.path.join(CFG.DATA_DIR, f)
print('Готово. Устройство для нейросети:', DEVICE)

# -------------------------------------------------------------------------
# Чекпойнты (возобновляемость)
# -------------------------------------------------------------------------

ART = CFG.ARTIFACTS_DIR
os.makedirs(ART, exist_ok=True)
apath = lambda name: os.path.join(ART, name)

def load_or(name, compute, force=False):
    p = apath(name)
    if CFG.USE_CACHE and (not force) and os.path.exists(p):
        print(f'[cache] загружено: {name}')
        return joblib.load(p)
    obj = compute()
    if CFG.USE_CACHE:
        joblib.dump(obj, p)
        print(f'[cache] сохранено: {name}')
    return obj

print('Каталог артефактов:', os.path.abspath(ART), '| кэш включён:', CFG.USE_CACHE)

# -------------------------------------------------------------------------
# 1. Загрузка данных и оптимизация памяти
# -------------------------------------------------------------------------

def reduce_mem_usage(df, verbose=True):
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        t = df[col].dtype
        if pd.api.types.is_integer_dtype(t):
            cmin, cmax = df[col].min(), df[col].max()
            if cmin >= 0:
                if   cmax < 2**8:  df[col] = df[col].astype(np.uint8)
                elif cmax < 2**16: df[col] = df[col].astype(np.uint16)
                elif cmax < 2**32: df[col] = df[col].astype(np.uint32)
                else:              df[col] = df[col].astype(np.uint64)
            else:
                if   cmin > np.iinfo(np.int8).min  and cmax < np.iinfo(np.int8).max:  df[col] = df[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                else: df[col] = df[col].astype(np.int64)
        elif pd.api.types.is_float_dtype(t):
            df[col] = df[col].astype(np.float32)
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f'  память: {start:.1f} MB -> {end:.1f} MB ({100*(start-end)/max(start,1e-9):.1f}% экономии)')
    return df

train_df = pd.read_parquet(path(CFG.TRAIN_DATA))
test_df  = pd.read_parquet(path(CFG.TEST_DATA))
target   = pd.read_csv(path(CFG.TRAIN_TARGET))
sample_sub = pd.read_csv(path(CFG.SAMPLE_SUB))

print('train_data:', train_df.shape)
print('test_data :', test_df.shape)
print('target    :', target.shape)
print('sample_sub:', sample_sub.shape)

print('\nОптимизация памяти:')
train_df = reduce_mem_usage(train_df)
test_df  = reduce_mem_usage(test_df)
gc.collect()
train_df.head()

def detect_feature_groups(df, id_col=CFG.ID_COL, seq_col=CFG.SEQ_COL, target_col=CFG.TARGET):
    service = {id_col, seq_col, target_col}
    feat = [c for c in df.columns if c not in service]
    paym = sorted([c for c in feat if c.startswith('enc_paym_')],
                  key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
    return {
        'all':      feat,
        'flags':    [c for c in feat if c.startswith('is_zero') or c.endswith('_flag')],
        'enc_paym': paym,
        'enc_cat':  [c for c in feat if c.startswith('enc_') and not c.startswith('enc_paym_')],
        'pre':      [c for c in feat if c.startswith('pre_')],
    }

groups = detect_feature_groups(train_df)
for k in ['pre', 'flags', 'enc_cat', 'enc_paym']:
    print(f'{k:9s}: {len(groups[k]):3d} признаков  ->', groups[k][:4], '...' if len(groups[k]) > 4 else '')

# -------------------------------------------------------------------------
# 3. Генерация признаков (агрегация на уровень `id`)
# -------------------------------------------------------------------------

def build_aggregated_features(df, feat_cols, id_col=CFG.ID_COL, seq_col=CFG.SEQ_COL):
    df_sorted = df.sort_values([id_col, seq_col])
    g = df_sorted.groupby(id_col, sort=True)
    agg = g[feat_cols].agg(['mean', 'std', 'min', 'max', 'sum', 'nunique'])
    agg.columns = [f'{c}_{s}' for c, s in agg.columns]
    last  = g[feat_cols].last().add_suffix('_last')
    first = g[feat_cols].first().add_suffix('_first')
    n_prod = g.size().rename('n_products').to_frame()
    out = pd.concat([agg, last, first, n_prod], axis=1).astype(np.float32)
    out['n_products'] = out['n_products'].astype(np.int32)
    return out

def build_categorical_count_features(df, cat_cols, id_col=CFG.ID_COL, max_cardinality=25):
    pieces = []
    for col in cat_cols:
        if 1 < df[col].nunique(dropna=True) <= max_cardinality:
            ct = pd.crosstab(df[id_col], df[col])
            ct.columns = [f'{col}_cnt_{int(v)}' for v in ct.columns]
            pieces.append(ct.astype(np.float32))
    return pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=pd.Index([], name=id_col))

def assemble_feature_table(df, groups):
    agg = build_aggregated_features(df, groups['all'])
    cnt = build_categorical_count_features(df, groups['enc_cat'] + groups['flags'])
    feats = agg.join(cnt, how='left') if len(cnt) else agg
    return feats.fillna(0)

def build_features_polars(parquet_path, groups, id_col=CFG.ID_COL, seq_col=CFG.SEQ_COL, max_card=25):
    feat = groups['all']
    lf = pl.scan_parquet(parquet_path).sort([id_col, seq_col])
    exprs = []
    for c in feat:
        exprs += [pl.col(c).mean().alias(f'{c}_mean'),   pl.col(c).std().alias(f'{c}_std'),
                  pl.col(c).min().alias(f'{c}_min'),     pl.col(c).max().alias(f'{c}_max'),
                  pl.col(c).sum().alias(f'{c}_sum'),     pl.col(c).n_unique().alias(f'{c}_nunique'),
                  pl.col(c).last().alias(f'{c}_last'),   pl.col(c).first().alias(f'{c}_first')]
    exprs.append(pl.len().alias('n_products'))
    out = lf.group_by(id_col).agg(exprs).collect()
    cat = groups['enc_cat'] + groups['flags']
    df = pl.scan_parquet(parquet_path).select([id_col] + cat).collect()
    for c in cat:
        if 1 < df.select(pl.col(c).n_unique()).item() <= max_card:
            wide = (df.group_by([id_col, c]).len()
                      .pivot(on=c, index=id_col, values='len', aggregate_function='first').fill_null(0))
            wide = wide.rename({col: f'{c}_cnt_{col}' for col in wide.columns if col != id_col})
            out = out.join(wide, on=id_col, how='left')
    pdf = out.fill_null(0).to_pandas().set_index(id_col).sort_index()
    for col in pdf.columns:
        pdf[col] = pdf[col].astype(np.float32 if col != 'n_products' else np.int32)
    return pdf

# -------------------------------------------------------------------------
# 4. Отбор признаков
# -------------------------------------------------------------------------

def _build_features():
    eng = 'polars' if (CFG.USE_POLARS and HAS_POLARS) else 'pandas'
    print('Движок агрегации:', eng)
    t0 = time.time()
    if eng == 'polars':
        Xtr = build_features_polars(path(CFG.TRAIN_DATA), groups)
        Xte = build_features_polars(path(CFG.TEST_DATA),  groups)
    else:
        Xtr = assemble_feature_table(train_df, groups)
        Xte = assemble_feature_table(test_df,  groups)
    print(f'  агрегация за {time.time()-t0:.1f}s')

    common = Xtr.columns.intersection(Xte.columns)
    Xtr, Xte = Xtr[common].copy(), Xte[common].copy()
    yv = target.set_index(CFG.ID_COL).reindex(Xtr.index)[CFG.TARGET]
    assert yv.notna().all(), 'Есть id из train_data без таргета — проверьте train_target.csv'
    yv = yv.astype(int)

    nun = Xtr.nunique(); const_cols = nun[nun <= 1].index.tolist()
    Xtr.drop(columns=const_cols, inplace=True); Xte.drop(columns=const_cols, inplace=True)

    Xs = Xtr.sample(min(len(Xtr), 20000), random_state=CFG.SEED)
    corr = Xs.corr().abs(); upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_cols = [c for c in upper.columns if (upper[c] > CFG.CORR_THRESHOLD).any()]
    Xtr.drop(columns=corr_cols, inplace=True); Xte.drop(columns=corr_cols, inplace=True)
    del Xs, corr, upper; gc.collect()

    imp = None
    if HAS_LGB:
        im = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                subsample=0.8, colsample_bytree=0.8, random_state=CFG.SEED,
                                n_jobs=-1, verbose=-1)
        im.fit(Xtr, yv)
        imp = pd.Series(im.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
        if CFG.IMP_TOP_K:
            keep = imp.head(CFG.IMP_TOP_K).index.tolist(); Xtr, Xte = Xtr[keep], Xte[keep]
    return dict(X=Xtr, X_test=Xte, y=yv, importances=imp,
                dropped_const=const_cols, dropped_corr=corr_cols)

_feat = load_or('features.joblib', _build_features)
X, X_test, y = _feat['X'], _feat['X_test'], _feat['y']
importances = _feat['importances']
print(f"Удалено: константных {len(_feat['dropped_const'])}, коррелированных {len(_feat['dropped_corr'])}")
print('X:', X.shape, '| X_test:', X_test.shape, '| доля дефолтов:', round(float(y.mean()), 4))
try:
    del train_df
except NameError:
    pass
gc.collect()
X.head()

def add_trend_features(df, base_feats):
    new = {}
    for c in base_feats:
        l, f, m = f'{c}_last', f'{c}_first', f'{c}_mean'
        if l in df.columns and f in df.columns:
            new[f'{c}_last_minus_first'] = df[l] - df[f]
        if l in df.columns and m in df.columns:
            new[f'{c}_last_minus_mean']  = df[l] - df[m]
    return pd.DataFrame(new, index=df.index).astype(np.float32)

base = groups['all']
if not any(c.endswith('_last_minus_first') for c in X.columns):
    X      = pd.concat([X,      add_trend_features(X,      base)], axis=1)
    X_test = pd.concat([X_test, add_trend_features(X_test, base)], axis=1)
    _feat['X'], _feat['X_test'] = X, X_test
    joblib.dump(_feat, apath('features.joblib'))
    print('Тренд-признаки добавлены и перекэшированы: X', X.shape, '| X_test', X_test.shape)
else:
    print('Тренд-признаки уже на месте: X', X.shape)

# -------------------------------------------------------------------------
# 5. Подготовка к обучению
# -------------------------------------------------------------------------

from tqdm.auto import tqdm

skf = StratifiedKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
folds = list(skf.split(X, y))
RESULTS = {}

def run_cv(name, fit_predict_fn):
    oof = np.zeros(len(X)); test_pred = np.zeros(len(X_test)); fold_aucs = []
    t0 = time.time()
    pbar = tqdm(list(enumerate(folds)), desc=name, unit='fold', leave=False)
    for k, (tr, va) in pbar:
        pv, pt = fit_predict_fn(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], X_test)
        oof[va] = pv; test_pred += pt / len(folds)
        a = roc_auc_score(y.iloc[va], pv); fold_aucs.append(a)
        pbar.set_postfix(fold_auc=f'{a:.4f}')
    auc = roc_auc_score(y, oof); dt = time.time() - t0
    return dict(oof=oof, test=test_pred, fold_aucs=fold_aucs, time=dt, auc=auc, kind='gbm')

def run_cv_cached(name, fit_predict_fn):
    RESULTS[name] = load_or(f'cv_{name}.joblib', lambda: run_cv(name, fit_predict_fn))
    r = RESULTS[name]
    print(f"{name:14s} OOF AUC={r['auc']:.5f}  folds={np.round(r['fold_aucs'], 4).tolist()}  time={r['time']:.1f}s")
    return r

print('Фолдов:', len(folds), '| train:', len(X), '| test:', len(X_test))

# -------------------------------------------------------------------------
# 6. Бустинги и бейзлайн (с отслеживанием времени обучения)
# -------------------------------------------------------------------------

def fit_lr(Xtr, ytr, Xva, yva, Xte):
    n = min(len(Xtr), 300_000)
    sidx = np.random.default_rng(CFG.SEED).choice(len(Xtr), n, replace=False)
    pipe = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(max_iter=300, C=0.1))
    pipe.fit(Xtr.iloc[sidx], ytr.iloc[sidx])
    return pipe.predict_proba(Xva)[:, 1], pipe.predict_proba(Xte)[:, 1]

run_cv_cached('LogReg', fit_lr)

if HAS_LGB:
    def fit_lgb(Xtr, ytr, Xva, yva, Xte):
        m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.02, num_leaves=63,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                               min_child_samples=50, random_state=CFG.SEED, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric='auc',
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    run_cv_cached('LightGBM', fit_lgb)
else:
    print('LightGBM недоступен — пропуск.')

if HAS_XGB:
    def fit_xgb(Xtr, ytr, Xva, yva, Xte):
        m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.02, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                              min_child_weight=5, eval_metric='auc', tree_method='hist', device='cuda',
                              random_state=CFG.SEED, n_jobs=-1, early_stopping_rounds=150)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    run_cv_cached('XGBoost', fit_xgb)
else:
    print('XGBoost недоступен — пропуск.')

if HAS_CAT:
    def fit_cat(Xtr, ytr, Xva, yva, Xte):
        m = CatBoostClassifier(iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
                               eval_metric='AUC', random_seed=CFG.SEED, verbose=False,
                               early_stopping_rounds=150, task_type='GPU')
        m.fit(Xtr, ytr, eval_set=(Xva, yva), verbose=False)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    run_cv_cached('CatBoost', fit_cat)
else:
    print('CatBoost недоступен — пропуск.')

# -------------------------------------------------------------------------
# 7. Нейросеть на последовательностях (GRU / Transformer)
# -------------------------------------------------------------------------

import pyarrow.parquet as pq

def read_seq_int8(parquet_path, seq_feats, id_col=CFG.ID_COL, seq_col=CFG.SEQ_COL, batch=1_000_000):
    pfile = pq.ParquetFile(parquet_path)
    ids_l, rns_l, arr_l = [], [], []
    for b in pfile.iter_batches(batch_size=batch, columns=[id_col, seq_col] + seq_feats):
        idx = {n: i for i, n in enumerate(b.schema.names)}
        ids_l.append(b.column(idx[id_col]).to_numpy())
        rns_l.append(b.column(idx[seq_col]).to_numpy())
        a = np.empty((b.num_rows, len(seq_feats)), dtype=np.int8)
        for j, c in enumerate(seq_feats):
            a[:, j] = b.column(idx[c]).to_numpy(zero_copy_only=False)
        arr_l.append(a)
    return np.concatenate(ids_l), np.concatenate(rns_l), np.concatenate(arr_l)

def build_flat(ids, rns, arr, vocab, feat_cols, max_len):
    order = np.lexsort((rns, ids)); ids = ids[order]; arr = arr[order]; del order
    np.add(arr, 1, out=arr)
    caps = np.array([vocab[c] for c in feat_cols], dtype=np.int8) + 1
    np.clip(arr, 0, caps, out=arr)
    uniq, start = np.unique(ids, return_index=True); start = np.append(start, len(ids))
    grp = np.repeat(np.arange(len(uniq)), np.diff(start))
    pos = np.arange(len(ids)) - start[grp]
    keep = pos >= ((start[grp + 1] - start[grp]) - max_len)
    flat = arr[keep]
    offsets = np.concatenate([[0], np.cumsum(np.minimum(np.diff(start), max_len))]).astype(np.int64)
    return uniq, flat, offsets

if CFG.RUN_NN and HAS_TORCH:
    import gc
    if 'X' not in globals():
        _f = joblib.load(apath('features.joblib')); X, X_test = _f['X'], _f['X_test']; del _f; gc.collect()

    seq_feats = groups['all']
    Xi  = X.index.values.copy()
    Xti = X_test.index.values.copy()
    ML  = CFG.NN_MAX_LEN or (int(np.clip(np.quantile(X['n_products'].values, 0.95), 4, 16))
                             if 'n_products' in X.columns else 16)

    def _build_seqs():
        ids, rns, arr = read_seq_int8(path(CFG.TRAIN_DATA), seq_feats)
        vocab = {c: int(arr[:, j].max()) for j, c in enumerate(seq_feats)}
        tr_id, tr_flat, tr_off = build_flat(ids, rns, arr, vocab, seq_feats, ML)
        del ids, rns, arr; gc.collect()
        tids = test_df[CFG.ID_COL].to_numpy(); trns = test_df[CFG.SEQ_COL].to_numpy()
        tarr = test_df[seq_feats].to_numpy().astype(np.int8)
        te_id, te_flat, te_off = build_flat(tids, trns, tarr, vocab, seq_feats, ML)
        del tids, trns, tarr; gc.collect()
        assert np.array_equal(tr_id, Xi),  'порядок train id ≠ X.index'
        assert np.array_equal(te_id, Xti), 'порядок test id ≠ X_test.index'
        return dict(tr_flat=tr_flat, tr_off=tr_off, te_flat=te_flat, te_off=te_off,
                    vocab_sizes=[vocab[c] for c in seq_feats], max_len=ML)

    del X, X_test; gc.collect()

    _seq = load_or('nn_seqs.joblib', _build_seqs)
    tr_flat, tr_off = _seq['tr_flat'], _seq['tr_off']
    te_flat, te_off = _seq['te_flat'], _seq['te_off']
    vocab_sizes, max_len = _seq['vocab_sizes'], _seq['max_len']
    print('NN_MAX_LEN =', max_len, '| train seq =', len(tr_off) - 1, '| test seq =', len(te_off) - 1,
          '| flat train =', tr_flat.shape, tr_flat.dtype, '| признаков =', len(seq_feats))
else:
    print('Нейросеть отключена (CFG.RUN_NN=False) или PyTorch недоступен.')

if CFG.RUN_NN and HAS_TORCH:
    class SeqDataset(Dataset):
        def __init__(self, flat, offsets, idx=None, targets=None):
            self.flat = flat; self.offsets = offsets
            self.idx = np.arange(len(offsets) - 1) if idx is None else np.asarray(idx)
            self.targets = targets
        def __len__(self): return len(self.idx)
        def __getitem__(self, j):
            i = int(self.idx[j]); s, e = self.offsets[i], self.offsets[i + 1]
            seq = torch.from_numpy(np.ascontiguousarray(self.flat[s:e]))
            y_ = -1.0 if self.targets is None else float(self.targets[i])
            return seq, y_

    def collate(batch):
        seqs, ys = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        Lmax = int(lengths.max()); F = seqs[0].shape[1]
        padded = torch.zeros(len(seqs), Lmax, F, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, :len(s)] = s
        return padded, lengths, torch.tensor(ys, dtype=torch.float32)

    class CreditSeqNet(nn.Module):
        def __init__(self, vocab_sizes, encoder='gru', hidden=128, n_layers=2, dropout=0.2, nhead=4):
            super().__init__()
            emb_dims = [min(16, max(2, int(round((v + 2) ** 0.5)))) for v in vocab_sizes]
            self.embs = nn.ModuleList([nn.Embedding(v + 2, d, padding_idx=0)
                                       for v, d in zip(vocab_sizes, emb_dims)])
            in_dim = sum(emb_dims); self.encoder = encoder
            if encoder == 'gru':
                self.rnn = nn.GRU(in_dim, hidden, num_layers=n_layers, batch_first=True,
                                  bidirectional=True, dropout=dropout if n_layers > 1 else 0.0)
                head_in = hidden * 2
            else:
                self.proj = nn.Linear(in_dim, hidden)
                layer = nn.TransformerEncoderLayer(hidden, nhead, hidden * 2, dropout,
                                                   batch_first=True, activation='gelu')
                self.tr = nn.TransformerEncoder(layer, n_layers); head_in = hidden
            self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, hidden),
                                      nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        def forward(self, x, lengths):
            h = torch.cat([emb(x[:, :, i]) for i, emb in enumerate(self.embs)], dim=-1)
            if self.encoder == 'gru':
                packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True,
                                                           enforce_sorted=False)
                _, h_n = self.rnn(packed)
                h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                h = self.proj(h); mask = (x[:, :, 0] == 0)
                h = self.tr(h, src_key_padding_mask=mask)
                valid = (~mask).unsqueeze(-1).float()
                h_last = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
            return self.head(h_last).squeeze(-1)
    print('Модель CreditSeqNet определена.')

if CFG.RUN_NN and HAS_TORCH:
    CFG.NN_SEEDS = [42, 1, 7]

    def predict_nn(model, loader):
        model.eval(); out = []
        with torch.no_grad():
            for xb, lb, *_ in loader:
                xb, lb = xb.to(DEVICE), lb.to(DEVICE)
                out.append(torch.sigmoid(model(xb, lb)).cpu().numpy())
        return np.concatenate(out)

    def pairwise_logistic_loss(logits, targets):
        pos = logits[targets == 1]; neg = logits[targets == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return logits.new_zeros(())
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        return torch.nn.functional.softplus(-diff).mean()

    def train_nn(tr_idx, va_idx):
        seed_everything()
        max_train = getattr(CFG, 'NN_MAX_TRAIN', 1_500_000)
        if len(tr_idx) > max_train:
            tr_idx = np.random.default_rng(CFG.SEED).choice(tr_idx, size=max_train, replace=False)
        yv = y.values
        ytr = yv[tr_idx]; yva = yv[va_idx]
        pos_w = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        rank_w = getattr(CFG, 'NN_RANK_W', 1.0)
        nw = getattr(CFG, 'NN_WORKERS', 0)
        tl  = DataLoader(SeqDataset(tr_flat, tr_off, idx=tr_idx, targets=yv),
                         batch_size=CFG.NN_BATCH, shuffle=True,  collate_fn=collate, num_workers=nw)
        vl  = DataLoader(SeqDataset(tr_flat, tr_off, idx=va_idx, targets=yv),
                         batch_size=CFG.NN_BATCH, shuffle=False, collate_fn=collate, num_workers=nw)
        tel = DataLoader(SeqDataset(te_flat, te_off),
                         batch_size=CFG.NN_BATCH, shuffle=False, collate_fn=collate, num_workers=nw)
        model = CreditSeqNet(vocab_sizes, encoder=CFG.NN_ENCODER, hidden=CFG.NN_HIDDEN).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=CFG.NN_LR, weight_decay=1e-5)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))
        best_auc, best_state = -1, None
        for ep in range(CFG.NN_EPOCHS):
            model.train()
            for xb, lb, yb in tl:
                xb, lb, yb = xb.to(DEVICE), lb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                logits = model(xb, lb)
                loss = crit(logits, yb) + rank_w * pairwise_logistic_loss(logits, yb)
                loss.backward(); opt.step()
            auc = roc_auc_score(yva, predict_nn(model, vl))
            if auc > best_auc:
                best_auc = auc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'    эпоха {ep+1}/{CFG.NN_EPOCHS}  val AUC={auc:.4f}')
        model.load_state_dict(best_state)
        vp, tp = predict_nn(model, vl), predict_nn(model, tel)
        del model, opt, crit, tl, vl, tel, best_state; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return vp, tp, best_auc

    def _run_nn():
        t0 = time.time()
        seeds = getattr(CFG, 'NN_SEEDS', [CFG.SEED])
        n_tr = len(tr_off) - 1
        oof = np.zeros(n_tr); test_pred = np.zeros(len(te_off) - 1); aucs = []
        base = CFG.SEED
        for sd in seeds:
            CFG.SEED = sd; oof_s = np.zeros(n_tr)
            print(f'  [{CFG.NN_ENCODER}] seed {sd}')
            for k, (tri, vai) in enumerate(folds):
                vp, tp, a = train_nn(tri, vai)
                oof_s[vai] = vp; test_pred += tp / (len(folds) * len(seeds)); aucs.append(a)
                gc.collect()
            oof += oof_s / len(seeds)
            print(f'  seed {sd}: OOF AUC={roc_auc_score(y.values, oof_s):.4f}')
        CFG.SEED = base
        auc = roc_auc_score(y.values, oof)
        return dict(oof=oof, test=test_pred, fold_aucs=aucs, auc=auc, kind='nn', time=time.time() - t0)

    for enc in ['gru', 'transformer']:
        CFG.NN_ENCODER = enc
        RESULTS[f'NN-{enc}'] = load_or(f'nn_{enc}.joblib', _run_nn)
        r = RESULTS[f'NN-{enc}']
        print(f"NN-{enc}: bagged+seed OOF AUC={r['auc']:.5f}  time={r['time']:.1f}s\n")
else:
    print('Нейросеть пропущена.')

# -------------------------------------------------------------------------
# 8. Сравнение моделей и выбор лучшей
# -------------------------------------------------------------------------

if 'X' not in globals():
    _f = joblib.load(apath('features.joblib')); X, X_test = _f['X'], _f['X_test']; del _f; gc.collect()
rows = []
for name, r in RESULTS.items():
    if r['kind'] == 'gbm':
        rows.append(dict(model=name, metric='OOF AUC', auc=r['auc'],
                         fold0_auc=r['fold_aucs'][0], time_s=r['time']))
    else:
        rows.append(dict(model=name, metric='fold-0 AUC', auc=r['auc'],
                         fold0_auc=r['auc'], time_s=r['time']))
comp = pd.DataFrame(rows).sort_values('auc', ascending=False).reset_index(drop=True)
print(comp.round(5).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
c = comp.sort_values('auc')
ax[0].barh(c['model'], c['auc'], color='#4C72B0')
ax[0].set_xlim(max(0.5, c['auc'].min() - 0.02), c['auc'].max() + 0.005)
ax[0].set_title('Качество (AUC)')
for i, v in enumerate(c['auc']): ax[0].text(v, i, f' {v:.4f}', va='center')
ax[1].barh(c['model'], c['time_s'], color='#55A868'); ax[1].set_title('Время обучения, с')
plt.tight_layout(); plt.show()

gbm_only = {k: v for k, v in RESULTS.items() if v['kind'] == 'gbm'}
best_gbm = max(gbm_only, key=lambda k: gbm_only[k]['auc'])
print('Лучший бустинг по OOF AUC:', best_gbm, f"({gbm_only[best_gbm]['auc']:.5f})")

# -------------------------------------------------------------------------
# 9. Подбор гиперпараметров (Optuna)
# -------------------------------------------------------------------------

if CFG.RUN_OPTUNA and HAS_OPT and HAS_XGB:
    opt_folds = folds[:3]
    def objective_xgb(trial):
        p = dict(n_estimators=3000, tree_method='hist', device='cuda',
                 eval_metric='auc', random_state=CFG.SEED, n_jobs=-1, early_stopping_rounds=100,
                 learning_rate=trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
                 max_depth=trial.suggest_int('max_depth', 4, 10),
                 min_child_weight=trial.suggest_int('min_child_weight', 1, 20),
                 subsample=trial.suggest_float('subsample', 0.6, 1.0),
                 colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                 reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                 reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                 gamma=trial.suggest_float('gamma', 1e-3, 5.0, log=True))
        a = []
        for tr, va in opt_folds:
            m = xgb.XGBClassifier(**p)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
            a.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        return float(np.mean(a))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction='maximize', study_name='xgb_credit',
                             storage=f'sqlite:///{apath("optuna_xgb.db")}', load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=CFG.SEED))
    rem = max(0, CFG.OPTUNA_TRIALS - sum(t.state.name == 'COMPLETE' for t in st.trials))
    if rem: st.optimize(objective_xgb, n_trials=rem, show_progress_bar=True)
    best_xgb = st.best_params; print('XGB best CV AUC', round(st.best_value, 5), best_xgb)

    import hashlib, json
    key = 'XGBoost_tuned_' + hashlib.md5(json.dumps(best_xgb, sort_keys=True).encode()).hexdigest()[:8]
    def fit_xgb_tuned(Xtr, ytr, Xva, yva, Xte):
        m = xgb.XGBClassifier(n_estimators=4000, tree_method='hist', device='cuda',
                              eval_metric='auc', random_state=CFG.SEED, n_jobs=-1,
                              early_stopping_rounds=150, **best_xgb)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    run_cv_cached(key, fit_xgb_tuned); RESULTS['XGBoost_tuned'] = RESULTS.pop(key)

if HAS_XGB:
    import hashlib, json
    _best_xgb = globals().get('best_xgb') or {}
    _sig = hashlib.md5(json.dumps(_best_xgb, sort_keys=True).encode()).hexdigest()[:8] if _best_xgb else 'base'
    _tuned_key = 'XGBoost_tuned_' + _sig
    def fit_xgb_tuned(Xtr, ytr, Xva, yva, Xte):
        m = xgb.XGBClassifier(n_estimators=4000, tree_method='hist', device='cuda',
                              eval_metric='auc', random_state=CFG.SEED, n_jobs=-1,
                              early_stopping_rounds=150,
                              **{**dict(learning_rate=0.02, max_depth=6, subsample=0.8,
                                        colsample_bytree=0.8, reg_lambda=2.0, min_child_weight=5),
                                 **_best_xgb})
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    run_cv_cached(_tuned_key, fit_xgb_tuned)
    RESULTS['XGBoost_tuned'] = RESULTS.pop(_tuned_key)
    gbm_only = {k: v for k, v in RESULTS.items() if v['kind'] == 'gbm'}
    best_gbm = max(gbm_only, key=lambda k: gbm_only[k]['auc'])
    print('Лучший бустинг после тюнинга:', best_gbm, f"({gbm_only[best_gbm]['auc']:.5f})")

# -------------------------------------------------------------------------
# 10. Финальная модель, блендинг и сабмит
# -------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from scipy.stats import rankdata

if 'X_test' not in globals():
    _f = joblib.load(apath('features.joblib')); X, X_test = _f['X'], _f['X_test']; del _f

stack_models = [k for k, v in RESULTS.items() if 'oof' in v]
OOF  = np.column_stack([RESULTS[k]['oof']  for k in stack_models])
TEST = np.column_stack([RESULTS[k]['test'] for k in stack_models])
yv = y.values

meta_oof = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), OOF, yv,
                             cv=5, method='predict_proba')[:, 1]
auc_stack = roc_auc_score(yv, meta_oof)
auc_rank  = roc_auc_score(yv, np.mean([rankdata(OOF[:, j]) for j in range(OOF.shape[1])], 0))

print('В стеке:', stack_models)
for k in stack_models: print(f'  {k:20s} OOF AUC={RESULTS[k]["auc"]:.5f}')
print(f'rank-average OOF AUC = {auc_rank:.5f}')
print(f'СТЕК (честный CV)    = {auc_stack:.5f}')

meta = LogisticRegression(C=1.0, max_iter=2000).fit(OOF, yv)
print('  веса:', dict(zip(stack_models, np.round(meta.coef_[0], 3))))
final_test = meta.predict_proba(TEST)[:, 1]
if auc_rank > auc_stack:
    print('rank-average выше — берём его')
    r = np.mean([rankdata(TEST[:, j]) for j in range(TEST.shape[1])], 0)
    final_test = (r - r.min()) / (r.max() - r.min())

sub = sample_sub[[CFG.ID_COL]].merge(
    pd.DataFrame({CFG.ID_COL: X_test.index.values, CFG.TARGET: final_test}), on=CFG.ID_COL, how='left')
sub[CFG.TARGET] = sub[CFG.TARGET].fillna(sub[CFG.TARGET].mean())
assert sub[CFG.ID_COL].is_unique and len(sub) == len(sample_sub)
sub.to_csv(CFG.SUB_OUT, index=False, float_format='%.6f')
print('Сабмит сохранён:', CFG.SUB_OUT, '| строк:', len(sub))

# -------------------------------------------------------------------------
# 11. Выводы и направления улучшения
# -------------------------------------------------------------------------
