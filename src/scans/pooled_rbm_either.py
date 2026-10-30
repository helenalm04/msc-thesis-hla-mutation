# Chapter 4 control for the 'either' score. Scoring a peptide as the better of
# two separately fitted models is not the same as fitting one model to both
# repertoires pooled, so this trains that pooled RBM and compares it against
# the two shipped models on held out ligands. The one script here that fits a
# model rather than loading one. The allele pair is set by PAIR_A and PAIR_B.

import os, sys, glob, time, argparse, importlib.util, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE); ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import (IEDB_MS_CSV, RESULTS_DIR, FIGURES_DIR, MODELS_DIR,
                    FS_LABEL, FS_TITLE, FS_TICK)
from encoding import convert_number
import RBM_utils

_spec = importlib.util.spec_from_file_location(
    'pgm_rbm', os.path.join(ROOT, 'libs', 'PGM', 'source', 'rbm.py'))
pgm_rbm = importlib.util.module_from_spec(_spec); sys.modules['pgm_rbm'] = pgm_rbm
_spec.loader.exec_module(pgm_rbm)

# the allele name in the compact form the model filenames use
def _tag(allele):
    return allele.replace('HLA-', '').replace('*', '').replace(':', '')


ALLELE_A = os.environ.get('PAIR_A', 'HLA-B*51:01')
ALLELE_B = os.environ.get('PAIR_B', 'HLA-B*27:05')
TAG_A, TAG_B = _tag(ALLELE_A), _tag(ALLELE_B)
_DEFAULT = (ALLELE_A, ALLELE_B) == ('HLA-B*51:01', 'HLA-B*27:05')
SUF = '' if _DEFAULT else f'_{TAG_A}_{TAG_B}'
DO_CTRL = os.environ.get('CTRL', '1' if _DEFAULT else '0') == '1'
SEED = 42
TEST_FRAC = 0.18
QUICK = os.environ.get('QUICK', '0') == '1'
N_ITER_POOL = 30 if QUICK else 600  # hu=10 alleles converged between 500 and 620
N_ITER_CTRL = 30 if QUICK else 1190  # iterations the shipped model used
AIS_M, AIS_BETAS = (20, 2000) if QUICK else (100, 20000)
HU_POOL, HU_SINGLE, L12_POOL = 10, 5, 0.01  # pool capacity matches two hu=5 models

POOLED_DIR = os.path.join(RESULTS_DIR, 'pooled_models')
os.makedirs(POOLED_DIR, exist_ok=True)
POOL_PATH = os.path.join(POOLED_DIR, f'pooled{SUF or "_B5101_B2705"}.data')
CTRL_PATH = os.path.join(POOLED_DIR, f'ctrl_{TAG_A}.data')
CACHE = os.path.join(FIGURES_DIR, f'pooled_rbm_either{SUF}.npz')
FIGOUT = os.path.join(FIGURES_DIR, f'pooled_rbm_either{SUF}.pdf')

C_HARD = '#EA9542'
C_SOFT = '#5C82D6'

FIT_FIXED = dict(batch_size=100, learning_rate=0.001, N_MC=15, decay_after=0.5)


def shipped_hp(tag):
    '''Read hu and l12 from a shipped model's filename.'''
    f = os.path.basename(glob.glob(os.path.join(MODELS_DIR, f'model_HLA-{tag}_*.data'))[0])
    m = re.search(r'_hu_(\d+)_l12_([\d.]+)_', f)
    return int(m.group(1)), float(m.group(2))


# the pretrained RBM supplied for a single allele
def load_shipped(tag):
    return RBM_utils.loadRBM(glob.glob(os.path.join(MODELS_DIR, f'model_HLA-{tag}_*.data'))[0])


def train_rbm(data, n_h, l1b, n_iter, seed=SEED):
    '''Train a Potts/dReLU RBM on encoded 9-mers.'''
    R = pgm_rbm.RBM(visible='Potts', hidden='dReLU', n_v=9, n_h=n_h, n_cv=20,
                    random_state=np.random.RandomState(seed))
    w = np.ones(len(data), dtype='float32')
    R.fit(data.astype(np.int16), weights=w, n_iter=n_iter, l1b=l1b,
          l2_fields=1.0 / len(data), verbose=0, **FIT_FIXED)
    return R


def ais_logZ(R, seed=0):
    R.random_state = np.random.RandomState(seed)
    R.AIS(M=AIS_M, n_betas=AIS_BETAS)
    return float(R.log_Z_AIS)


def split(seqs, frac=TEST_FRAC, seed=SEED):
    idx = np.random.RandomState(seed).permutation(len(seqs))
    n_test = int(round(frac * len(seqs)))
    return [seqs[i] for i in idx[n_test:]], [seqs[i] for i in idx[:n_test]]


def build():
    df = pd.read_csv(IEDB_MS_CSV)
    seqs_A = df[df['MHC allele'] == ALLELE_A]['Peptide'].tolist()
    seqs_B = df[df['MHC allele'] == ALLELE_B]['Peptide'].tolist()
    trA, teA = split(seqs_A); trB, teB = split(seqs_B)
    print(f'{ALLELE_A}: {len(seqs_A)}  (train {len(trA)}, test {len(teA)})')
    print(f'{ALLELE_B}: {len(seqs_B)}  (train {len(trB)}, test {len(teB)})')

    enc_trP = convert_number(trA + trB)
    piA = len(trA) / (len(trA) + len(trB)); piB = 1 - piA
    print(f'pool train N={len(enc_trP)}  pi_A={piA:.3f} pi_B={piB:.3f}')

    t = time.time()
    pool = train_rbm(enc_trP, HU_POOL, L12_POOL, N_ITER_POOL)
    RBM_utils.saveRBM(POOL_PATH, pool)
    print(f'pooled RBM (hu={HU_POOL}) trained in {time.time()-t:.0f}s -> {POOL_PATH}')

    # retrain the shipped model through this same code
    r_ctrl = float('nan')
    if DO_CTRL:
        t = time.time()
        hu_c, l12_c = shipped_hp(TAG_A)
        ctrl = train_rbm(convert_number(seqs_A), hu_c, l12_c, N_ITER_CTRL)
        RBM_utils.saveRBM(CTRL_PATH, ctrl)
        shipA_all = load_shipped(TAG_A).free_energy(convert_number(seqs_A).astype(np.int16))
        ctrl_all = ctrl.free_energy(convert_number(seqs_A).astype(np.int16))
        r_ctrl = pearsonr(shipA_all, ctrl_all)[0]
        print(f'control {ALLELE_A} (hu={hu_c}) trained in {time.time()-t:.0f}s | '
              f'free-energy r vs shipped = {r_ctrl:.4f}')

    np.savez(CACHE.replace('.npz', '_meta.npz'),
             teA=np.array(teA), teB=np.array(teB), piA=piA, piB=piB, r_ctrl=r_ctrl)
    return teA, teB, piA, piB, r_ctrl


def analyse(teA, teB, piA, piB, r_ctrl):
    shipA, shipB = load_shipped(TAG_A), load_shipped(TAG_B)
    pool = RBM_utils.loadRBM(POOL_PATH)

    print('AIS log-Z ...')
    logZ_A, logZ_B, logZ_P = ais_logZ(shipA), ais_logZ(shipB), ais_logZ(pool)
    print(f'  logZ_A={logZ_A:.3f}  logZ_B={logZ_B:.3f}  logZ_pool={logZ_P:.3f}')

    enc = convert_number(list(teA) + list(teB)).astype(np.int16)
    origin = np.array([0] * len(teA) + [1] * len(teB))  # 0 is B51 test, 1 is B27 test
    x1 = shipA.free_energy(enc) + logZ_A  # minus log p under allele A
    x2 = shipB.free_energy(enc) + logZ_B
    y = pool.free_energy(enc) + logZ_P
    hard = np.minimum(x1, x2)
    soft = -np.logaddexp(np.log(piA) - x1, np.log(piB) - x2)

    def stats(pred):
        resid = y - pred
        off = resid.mean()  # constant offset from AIS
        return dict(r=pearsonr(pred, y)[0], rho=spearmanr(pred, y)[0],
                    rmse=float(np.sqrt(np.mean(resid**2))),
                    resid_std=float(resid.std()), offset=float(off))

    s_hard, s_soft = stats(hard), stats(soft)
    poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly.fit(np.c_[x1, x2], y); yhat = poly.predict(np.c_[x1, x2])
    s_poly = dict(r=pearsonr(yhat, y)[0], resid_std=float((y - yhat).std()),
                  rmse=float(np.sqrt(np.mean((y - yhat)**2))))

    print(f'\nhard-min : r={s_hard["r"]:.3f} rho={s_hard["rho"]:.3f} '
          f'RMSE={s_hard["rmse"]:.3f} resid_std={s_hard["resid_std"]:.3f} off={s_hard["offset"]:+.3f}')
    print(f'soft-min : r={s_soft["r"]:.3f} rho={s_soft["rho"]:.3f} '
          f'RMSE={s_soft["rmse"]:.3f} resid_std={s_soft["resid_std"]:.3f} off={s_soft["offset"]:+.3f}')
    print(f'best f() : r={s_poly["r"]:.3f} resid_std={s_poly["resid_std"]:.3f} '
          f'(cross-allele residual floor)')

    resid = y - soft
    bothness = -np.maximum(x1, x2)  # high when even the worse allele presents it
    rho_both = float(spearmanr(bothness, resid)[0])
    thr = float(np.median(np.maximum(x1, x2)))
    ov = np.maximum(x1, x2) <= thr
    rc = resid - resid.mean()  # remove the offset shared by all peptides
    ov_sd, sg_sd = float(rc[ov].std()), float(rc[~ov].std())
    print(f'overlap  : rho(resid,bothness)={rho_both:+.3f}  '
          f'resid_sd overlap={ov_sd:.3f} vs single={sg_sd:.3f} (ratio {ov_sd/sg_sd:.2f})')

    np.savez(CACHE, x1=x1, x2=x2, y=y, hard=hard, soft=soft, yhat=yhat, origin=origin,
             piA=piA, piB=piB, logZ_A=logZ_A, logZ_B=logZ_B, logZ_P=logZ_P,
             r_ctrl=r_ctrl, rho_both=rho_both, ov_sd=ov_sd, sg_sd=sg_sd,
             s_hard=np.array([s_hard[k] for k in ('r','rho','rmse','resid_std','offset')]),
             s_soft=np.array([s_soft[k] for k in ('r','rho','rmse','resid_std','offset')]),
             s_poly=np.array([s_poly['r'], s_poly['resid_std'], s_poly['rmse']]))
    plot()


def plot():
    d = np.load(CACHE, allow_pickle=True)
    x1, x2, y, hard, soft, yhat = d['x1'], d['x2'], d['y'], d['hard'], d['soft'], d['yhat']
    sh, ss, sp = d['s_hard'], d['s_soft'], d['s_poly']
    lo = float(min(y.min(), hard.min(), soft.min())); hi = float(max(y.max(), hard.max(), soft.max()))
    ident = np.array([lo, hi])

    fig, ax = plt.subplots(1, 3, figsize=(15.2, 4.6))

    ax[0].plot(ident, ident, 'k--', lw=1, zorder=1)
    ax[0].scatter(hard, y, s=8, c=C_HARD, alpha=.45, lw=0, label='best-presenter min')
    ax[0].scatter(soft, y, s=8, c=C_SOFT, alpha=.45, lw=0, label='soft min')
    ax[0].set_xlabel(r'score prediction  $-\log p$  (nats)', fontsize=FS_LABEL)
    ax[0].set_ylabel(r'pooled RBM  $-\log p_{\mathrm{pool}}$  (nats)', fontsize=FS_LABEL)
    ax[0].legend(fontsize=FS_TICK, loc='upper left', framealpha=.9)
    ax[0].set_title('(a) score against the pooled model', fontsize=FS_TITLE)

    eh = (y - hard) - (y - hard).mean()
    es = (y - soft) - (y - soft).mean()
    bins = np.linspace(min(eh.min(), es.min()), max(eh.max(), es.max()), 40)
    ax[1].hist(eh, bins=bins, color=C_HARD, alpha=.55, label=f'best-presenter min, sd {eh.std(ddof=1):.2f}')
    ax[1].hist(es, bins=bins, color=C_SOFT, alpha=.55, label=f'soft min, sd {es.std(ddof=1):.2f}')
    ax[1].axvline(0, color='k', lw=1, ls='--')
    ax[1].set_xlabel('per-peptide error, offset removed (nats)', fontsize=FS_LABEL)
    ax[1].set_ylabel('held-out ligands', fontsize=FS_LABEL)
    ax[1].legend(fontsize=FS_TICK, loc='upper left', framealpha=.9)
    ax[1].set_title('(b) error once the offset is removed', fontsize=FS_TITLE)

    lo2 = float(min(y.min(), yhat.min())); hi2 = float(max(y.max(), yhat.max()))
    ax[2].plot([lo2, hi2], [lo2, hi2], 'k--', lw=1, zorder=1)
    ax[2].scatter(yhat, y, s=8, c='0.35', alpha=.45, lw=0,
                  label='quadratic in $E_A,E_B$')
    ax[2].scatter(soft + (y - soft).mean(), y, s=8, c=C_SOFT, alpha=.35, lw=0,
                  label='soft min')
    ax[2].set_xlabel(r'fitted  $-\log p$  (nats)', fontsize=FS_LABEL)
    ax[2].set_ylabel(r'pooled RBM  $-\log p_{\mathrm{pool}}$  (nats)', fontsize=FS_LABEL)
    ax[2].legend(fontsize=FS_TICK, loc='upper left', framealpha=.9)
    ax[2].set_title('(c) the marginals suffice', fontsize=FS_TITLE)

    for a in ax:
        a.tick_params(labelsize=FS_TICK)

    print(f'per-peptide error sd (offset removed): '
          f'hard-min {eh.std(ddof=1):.2f}, soft-min {es.std(ddof=1):.2f} nats')
    print(f'marginals sufficiency: cross-allele residual sd = {sp[1]:.2f} nats')

    fig.tight_layout()
    fig.savefig(FIGOUT, bbox_inches='tight', dpi=200)
    fig.savefig(FIGOUT.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    print(f'figure -> {FIGOUT}')


def noise_floor(seed2=7):
    df = pd.read_csv(IEDB_MS_CSV)
    trA, teA = split(df[df['MHC allele'] == ALLELE_A]['Peptide'].tolist())
    trB, teB = split(df[df['MHC allele'] == ALLELE_B]['Peptide'].tolist())
    pool2 = train_rbm(convert_number(trA + trB), HU_POOL, L12_POOL,
                      N_ITER_POOL, seed=seed2)
    logZ2 = ais_logZ(pool2, seed=1)
    enc = convert_number(list(teA) + list(teB)).astype(np.int16)
    y2 = pool2.free_energy(enc) + logZ2
    d = np.load(CACHE)
    y1 = d['y']
    resid = y1 - y2
    noise_ss = float((resid - resid.mean()).std())
    floor = float(d['s_poly'][1])  # residual sd of the best fit
    noise_1 = noise_ss / np.sqrt(2)
    struct = float(np.sqrt(max(floor**2 - noise_1**2, 0.0)))
    print(f'seed-to-seed -log p: offset={resid.mean():+.3f}  noise_sd={noise_ss:.3f}')
    print(f'  per-model noise ~ {noise_1:.3f} nats')
    print(f'  best-f residual floor = {floor:.3f} nats  (soft-min resid_std = {d["s_soft"][3]:.3f})')
    print(f'  => implied non-marginal structure ~ {struct:.3f} nats')
    verdict = ('FLOOR IS NOISE -> marginals fully sufficient, no cross-allele structure'
               if struct < 0.25 else
               'residual exceeds noise -> some non-marginal structure remains (inconclusive)')
    print(f'  verdict: {verdict}')
    np.savez(CACHE.replace('.npz', '_noise.npz'), noise_ss=noise_ss, noise_1=noise_1,
             floor=floor, struct=struct, logZ2=logZ2)


def capacity_control(n_iter=None):
    n_iter = n_iter or N_ITER_POOL
    df = pd.read_csv(IEDB_MS_CSV)
    trA, teA = split(df[df['MHC allele'] == ALLELE_A]['Peptide'].tolist())
    trB, teB = split(df[df['MHC allele'] == ALLELE_B]['Peptide'].tolist())
    enc = convert_number(list(teA) + list(teB)).astype(np.int16)
    d = np.load(CACHE); y = d['y']; piA = float(d['piA']); piB = float(d['piB'])
    noise_1 = float(np.load(CACHE.replace('.npz', '_noise.npz'))['noise_1'])

    def resid_for(hu):
        sA = train_rbm(convert_number(trA), hu, 0.01, n_iter, seed=SEED)  # l12 held fixed
        sB = train_rbm(convert_number(trB), hu, 0.01, n_iter, seed=SEED)
        x1 = sA.free_energy(enc) + ais_logZ(sA, seed=2)
        x2 = sB.free_energy(enc) + ais_logZ(sB, seed=3)
        soft = -np.logaddexp(np.log(piA) - x1, np.log(piB) - x2)
        poly = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(np.c_[x1, x2], y)
        bestf = float((y - poly.predict(np.c_[x1, x2])).std())
        rsoft = y - soft
        struct = float(np.sqrt(max(bestf**2 - noise_1**2, 0.0)))
        return dict(soft_sd=float((rsoft - rsoft.mean()).std()), bestf=bestf, struct=struct)

    r5, r10 = resid_for(5), resid_for(10)
    print(f'noise floor: per-model {noise_1:.3f}  (seed-to-seed {noise_1*np.sqrt(2):.3f})')
    print(f'hu=5  singles (train): soft {r5["soft_sd"]:.3f}  best-f {r5["bestf"]:.3f}  struct {r5["struct"]:.3f}')
    print(f'hu=10 singles (train): soft {r10["soft_sd"]:.3f}  best-f {r10["bestf"]:.3f}  struct {r10["struct"]:.3f}')
    print(f'capacity effect (best-f resid hu5->hu10): {r5["bestf"]:.3f} -> {r10["bestf"]:.3f} '
          f'(drop {r5["bestf"]-r10["bestf"]:+.3f})')
    verdict = ('CAPACITY ARTIFACT -> pool ~ mixture of matched marginals, soft-min is the '
               'ceiling, no cross-allele epistasis' if r10['struct'] < 0.25 else
               'STRUCTURE SURVIVES capacity matching -> genuine cross-allele coupling '
               f'(~{r10["struct"]:.2f} nats the marginals cannot express)')
    print(f'verdict: {verdict}')
    np.savez(CACHE.replace('.npz', '_capacity.npz'),
             r5_soft=r5['soft_sd'], r5_bestf=r5['bestf'], r5_struct=r5['struct'],
             r10_soft=r10['soft_sd'], r10_bestf=r10['bestf'], r10_struct=r10['struct'],
             noise_1=noise_1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true', help='reload cached models + replot')
    ap.add_argument('--noise-floor', action='store_true',
                    help='retrain pool with a 2nd seed to isolate the noise floor')
    ap.add_argument('--capacity', action='store_true',
                    help='refit marginals from hu=5 vs hu=10 single models (capacity control)')
    args = ap.parse_args()
    if args.noise_floor:
        noise_floor()
    elif args.capacity:
        capacity_control()
    elif args.plot:
        m = np.load(CACHE.replace('.npz', '_meta.npz'), allow_pickle=True)
        analyse(list(m['teA']), list(m['teB']), float(m['piA']), float(m['piB']), float(m['r_ctrl']))
    else:
        analyse(*build())
