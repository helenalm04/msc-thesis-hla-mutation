# Appendix, the convergence table. Several chains are run on the same endpoint
# pairs and the diagnostics applied to two observables, the barrier height and
# where along the path the peak sits. Both, because a chain can mix well in
# energy while still disagreeing about where the barrier is.

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
ROOT = os.path.dirname(SRC)
for _p in (ROOT, SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import IEDB_MS_CSV, FIGURES_DIR, FS_LABEL, FS_TITLE, FS_TICK, L, AA_DICT
import diagnostics as dg
from diagnostics import run_chains, scalar_series, diagnose, format_report
from sampler import choose_endpoints_escape, hamming
from path_stats import substitution_order, route_entropy, mutational_load
from encoding import convert_number, convert_letter

BLUE, ORANGE = '#5C82D6', '#EA9542'  # palette from figure_conventions
ANCH_A, ANCH_B = {'P', 'A'}, {'R'}


# true for an encoded peptide built from at least min_distinct different amino acids
def _complex(enc, min_distinct=6):
    return len({int(x) for x in enc}) >= min_distinct


def natural_candidates(allele, df_ms, E_fn, anchors, top_frac=0.25):
    '''Real ligands with good energy and an anchor, best energy first.'''
    seqs = df_ms[df_ms['MHC allele'] == allele]['Peptide'].tolist()
    enc = convert_number(seqs)
    order = np.argsort(E_fn(enc))
    cut = max(int(len(order) * top_frac), 40)
    codes = {AA_DICT[a] for a in anchors}
    return [enc[i] for i in order[:cut] if int(enc[i][1]) in codes and _complex(enc[i])]


def pick_pairs(candA, candB, n_pairs, min_d=5):
    '''Pick n_pairs distinct endpoint pairs, far apart and low in energy.'''
    scored = []
    for i, a in enumerate(candA[:12]):
        for j, b in enumerate(candB[:12]):
            d = hamming(a, b)
            if d >= min_d:
                scored.append((i + j, d, a, b))
    scored.sort(key=lambda x: x[0])  # lowest rank sum first
    pairs, usedA, usedB = [], set(), set()
    for _, d, a, b in scored:
        ka, kb = a.tobytes(), b.tobytes()
        if ka in usedA or kb in usedB:
            continue
        pairs.append((a, b, d)); usedA.add(ka); usedB.add(kb)
        if len(pairs) >= n_pairs:
            break
    return pairs


# path observable, the highest energy an interior node reaches above the mean of the two endpoints
def barrier_obs(e_fn):
    def f(path):
        e = e_fn(path)
        return float(e[1:-1].max() - 0.5 * (e[0] + e[-1])) if len(e) > 2 else 0.0
    return f


# path observable, the index of the interior node whose energy is highest
def peak_pos_obs(e_fn):
    def f(path):
        e = e_fn(path)
        return float(1 + int(np.argmax(e[1:-1]))) if len(e) > 2 else 0.0
    return f


# path observable, the total number of residue substitutions the path makes over all its steps
def load_obs(path):
    return float(mutational_load(path))


# path observable, the sequence position substituted by the first step of the path
def first_mut_obs(path):
    return float(substitution_order(path)[0])


# path observable, the number of substitution pairs made out of ascending position order
def n_inversions_obs(path):
    order = substitution_order(path)
    return float(sum(order[i] > order[j]
                     for i in range(len(order)) for j in range(i + 1, len(order))))


def make_energy(allele, use_pwm):
    df = pd.read_csv(IEDB_MS_CSV)
    if use_pwm:
        seqs = [s for s in df[df['MHC allele'] == allele]['Peptide'] if len(s) == 9]
        enc = convert_number(seqs)
        counts = np.full((L, 20), 0.5)  # pseudocount
        for i in range(L):
            idx, c = np.unique(enc[:, i], return_counts=True)
            counts[i, idx] += c
        pwm = counts / counts.sum(axis=1, keepdims=True)
        print(f"{allele}: PWM from {len(seqs)} ligands")
        return (lambda s: -np.sum(np.log(pwm[np.arange(L), np.atleast_2d(s)]),
                                  axis=1)), df
    from config import PGM_SOURCE, PGM_UTILS
    for _p in (PGM_UTILS, PGM_SOURCE):
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)
    # entropy registers PGM's rbm under that bare name for the unpickler
    import entropy
    from b51_either_sweep import load_rbm, rbm_energy_batch
    rbm = load_rbm(allele)
    return (lambda s: rbm_energy_batch(np.atleast_2d(s), rbm)), df


def centred(E_fn, allele, df):
    '''Shift an energy so the allele's mean ligand energy sits at zero.'''
    seqs = [s for s in df[df['MHC allele'] == allele]['Peptide'] if len(s) == 9]
    off = float(np.mean(E_fn(convert_number(seqs))))
    return (lambda s, o=off: E_fn(s) - o), off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allele', default='HLA-B*51:01')
    ap.add_argument('--target', default=None,
                    help="second allele -> two-allele 'either' corridor (e.g. HLA-B*27:05)")
    ap.add_argument('--slack', type=int, default=0,
                    help='T = D + slack; slack > 0 gives an indirect path')
    ap.add_argument('--lam', type=float, default=0.5,
                    help='elastic Lambda for the indirect regime')
    ap.add_argument('--pwm', action='store_true', help='use the PWM energy (fast smoke run)')
    ap.add_argument('--chains', type=int, default=6)
    ap.add_argument('--iter', type=int, default=16000)
    ap.add_argument('--burn', type=int, default=3000)
    ap.add_argument('--thin', type=int, default=4)
    ap.add_argument('--beta', type=float, default=1.5)
    args = ap.parse_args()

    two = args.target is not None
    indirect = args.slack > 0
    model = 'PWM' if args.pwm else 'RBM'

    E_A_raw, df = make_energy(args.allele, args.pwm)

    if two:
        E_B_raw, _ = make_energy(args.target, args.pwm)
        cA, offA = centred(E_A_raw, args.allele, df)
        cB, offB = centred(E_B_raw, args.target, df)
        # picked the same way as the b51_either_sweep runs
        candA = natural_candidates(args.allele, df, cA, ANCH_A)
        candB = natural_candidates(args.target, df, cB, ANCH_B)
        pairs = pick_pairs(candA, candB, 1)
        if not pairs:
            raise SystemExit(f"no {args.allele} -> {args.target} endpoint pair found "
                             f"(anchor/complexity/Hamming filters too strict).")
        v0, vT, D = pairs[0]
        E_A_fn, E_B_fn, score = cA, cB, 'either'
        e_fn = lambda p: np.minimum(cA(p), cB(p))  # either energy at each node
    else:
        E_A_fn, E_B_fn, score = E_A_raw, None, 'both'
        v0, vT, D = choose_endpoints_escape(args.allele, df, E_A_raw, top_k=10)
        e_fn = E_A_raw

    T = D + args.slack
    kw = dict(beta=args.beta, beta_init=0.1, anneal='linear',
              n_burn=args.burn, n_iter=args.iter, n_thin=args.thin)
    if two:
        kw['score'] = 'either'
    if indirect:
        kw['T'], kw['Lambda'] = T, args.lam

    corridor = f"{args.allele} -> {args.target} (either)" if two else f"{args.allele}"
    regime = (f"indirect T=D+{args.slack}={T}, Lambda={args.lam}"
              if indirect else f"direct T=D={D}")
    print(f"\nv0={convert_letter(np.asarray(v0))[0]}  vT={convert_letter(np.asarray(vT))[0]}")
    print(f"Running {args.chains} chains, {model}, {corridor}, {regime}, "
          f"D={D}, beta={args.beta}, burn={args.burn}, iter={args.iter}, thin={args.thin}")

    samples, energies, infos = run_chains(
        v0, vT, E_A_fn, E_B_fn, n_chains=args.chains, base_seed=1, **kw)
    n_rec = energies.shape[1]
    print(f"recorded {n_rec} samples/chain (accept rates "
          f"{[round(i['accept_rate'], 2) for i in infos]})")

    obs = {'energy': energies}  # already a total for each path
    obs['barrier'] = np.array([scalar_series(s, barrier_obs(e_fn)) for s in samples])
    if two or indirect:
        obs['peak_pos'] = np.array([scalar_series(s, peak_pos_obs(e_fn)) for s in samples])
        obs['load'] = np.array([scalar_series(s, load_obs) for s in samples])
    else:
        obs['first_mut'] = np.array([scalar_series(s, first_mut_obs) for s in samples])
        obs['n_inv'] = np.array([scalar_series(s, n_inversions_obs) for s in samples])

    results = [diagnose(obs[k], thin=args.thin, name=k) for k in obs]
    report = format_report(results)
    print('\n' + report)

    # a route is a path's substitution order
    ent_mean = ent_mcse = float('nan')
    if not indirect:
        n_batch = 10
        ent_batches = []
        for s in samples:
            b = len(s) // n_batch
            for k in range(n_batch):
                block = s[k * b:(k + 1) * b]
                if block:
                    ent_batches.append(route_entropy(block)['S_path'])
        ent_mean, ent_mcse = dg.batch_means_mcse(ent_batches)
        n_routes = route_entropy([p for s in samples for p in s])['n_distinct']
        print(f"\n{'route entropy S_path':>24} = {ent_mean:.3f} +/- {ent_mcse:.3f} nats "
              f"(batch means, {len(ent_batches)} batches; {n_routes} distinct routes seen)")
    else:
        print("\nroute entropy S_path omitted (defined for direct paths only); the "
              "indirect route coordinate is 'load' (M), diagnosed above.")

    worst_rhat = max(r['split_rhat'] for r in results)
    min_ess = min(r['ess'] for r in results)
    ok = all(r['ok'] for r in results)
    print(f"\nVERDICT: {'CONVERGED' if ok else 'NOT converged'}"
          f"  (worst split-Rhat={worst_rhat:.4f}, min ESS={min_ess:.0f})")
    rec = max((r['rec_thin_steps'] or 0) for r in results)
    print(f"Recommended thinning from IAT: n_thin >= {rec} raw steps "
          f"(currently {args.thin}); recorded samples cover ~{n_rec * args.thin} steps.")

    _figure(energies, results, args, D, model, ent_mean, ent_mcse, two, indirect)


def _figure(energies, results, args, D, model, ent_mean, ent_mcse, two, indirect):
    '''Draw the diagnostics figure: traces, running means, autocorrelation, summary.'''
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    x = np.arange(energies.shape[1]) * args.thin

    for c in energies:
        ax[0, 0].plot(x, c, lw=0.5, alpha=0.7)
    ax[0, 0].set_title('(a) energy traces (all chains)', fontsize=FS_TITLE)
    ax[0, 0].set_xlabel('sampler step', fontsize=FS_LABEL)
    ax[0, 0].set_ylabel('path energy', fontsize=FS_LABEL)

    for c in energies:
        rm = np.cumsum(c) / np.arange(1, len(c) + 1)
        ax[0, 1].plot(x, rm, lw=1.0, alpha=0.8)
    ax[0, 1].set_title('(b) running mean of energy', fontsize=FS_TITLE)
    ax[0, 1].set_xlabel('sampler step', fontsize=FS_LABEL)
    ax[0, 1].set_ylabel('mean path energy', fontsize=FS_LABEL)

    rho = dg._autocov(energies[0]); rho = rho / rho[0]
    lags = np.arange(len(rho))
    show = min(len(rho), max(50, int(6 * results[0]['iat'])))
    ax[1, 0].axhline(0, color='0.6', lw=0.8)
    ax[1, 0].plot(lags[:show] * args.thin, rho[:show], color=BLUE, lw=1.2)
    ax[1, 0].axvline(results[0]['iat'] * args.thin, color=ORANGE, ls='--',
                     label=f"IAT tau={results[0]['iat']:.1f} samples")
    ax[1, 0].set_title('(c) energy autocorrelation (chain 1)', fontsize=FS_TITLE)
    ax[1, 0].set_xlabel('lag (sampler steps)', fontsize=FS_LABEL)
    ax[1, 0].set_ylabel('autocorrelation', fontsize=FS_LABEL)
    ax[1, 0].legend(fontsize=FS_TICK)

    ax[1, 1].axis('off')
    corridor = (f"{args.allele} -> {args.target} either" if two else f"{args.allele}")
    regime = (f"indirect T=D+{args.slack}, Lambda={args.lam}" if indirect else "direct T=D")
    lines = [f"{model}  {corridor}", f"{regime}   D={D}   {args.chains} chains x "
             f"{energies.shape[1]} samples", '']
    lines.append(f"{'obs':>10} {'split-Rhat':>10} {'ESS':>7} {'MCSE%':>7}")
    for r in results:
        lines.append(f"{r['name']:>10} {r['split_rhat']:>10.4f} {r['ess']:>7.0f} "
                     f"{100 * r['mcse_frac']:>6.1f}%")
    if np.isfinite(ent_mean):
        lines += ['', f"route entropy = {ent_mean:.3f} +/- {ent_mcse:.3f} nats"]
    lines += ['', 'good: split-Rhat <= 1.01 and ESS >= 400']
    ax[1, 1].text(0.0, 1.0, '\n'.join(lines), va='top', ha='left',
                  family='monospace', fontsize=FS_TICK, transform=ax[1, 1].transAxes)

    for a in ax.flat:
        a.tick_params(labelsize=FS_TICK)
    fig.suptitle('Path-sampler convergence diagnostics', fontsize=FS_TITLE + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    tag = args.allele.replace('*', '').replace(':', '')
    if two:
        tag += '_' + args.target.replace('*', '').replace(':', '')
    if indirect:
        tag += f'_indirect_k{args.slack}'
    out = os.path.join(FIGURES_DIR, f'sampler_diagnostics_{model}_{tag}.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == '__main__':
    main()
