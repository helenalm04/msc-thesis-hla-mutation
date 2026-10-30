# Autocorrelation time, effective sample size, split-Rhat and the Monte Carlo
# standard error, behind the Appendix table. The autocorrelation time uses
# Geyer's initial positive sequence, which truncates the sum over lags at the
# first non-positive term rather than letting long-lag noise accumulate.

import numpy as np


def _autocov(x):
    '''Biased autocovariance of a 1D series at every lag, by FFT.'''
    x = np.asarray(x, dtype=float)
    n = x.size
    x = x - x.mean()
    m = 1 << int(np.ceil(np.log2(2 * n - 1)))  # pad to a power of two
    f = np.fft.rfft(x, n=m)
    acov = np.fft.irfft(f * np.conjugate(f), n=m)[:n].real
    return acov / n


def autocorr_time(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return float('nan')
    acov = _autocov(x)
    if acov[0] == 0.0:  # constant series, tau is 1
        return 1.0
    rho = acov / acov[0]
    # pairing consecutive lags is what makes the sequence provably positive,
    # so the first term that is not gives a place to stop
    max_m = (n - 1) // 2
    gamma = rho[0:2 * max_m:2] + rho[1:2 * max_m + 1:2]  # gamma_m for each m below max_m
    neg = np.where(gamma <= 0.0)[0]
    end = int(neg[0]) if neg.size else gamma.size  # initial positive sequence
    if end == 0:
        return 1.0
    # the true sequence decreases, so force that on the estimate as well
    gamma = np.minimum.accumulate(gamma[:end])
    return max(2.0 * float(gamma.sum()) - 1.0, 1.0)


# the draws as a rectangular array, one row per chain and one column per recorded step
def _as_chains(chains):
    a = np.atleast_2d(np.asarray(chains, dtype=float))
    return a


def _var_plus(chains):
    '''Return the marginal variance var+ and the within chain variance W.'''
    a = _as_chains(chains)
    m, n = a.shape
    # W averages the spread inside each chain, B the spread between their
    # means. Chains that have not met push B up and so push var+ above W.
    W = a.var(axis=1, ddof=1).mean()
    if m > 1:
        B = n * a.mean(axis=1).var(ddof=1)
        var_plus = (n - 1) / n * W + B / n
    else:
        var_plus = W
    return float(var_plus), float(W)


def _rhat(chains):
    a = _as_chains(chains)
    m, n = a.shape
    if n < 2:
        return float('nan')
    var_plus, W = _var_plus(a)
    if W == 0.0:
        return 1.0 if var_plus == 0.0 else float('inf')
    return float(np.sqrt(var_plus / W))


# Splitting before comparing is what catches a single chain that drifts: its
# two halves then sit apart and count as chains that disagree.
def _split(chains):
    '''Split each chain into its two halves.'''
    a = _as_chains(chains)
    m, n = a.shape
    half = n // 2
    return np.concatenate([a[:, :half], a[:, n - half:]], axis=0)


# the potential scale reduction factor, pooled over within chain standard deviation, with each chain cut in half
def split_rhat(chains):
    return _rhat(_split(chains))


def ess(chains):
    a = _as_chains(chains)
    m, n = a.shape
    if n < 4:
        return float('nan')
    acov = np.array([_autocov(c) for c in a])  # (m, n), biased
    W = (acov[:, 0] * n / (n - 1)).mean()  # n/(n-1) undoes the bias in _autocov
    if W == 0.0:
        return float(m * n)
    var_plus, _ = _var_plus(a)
    mean_acov = acov.mean(axis=0)  # (n,)
    rho = 1.0 - (W - mean_acov) / var_plus
    rho[0] = 1.0
    max_t = (n - 1) // 2
    gamma = rho[0:2 * max_t:2] + rho[1:2 * max_t + 1:2]
    neg = np.where(gamma <= 0.0)[0]
    end = int(neg[0]) if neg.size else gamma.size
    if end == 0:
        tau = 1.0
    else:
        gamma = np.minimum.accumulate(gamma[:end])
        tau = max(2.0 * float(gamma.sum()) - 1.0, 1.0)
    return float(m * n / tau)


def mcse_mean(chains):
    a = _as_chains(chains)
    var_plus, _ = _var_plus(a)
    S = ess(a)
    if not np.isfinite(S) or S <= 0:
        return float('nan')
    return float(np.sqrt(var_plus / S))


def diagnose(chains, thin=1, name=''):
    '''Return the diagnostics for one scalar observable as a dict.'''
    a = _as_chains(chains)
    m, n = a.shape
    var_plus, _ = _var_plus(a)
    sd = float(np.sqrt(var_plus))
    S = ess(a)
    mc = mcse_mean(a)
    rh = split_rhat(a)
    iats = [autocorr_time(c) for c in a]
    iat = float(np.nanmean(iats)) if np.any(np.isfinite(iats)) else float('nan')
    rec_thin = int(np.ceil(iat) * thin) if np.isfinite(iat) else None
    ok = bool(np.isfinite(rh) and rh <= 1.01 and np.isfinite(S) and S >= 400)
    return {
        'name': name,
        'n_chains': int(m),
        'n_per_chain': int(n),
        'n_total': int(m * n),
        'mean': float(a.mean()),
        'sd': sd,
        'mcse': mc,
        'mcse_frac': float(mc / sd) if sd > 0 else float('nan'),
        'ess': S,
        'ess_per_chain': S / m if np.isfinite(S) else float('nan'),
        'split_rhat': rh,
        'iat': iat,
        'rec_thin_steps': rec_thin,
        'ok': ok,
    }


def batch_means_mcse(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float(v.mean()) if v.size else float('nan'), float('nan'))
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def run_chains(s0, sT, E_A_fn, E_B_fn=None, n_chains=4, base_seed=0, **kwargs):
    from sampler import run_sampler
    all_samples, all_energies, infos = [], [], []
    for j in range(n_chains):
        samp, ener, _acc, info = run_sampler(
            s0, sT, E_A_fn, E_B_fn, seed=base_seed + j, **kwargs)
        all_samples.append(samp)
        all_energies.append(np.asarray(ener, dtype=float))
        infos.append(info)
    lengths = {len(e) for e in all_energies}
    if len(lengths) != 1:
        raise ValueError(f"chains recorded different sample counts: {lengths}")
    return all_samples, np.array(all_energies), infos


# the trace of one scalar path observable over the recorded samples of a chain
def scalar_series(samples, obs_fn):
    return np.array([float(obs_fn(np.stack(p))) for p in samples], dtype=float)


def format_report(results):
    head = (f"{'observable':>16} {'mean':>9} {'mcse':>8} {'mcse%':>6} "
            f"{'ESS':>7} {'ESS/ch':>7} {'split-Rhat':>10} {'IAT':>6} "
            f"{'rec_thin':>8}  verdict")
    lines = [head, '-' * len(head)]
    for r in results:
        verdict = 'OK' if r['ok'] else 'CHECK'
        flags = []
        if not (np.isfinite(r['split_rhat']) and r['split_rhat'] <= 1.01):
            flags.append('Rhat>1.01')
        if not (np.isfinite(r['ess']) and r['ess'] >= 400):
            flags.append('ESS<400')
        tag = verdict + (('' + ','.join(flags)) if flags else '')
        rec = r['rec_thin_steps'] if r['rec_thin_steps'] is not None else -1
        lines.append(
            f"{r['name']:>16} {r['mean']:>9.3f} {r['mcse']:>8.4f} "
            f"{100 * r['mcse_frac']:>5.1f}% {r['ess']:>7.0f} {r['ess_per_chain']:>7.0f} "
            f"{r['split_rhat']:>10.4f} {r['iat']:>6.1f} {rec:>8d}  {tag}")
    return '\n'.join(lines)
