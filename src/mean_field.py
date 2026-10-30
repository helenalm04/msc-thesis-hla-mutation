# The mean-field treatment of the path measure, the analytic counterpart to
# the chain in sampler.py. The joint law over the interior nodes is replaced
# by a product over sites, and what survives of the coupling, each pattern's
# projection, is solved for self-consistently.

import os
import sys

import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.append(_SRC)


# One or more RBMs stacked into a single energy landscape. An RBM's effective
# energy, with the hidden units summed out, is a field term plus one cumulant
# generating function per hidden unit, evaluated at that unit's input. Those
# hidden units are the patterns the mean field couples through, so stacking
# models just concatenates their weight rows and tracks which rows came from
# which model.
class Landscape:
    def __init__(self, models, weights=None):
        if not isinstance(models, (list, tuple)):
            models = [models]
        if weights is None:
            weights = [1.0] * len(models)
        if len(weights) != len(models):
            raise ValueError('weights and models must have the same length')

        self.models = list(models)
        self.weights = [float(w) for w in weights]
        self.layers = [m.hlayer for m in models]

        self.g = np.zeros((models[0].n_v, models[0].n_cv), dtype=np.float64)
        Ws, coefs, self.slices = [], [], []
        row = 0
        for m, w in zip(models, self.weights):
            self.g += w * np.asarray(m.vlayer.fields, dtype=np.float64)
            Wm = np.asarray(m.weights, dtype=np.float64)  # (n_h, n_v, n_cv)
            Ws.append(Wm)
            coefs.append(np.full(Wm.shape[0], w))
            self.slices.append(slice(row, row + Wm.shape[0]))  # rows of W for this model
            row += Wm.shape[0]

        self.W = np.concatenate(Ws, axis=0)
        self.coef = np.concatenate(coefs)
        self.N = self.W.shape[1]
        self.q = self.W.shape[2]
        self.M = self.W.shape[0]

    @property
    def features(self):
        return self.W

    @property
    def exact_field(self):
        return self.g

    # psi is what the patterns contribute to the free energy, and its
    # gradient is the conjugate field the site chains then see
    def psi_and_grad(self, I, beta):
        bc = beta * self.coef[None, :]
        return (bc * self.gamma(I)).sum(axis=1), bc * self.dgamma(I)

    def inputs(self, V):
        V = np.atleast_2d(np.asarray(V, dtype=np.int64))
        return np.einsum('mln->nm', self.W[:, np.arange(self.N)[:, None], V.T])  # sums w[:, i, V[n, i]] over i

    # the hidden layer's cumulant generating function, and its derivative,
    # which is the mean activation of the unit. Both come from PGM.
    def gamma(self, I):
        return self._per_layer(I, 'cgf_from_inputs')

    def dgamma(self, I):
        return self._per_layer(I, 'mean_from_inputs')

    # PGM works in float32, so each model's block is handed over at that
    # precision and the result widened again
    def _per_layer(self, I, method):
        I = np.atleast_2d(np.asarray(I, dtype=np.float64))
        out = np.empty_like(I)
        for layer, sl in zip(self.layers, self.slices):
            block = np.ascontiguousarray(I[:, sl], dtype=np.float32)
            out[:, sl] = getattr(layer, method)(block).astype(np.float64)
        return out

    def energy(self, V):
        V = np.atleast_2d(np.asarray(V, dtype=np.int64))
        field = self.g[np.arange(self.N)[None, :], V].sum(axis=1)
        return -field - (self.coef * self.gamma(self.inputs(V))).sum(axis=1)

    def mean_energy(self, marg):
        '''Landscape energy of the marginals at each interior step.'''
        field = np.einsum('tia,ia->t', marg, self.g)
        I = np.einsum('tia,mia->tm', marg, self.W)
        return -field - (self.coef[None, :] * self.gamma(I)).sum(axis=1)


# The 'either' score, min(E_A, E_B), for the mean field. A hard minimum has
# no gradient at the crossover, so it is replaced by a soft minimum of
# sharpness kappa, which recovers the hard one as kappa grows. Each allele
# contributes its field as a variable of its own here, not just its patterns,
# because the soft minimum mixes the two energies non-linearly and the field
# can no longer be held outside that mixing.
class EitherLandscape:
    def __init__(self, rbm_A, rbm_B, kappa=8.0, offsets=(0.0, 0.0)):
        self.parts = [Landscape([rbm_A]), Landscape([rbm_B])]
        self.kappa = float(kappa)
        self.offsets = (float(offsets[0]), float(offsets[1]))
        self.N = self.parts[0].N
        self.q = self.parts[0].q

        blocks, self.slices, row = [], [], 0
        for p in self.parts:
            blocks.append(p.g[None, :, :])
            blocks.append(p.W)
            self.slices.append((row, row + 1, row + 1 + p.M))  # field row, then patterns
            row += 1 + p.M
        self._features = np.concatenate(blocks, axis=0)
        self.M = self._features.shape[0]

    @property
    def features(self):
        return self._features

    @property
    def exact_field(self):
        return np.zeros((self.N, self.q))

    def _energies(self, X):
        '''Energy of each allele and its gradient in the stacked variables.'''
        E, dE = [], []
        for p, (g0, i0, i1), off in zip(self.parts, self.slices, self.offsets):
            I = X[:, i0:i1]
            E.append(-X[:, g0] - (p.coef * p.gamma(I)).sum(axis=1) - off)
            dE.append((-np.ones((X.shape[0], 1)),
                       -p.coef[None, :] * p.dgamma(I)))
        return E, dE

    def psi_and_grad(self, X, beta):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        E, dE = self._energies(X)
        # the soft minimum, computed by shifting out the largest exponent
        u = -self.kappa * np.stack(E, axis=1)
        umax = u.max(axis=1, keepdims=True)
        w = np.exp(u - umax)
        s = w.sum(axis=1, keepdims=True)
        soft = -(umax[:, 0] + np.log(s[:, 0])) / self.kappa
        # p is how much each allele owns this node, and it weights the
        # gradient, so the chain follows whichever allele is winning there
        p = w / s

        grad = np.zeros_like(X)
        for k, ((g0, i0, i1), (dg, dI)) in enumerate(zip(self.slices, dE)):
            grad[:, g0:g0 + 1] = -beta * p[:, [k]] * dg
            grad[:, i0:i1] = -beta * p[:, [k]] * dI
        return -beta * soft, grad

    # the real score for scoring sequences, hard by default; the soft form is
    # only needed where a gradient is
    def energy(self, V, hard=True):
        E = np.stack([p.energy(V) - off
                      for p, off in zip(self.parts, self.offsets)], axis=1)
        if hard:
            return E.min(axis=1)
        u = -self.kappa * E
        umax = u.max(axis=1, keepdims=True)
        return -(umax[:, 0] + np.log(np.exp(u - umax).sum(axis=1))) / self.kappa

    def mean_energy(self, marg):
        X = np.einsum('tia,kia->tk', marg, self.features)
        E, _ = self._energies(X)
        u = -self.kappa * np.stack(E, axis=1)
        umax = u.max(axis=1, keepdims=True)
        return -(umax[:, 0] + np.log(np.exp(u - umax).sum(axis=1))) / self.kappa

    def allele_weights(self, marg):
        '''Softmax weight of the first allele at each interior step.'''
        X = np.einsum('tia,kia->tk', marg, self.features)
        E, _ = self._energies(X)
        u = -self.kappa * np.stack(E, axis=1)
        umax = u.max(axis=1, keepdims=True)
        w = np.exp(u - umax)
        return (w / w.sum(axis=1, keepdims=True))[:, 0]


# the elastic transfer matrix, ones on the diagonal and off on every other entry, applied to x
def _apply_transfer(x, off):
    return off * x.sum(axis=-1, keepdims=True) + (1.0 - off) * x


# With the projections held fixed the sites no longer talk to each other, so
# each one is an independent chain over its T-1 interior values and can be
# solved exactly. Every site is done at once here, in parallel across N.
def _forward_backward(node_logw, s0, sT, off):
    '''Exact marginals and edge overlaps for each site chain.'''
    Tm1, N, A = node_logw.shape
    T = Tm1 + 1

    eta = np.exp(node_logw - node_logw.max(axis=-1, keepdims=True))
    shift = node_logw.max(axis=-1)  # added back in logZ_sites below

    # the forward pass starts pinned on s0, and is rescaled at every step
    # because the products underflow quickly over a long path
    alpha = np.zeros((T + 1, N, A))
    alpha[0][np.arange(N), s0] = 1.0
    logscale = np.zeros(N)  # one scale per site
    for t in range(1, T):
        a = _apply_transfer(alpha[t - 1], off[t - 1]) * eta[t - 1]
        c = a.max(axis=-1, keepdims=True)
        alpha[t] = a / c
        logscale += np.log(c[:, 0])
    # the last step is pinned on sT, so only that one entry survives
    a = _apply_transfer(alpha[T - 1], off[T - 1])
    alpha[T] = np.zeros((N, A))
    alpha[T][np.arange(N), sT] = a[np.arange(N), sT]

    logZ_sites = np.log(alpha[T].sum(axis=-1)) + logscale + shift.sum(axis=0)

    bet = np.zeros((T + 1, N, A))
    bet[T][np.arange(N), sT] = 1.0
    for t in range(T - 1, 0, -1):
        b = _apply_transfer(bet[t + 1], off[t]) * eta[t - 1]
        bet[t] = b / b.max(axis=-1, keepdims=True)
    bet[0] = np.zeros((N, A))
    b = _apply_transfer(bet[1], off[0])
    bet[0][np.arange(N), s0] = b[np.arange(N), s0]

    # alpha and bet both carry the node weight, so divide one copy back out
    marg = alpha[1:T] * bet[1:T] / np.where(eta > 0, eta, 1.0)
    marg /= marg.sum(axis=-1, keepdims=True)

    stay = np.zeros((T, N))
    for t in range(T):
        num = (alpha[t] * bet[t + 1]).sum(axis=-1)  # M(a,a) = 1 makes this the overlap
        den = (alpha[t] * _apply_transfer(bet[t + 1], off[t])).sum(axis=-1)
        stay[t] = num / np.where(den > 0, den, 1.0)

    return marg, stay, logZ_sites


def solve(landscape, s0, sT, T, beta, Lambda, alphabets=None, elastic='cont', K=1.0,
          max_iter=500, tol=1e-6, damping=0.5, verbose=False, I_init=None):
    '''Solve the mean-field path equations for a path fixed at both ends.

    Args:
        alphabets (list[np.ndarray] | None): Per-site allowed residues, the
            full alphabet when None.
        elastic (str): Elastic potential. 'evo' holds Lambda fixed, 'cont'
            tunes it in an outer loop until the mean step size reaches K.
        K (float): Target mean mutations per step, read by 'cont' only.
        I_init (np.ndarray | None): Starting projections, shape (T-1, M);
            the straight line between the endpoints when None.

    Returns:
        res (dict): The mean-field solution. Keys:
            I_traj (T-1, M) pattern inputs Ibar_t^mu, m_traj the same over N,
            hat_m (T-1, M) conjugate fields, marg (T-1, L, q) site marginals,
            stay (T, L) edge overlaps per site, q_traj (T,) overlap per step,
            d_traj (T,) mean mutations per step, logZ, f_path the free energy
            per site, E_traj (T-1,) energy of the marginals, hat_q (T,) the
            elastic conjugate field, elastic, converged, n_iter, residual
            (final max |Ibar - <I>|), and the inputs D, T, beta and Lambda.
    '''
    s0 = np.asarray(s0, dtype=np.int64)
    sT = np.asarray(sT, dtype=np.int64)
    N, A, M = landscape.N, landscape.q, landscape.M
    if T < 1:
        raise ValueError('T must be at least 1')
    D = int((s0 != sT).sum())
    if T < D:
        raise ValueError(f'T = {T} cannot reach the endpoint at Hamming distance {D}')

    feats = landscape.features
    betag = beta * landscape.exact_field

    mask = np.zeros((N, A))
    if alphabets is not None:
        mask[:] = -np.inf
        for i, allowed in enumerate(alphabets):
            mask[i, np.asarray(allowed, dtype=np.int64)] = 0.0
    betag = betag + mask

    Ibar = np.zeros((T - 1, M))
    if I_init is not None:
        Ibar = np.array(I_init, dtype=float).reshape(T - 1, M)
    elif T > 1 and M:
        # start on the straight line between the endpoints
        interp = np.tile(s0, (T - 1, 1))
        order = np.flatnonzero(s0 != sT)
        for k, t in enumerate(np.linspace(0, 1, T + 1)[1:-1]):
            take = order[:int(round(t * len(order)))]
            interp[k, take] = sT[take]
        Ibar = np.einsum('kia->ak', feats[:, np.arange(N)[:, None], interp.T])

    def evaluate(I, off):
        psi, hat = landscape.psi_and_grad(I, beta)
        node_logw = betag[None, :, :] + np.tensordot(hat, feats, axes=(1, 0))
        mg, st, lZ_sites = _forward_backward(node_logw, s0, sT, off)
        lin_const = float((psi - (hat * I).sum(axis=1)).sum()) if M else 0.0
        return float(lin_const + lZ_sites.sum()), mg, st, hat

    def inner(Ibar, off):
        '''Iterate the projections to self-consistency at a fixed elastic field.'''
        logZ, marg, stay, hat_m = evaluate(Ibar, off)
        converged, it, step, delta = False, 0, float(damping), 0.0
        for it in range(1, max_iter + 1):
            I_new = np.einsum('tia,kia->tk', marg, feats) if M else Ibar
            delta = float(np.abs(I_new - Ibar).max()) if M else 0.0
            if delta < tol:
                converged = True
                break

            gain = 1e-10 * max(1.0, abs(logZ))  # must beat the float32 noise on ln Z_MF
            while step > 1e-12:
                trial = (1.0 - step) * Ibar + step * I_new
                lz, mg, st, hm = evaluate(trial, off)
                if lz > logZ + gain:
                    Ibar, logZ, marg, stay, hat_m = trial, lz, mg, st, hm
                    step = min(1.0, step * 1.1)
                    break
                step *= 0.5
            else:
                converged = True
                break
        return Ibar, logZ, marg, stay, hat_m, converged, it, delta

    off = np.full(T, float(np.exp(-beta * Lambda)))  # off-diagonal entry of the transfer matrix
    Ibar, logZ, marg, stay, hat_m, converged, it, delta = inner(Ibar, off)

    wall_ok = True
    if elastic == 'cont':
        wall_ok = False
        best = None
        for outer in range(400):
            d = N * (1.0 - stay.mean(axis=1))
            err = float(np.abs(d - K).max())
            if best is None or err < best[0]:
                best = (err, off.copy(), Ibar.copy(), logZ, marg, stay, hat_m)
            if err < 1e-3:
                wall_ok = True
                break
            dc = np.clip(d / N, 1e-9, 1 - 1e-9)
            tc = np.clip(K / N, 1e-9, 1 - 1e-9)
            step_q = np.log(tc / (1 - tc)) - np.log(dc / (1 - dc))  # d_t is logistic in ln off_t
            log_off = np.log(np.clip(off, 1e-300, None))
            log_off += np.clip(0.3 * step_q, -1.5, 1.5)
            off = np.exp(np.clip(log_off, -700, 50))
            Ibar, logZ, marg, stay, hat_m, converged, it, delta = inner(Ibar, off)
            if verbose and outer % 25 == 0:
                print(f'  cont outer {outer:3d}  max|<d>-K| = {err:.3e}')
        if not wall_ok:
            _, off, Ibar, logZ, marg, stay, hat_m = best
    elif elastic != 'evo':
        raise ValueError("elastic must be 'evo' or 'cont'")
    converged = bool(converged and wall_ok)

    q_traj = stay.mean(axis=1)
    return dict(
        I_traj=Ibar,
        m_traj=Ibar / N,  # everything else keeps I unnormalised
        hat_m=hat_m,
        marg=marg,
        stay=stay,
        q_traj=q_traj,
        d_traj=N * (1.0 - q_traj),
        logZ=logZ,
        f_path=float(-logZ / (beta * N)),
        E_traj=landscape.mean_energy(marg),
        hat_q=-np.log(np.clip(off, 1e-300, None)),
        elastic=elastic,
        converged=converged,
        n_iter=it,
        residual=delta,
        D=D,
        T=T,
        beta=beta,
        Lambda=Lambda,
    )


def d_direct_subspace(marg, s0, sT):
    Tm1, N, _ = marg.shape
    idx = np.arange(N)
    p_direct = marg[:, idx, s0].copy()
    off = s0 != sT
    p_direct[:, off] += marg[:, idx[off], sT[off]]
    return N - p_direct.sum(axis=1)


def d_direct_subspace_sd(marg, s0, sT):
    Tm1, N, _ = marg.shape
    idx = np.arange(N)
    p_direct = marg[:, idx, s0].copy()
    off = s0 != sT
    p_direct[:, off] += marg[:, idx[off], sT[off]]
    p_off = 1.0 - p_direct
    return np.sqrt((p_off * (1.0 - p_off)).sum(axis=1))
