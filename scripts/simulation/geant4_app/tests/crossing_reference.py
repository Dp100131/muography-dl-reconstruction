#!/usr/bin/env python3
"""OE1 stage 1, test R1: independent reference for include/RegionCrossing.hh.

Region Z of geo-train-06 = conduit (cylinder) AND dense body (ellipsoid),
original catalog primitives, world frame, metres. Three references, none of
which shares code with the C++ detector:

  1. a vectorised float64 numpy implementation (closest-approach form of the
     quadratics) -- compared on every segment, tolerance 1e-9 m;
  2. an arbitrary-precision (decimal, 60 digits) evaluation of the same
     geometry on a random subset, which also covers near-tangent chords;
  3. a dense-sampling membership oracle on a subset: K = 200,001 points per
     segment with the margin() convention of the director's auditar_oe1.py
     (inside <=> max(margin_conduit, margin_body) < 0), compared with
     |d| * fraction_inside within 2 |d| / (K - 1).

Usage:
  crossing_reference.py gen     --out SEG.csv [--n 100000] [--seed 20260930]
  crossing_reference.py compare SEG.csv EVAL.csv [--dense 2000] [--exact 2000]

SEG.csv rows are "ax,ay,az,bx,by,bz" (metres). EVAL.csv is the output of
`crossing_eval SEG.csv` (columns lo,hi,len,deep,graze,lc,le,near). Requires
numpy (>= 1.20); nothing else. Exit status 0 = all comparisons within tolerance.

Tolerance. The plan asks +-1e-9 m. A chord that grazes a surface at depth
delta has length ~ 2 sqrt(2 S delta) (S = radius of curvature scale), and in
IEEE double the depth of a line relative to a primitive is known only to
~ u X, where u = 2^-53 and X is the size of the coordinates that enter the
computation (the endpoints of the chord: rounding them moves the line by
~ u |coordinate|). The length of a near-tangent chord therefore carries an
unavoidable error of ~ 4 u S X / len. Both this numpy model and the C++ code
show it (errors of 5e-9 to 8e-9 m at len = 1 mm and X = 135 to 220 m against
the 60-digit value); it is physically irrelevant but it means +-1e-9 m cannot
be demanded of millimetre-long, near-tangent chords. The acceptance used here
is therefore
    tol(len, X) = 1e-9 m + 8 u S X / max(len, 1e-12 m),   S = 72.9 m,
    X = max(S, largest |coordinate| of the two endpoints of the segment),
capped at len + 1e-9 m. (Plan correction W1: the first version used X = S for
every segment, which fails for segments whose endpoints are a few hundred
metres or kilometres from the origin: seeds 12 and 13 of N = 1e5.) The
statistics printed show how many segments needed more than 1e-9 m.

near. The C++ column `near` (chord meets both primitives grown by NEAR_M = 1 m)
is compared with this numpy model; the segments whose flag changes when the
margin moves by +-1e-9 m are ambiguous and excluded.
"""
import argparse
import csv
import decimal
import math
import sys

import numpy as np

CYL = dict(cx=20.3, cy=-24.5, r=21.0, z0=0.0, z1=340.0)
ELL = dict(cx=32.2, cy=-74.1, cz=137.8, ax=72.9, ay=61.6, az=64.6)
EPS = 1.0e-6
NEAR_M = 1.0    # oe1::kNearZMarginM
TOL_M = 1.0e-9  # plan: +-1e-9 m
U = 1.1102230246251565e-16  # 2^-53
S_MAX = 72.9    # largest primitive scale (m)


def coord_scale(segs):
    """Per-segment X of tol_len: the largest |coordinate| of its endpoints (m)."""
    return np.abs(np.asarray(segs, dtype=float)).max(axis=-1)


def tol_len(length, xmax=None):
    """Acceptance on a length (m), with the near-tangent conditioning term.

    ``xmax`` is the largest |endpoint coordinate| of the segment (see the module
    docstring); without it the coordinates are taken to be of the size of the
    primitives, S_MAX. X is never taken below S_MAX.
    """
    length = np.asarray(length, dtype=float)
    x = S_MAX if xmax is None else np.maximum(np.asarray(xmax, dtype=float), S_MAX)
    cond = 8.0 * U * S_MAX * x / np.maximum(length, 1e-12)
    return np.minimum(TOL_M + cond, length + TOL_M)


# ----------------------------------------------------------------- generation
def gen_segments(n, seed):
    rng = np.random.default_rng(seed)
    zlo = np.array([max(CYL['cx'] - CYL['r'], ELL['cx'] - ELL['ax']),
                    max(CYL['cy'] - CYL['r'], ELL['cy'] - ELL['ay']),
                    max(CYL['z0'], ELL['cz'] - ELL['az'])])
    zhi = np.array([min(CYL['cx'] + CYL['r'], ELL['cx'] + ELL['ax']),
                    min(CYL['cy'] + CYL['r'], ELL['cy'] + ELL['ay']),
                    min(CYL['z1'], ELL['cz'] + ELL['az'])])
    w = 20.0

    def in_box(m):
        return zlo - w + rng.random((m, 3)) * (zhi - zlo + 2 * w)

    def unit(m):
        d = rng.normal(size=(m, 3))
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    n_tan = n // 20                 # 5 %: tangent / near-tangent chords
    n_box = (n - n_tan) // 2        # 47.5 %: both ends in the widened box
    n_short = (n - n_tan) * 3 // 10  # 28.5 %: 0.05-3 m steps
    n_long = n - n_tan - n_box - n_short  # 19 %: 0.5-3 km lines

    segs = []
    segs.append(np.hstack([in_box(n_box), in_box(n_box)]))
    p = in_box(n_short)
    segs.append(np.hstack([p, p + unit(n_short) * (0.05 + rng.random((n_short, 1)) * 2.95)]))
    p = in_box(n_long)
    d = unit(n_long)
    segs.append(np.hstack([p - d * (500 + rng.random((n_long, 1)) * 2500),
                           p + d * (500 + rng.random((n_long, 1)) * 2500)]))

    # Tangent chords. Half to the cylinder wall, half to the ellipsoid surface,
    # at depth +-10^U(-9,-3) m, tangent point restricted to the other primitive.
    rows = []
    while len(rows) < n_tan:
        delta = 10 ** rng.uniform(-9, -3) * rng.choice([-1.0, 1.0])
        length = rng.uniform(1.0, 100.0)
        if len(rows) % 2 == 0:
            phi = rng.uniform(0, 2 * math.pi)
            z = rng.uniform(CYL['z0'], CYL['z1'])
            nrm = np.array([math.cos(phi), math.sin(phi), 0.0])
            tan = np.array([-math.sin(phi), math.cos(phi), 0.0])
            surf = np.array([CYL['cx'], CYL['cy'], z]) + CYL['r'] * nrm
            if not (ellipsoid_margin(surf) < 0):
                continue
            if rng.random() < 0.5:
                tan = np.array([0.0, 0.0, 1.0])
            base = surf - delta * nrm  # delta > 0: inside
        else:
            th, ph = rng.uniform(0, math.pi), rng.uniform(0, 2 * math.pi)
            u = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
            surf = np.array([ELL['cx'] + ELL['ax'] * u[0], ELL['cy'] + ELL['ay'] * u[1],
                             ELL['cz'] + ELL['az'] * u[2]])
            if not (cylinder_margin(surf) < 0):
                continue
            nrm = np.array([u[0] / ELL['ax'], u[1] / ELL['ay'], u[2] / ELL['az']])
            nrm /= np.linalg.norm(nrm)
            tan = np.cross(nrm, rng.normal(size=3))
            tan /= np.linalg.norm(tan)
            base = surf - delta * nrm
        rows.append(np.concatenate([base - length * tan, base + length * tan]))
    segs.append(np.array(rows))
    out = np.vstack(segs)
    return out[rng.permutation(len(out))]


def cylinder_margin(p):
    c = np.array([CYL['cx'], CYL['cy'], 0.5 * (CYL['z0'] + CYL['z1'])])
    s = np.array([CYL['r'], CYL['r'], 0.5 * (CYL['z1'] - CYL['z0'])])
    v = (p - c) / s
    return max(v[0] ** 2 + v[1] ** 2 - 1, abs(v[2]) - 1)


def ellipsoid_margin(p):
    c = np.array([ELL['cx'], ELL['cy'], ELL['cz']])
    s = np.array([ELL['ax'], ELL['ay'], ELL['az']])
    v = (p - c) / s
    return float((v * v).sum() - 1)


# ----------------------------------------------------- float64 vectorised model
def _ellipsoid_interval(a, d, e):
    """t-interval of a + t d inside an axis-aligned ellipsoid (closest approach)."""
    c = np.array([e['cx'], e['cy'], e['cz']])
    s = np.array([e['ax'], e['ay'], e['az']])
    u = (a - c) / s
    w = d / s
    ww = (w * w).sum(axis=1)
    uw = (u * w).sum(axis=1)
    cr = np.cross(u, w)
    cross2 = (cr * cr).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        tstar = -uw / ww
        # squared distance-to-centre at the closest approach = |u x w|^2 / |w|^2
        half2 = (1.0 - cross2 / ww) / ww
        half = np.sqrt(np.where(half2 >= 0, half2, np.nan))
        lo, hi = tstar - half, tstar + half
    degenerate = ww == 0
    inside0 = (u * u).sum(axis=1) <= 1.0
    lo = np.where(degenerate, np.where(inside0, -np.inf, np.inf), lo)
    hi = np.where(degenerate, np.where(inside0, np.inf, -np.inf), hi)
    empty = np.isnan(lo)
    lo = np.where(empty, np.inf, lo)
    hi = np.where(empty, -np.inf, hi)
    return lo, hi


def _cylinder_interval(a, d, cyl):
    if not (cyl['r'] > 0 and cyl['z1'] >= cyl['z0']):
        n = len(a)
        return np.full(n, np.inf), np.full(n, -np.inf)
    px = a[:, 0] - cyl['cx']
    py = a[:, 1] - cyl['cy']
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    A = dx * dx + dy * dy
    with np.errstate(divide='ignore', invalid='ignore'):
        tstar = -(px * dx + py * dy) / A
        cross = np.abs(px * dy - py * dx)
        sA = np.sqrt(A)
        # (r sqrt(A) - |cross|)(r sqrt(A) + |cross|) / A^2 = half-width^2
        half2 = (cyl['r'] * sA - cross) * (cyl['r'] * sA + cross) / (A * A)
        half = np.sqrt(np.where(half2 >= 0, half2, np.nan))
        rlo, rhi = tstar - half, tstar + half
    vertical = A == 0
    inside0 = px * px + py * py <= cyl['r'] ** 2
    rlo = np.where(vertical, np.where(inside0, -np.inf, np.inf), rlo)
    rhi = np.where(vertical, np.where(inside0, np.inf, -np.inf), rhi)
    empty = np.isnan(rlo)
    rlo = np.where(empty, np.inf, rlo)
    rhi = np.where(empty, -np.inf, rhi)
    with np.errstate(divide='ignore', invalid='ignore'):
        t0 = (cyl['z0'] - a[:, 2]) / dz
        t1 = (cyl['z1'] - a[:, 2]) / dz
    flat = dz == 0
    inz = (a[:, 2] >= cyl['z0']) & (a[:, 2] <= cyl['z1'])
    slo = np.where(flat, np.where(inz, -np.inf, np.inf), np.minimum(t0, t1))
    shi = np.where(flat, np.where(inz, np.inf, -np.inf), np.maximum(t0, t1))
    return np.maximum(rlo, slo), np.minimum(rhi, shi)


def _shrink(cyl, ell, delta):
    c2 = dict(cyl, r=cyl['r'] + delta, z0=cyl['z0'] - delta, z1=cyl['z1'] + delta)
    e2 = dict(ell, ax=ell['ax'] + delta, ay=ell['ay'] + delta, az=ell['az'] + delta)
    return c2, e2


def model(segs, eps=EPS):
    a, b = segs[:, :3], segs[:, 3:]
    d = b - a
    chord = np.sqrt((d * d).sum(axis=1))
    clo, chi = _cylinder_interval(a, d, CYL)
    elo, ehi = _ellipsoid_interval(a, d, ELL)
    lc = np.clip(np.minimum(chi, 1.0) - np.maximum(clo, 0.0), 0.0, None) * chord
    le = np.clip(np.minimum(ehi, 1.0) - np.maximum(elo, 0.0), 0.0, None) * chord
    zlo = np.maximum(np.maximum(clo, elo), 0.0)
    zhi = np.minimum(np.minimum(chi, ehi), 1.0)
    hit = zlo <= zhi
    length = np.where(zhi > zlo, zhi - zlo, 0.0) * chord
    # deep: positive length inside Z with every primitive shrunk by eps.
    ci, ei = _shrink(CYL, ELL, -eps)
    ilo = np.maximum(np.maximum(_cylinder_interval(a, d, ci)[0], _ellipsoid_interval(a, d, ei)[0]), 0.0)
    ihi = np.minimum(np.minimum(_cylinder_interval(a, d, ci)[1], _ellipsoid_interval(a, d, ei)[1]), 1.0)
    deep = (chord > 0) & (ihi > ilo)
    co, eo = _shrink(CYL, ELL, +eps)
    olo = np.maximum(np.maximum(_cylinder_interval(a, d, co)[0], _ellipsoid_interval(a, d, eo)[0]), 0.0)
    ohi = np.minimum(np.minimum(_cylinder_interval(a, d, co)[1], _ellipsoid_interval(a, d, eo)[1]), 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        band = 2.0 * eps / chord
    graze = (~deep) & (chord > 0) & (olo <= ohi) & (olo > band) & (ohi < 1.0 - band)
    return dict(hit=hit, lo=zlo, hi=zhi, len=length, deep=deep, graze=graze, lc=lc, le=le,
                chord=chord)


def near_flags(segs, margin=NEAR_M):
    """The chord meets BOTH primitives grown by ``margin`` (oe1::RegionZ nearZ flag)."""
    a, b = segs[:, :3], segs[:, 3:]
    d = b - a
    cyl_g, ell_g = _shrink(CYL, ELL, margin)   # a positive delta grows both primitives
    clo, chi = _cylinder_interval(a, d, cyl_g)
    elo, ehi = _ellipsoid_interval(a, d, ell_g)
    return (np.maximum(clo, 0.0) <= np.minimum(chi, 1.0)) & (np.maximum(elo, 0.0) <= np.minimum(ehi, 1.0))


# ------------------------------------------------ arbitrary-precision reference
decimal.getcontext().prec = 60
D = decimal.Decimal


def _exact_interval_quadric(a, d, centre, scale):
    """t-interval of a + t d inside sum(((p-c)/s)^2) <= 1, Decimal arithmetic."""
    u = [(D(a[i]) - D(centre[i])) / D(scale[i]) for i in range(3)]
    w = [D(d[i]) / D(scale[i]) for i in range(3)]
    ww = sum(x * x for x in w)
    uw = sum(x * y for x, y in zip(u, w))
    uu = sum(x * x for x in u)
    if ww == 0:
        return (D('-Infinity'), D('Infinity')) if uu <= 1 else (D('Infinity'), D('-Infinity'))
    disc = uw * uw - ww * (uu - 1)
    if disc < 0:
        return D('Infinity'), D('-Infinity')
    s = disc.sqrt()
    return (-uw - s) / ww, (-uw + s) / ww


def _exact_interval_cylinder(a, d, cyl):
    px, py = D(a[0]) - D(cyl['cx']), D(a[1]) - D(cyl['cy'])
    dx, dy = D(d[0]), D(d[1])
    A = dx * dx + dy * dy
    r = D(cyl['r'])
    if A == 0:
        radial = (D('-Infinity'), D('Infinity')) if px * px + py * py <= r * r else (D('Infinity'), D('-Infinity'))
    else:
        bh = px * dx + py * dy
        disc = bh * bh - A * (px * px + py * py - r * r)
        if disc < 0:
            radial = (D('Infinity'), D('-Infinity'))
        else:
            s = disc.sqrt()
            radial = ((-bh - s) / A, (-bh + s) / A)
    dz = D(d[2])
    z0, z1 = D(cyl['z0']), D(cyl['z1'])
    if dz == 0:
        slab = (D('-Infinity'), D('Infinity')) if z0 <= D(a[2]) <= z1 else (D('Infinity'), D('-Infinity'))
    else:
        t0, t1 = (z0 - D(a[2])) / dz, (z1 - D(a[2])) / dz
        slab = (min(t0, t1), max(t0, t1))
    return max(radial[0], slab[0]), min(radial[1], slab[1])


def exact_len(seg, cyl=CYL, ell=ELL):
    a, b = seg[:3], seg[3:]
    d = [D(b[i]) - D(a[i]) for i in range(3)]
    chord = sum(x * x for x in d).sqrt()
    clo, chi = _exact_interval_cylinder(a, d, cyl)
    elo, ehi = _exact_interval_quadric(a, d, (ell['cx'], ell['cy'], ell['cz']),
                                       (ell['ax'], ell['ay'], ell['az']))
    lo = max(clo, elo, D(0))
    hi = min(chi, ehi, D(1))
    return float(max(hi - lo, D(0)) * chord)


# ---------------------------------------------------------- dense-sampling oracle
def dense_len(seg, k=200001):
    a, b = seg[:3], seg[3:]
    t = np.linspace(0.0, 1.0, k)
    p = a[None, :] + t[:, None] * (b - a)[None, :]
    c = np.array([CYL['cx'], CYL['cy'], 0.5 * (CYL['z0'] + CYL['z1'])])
    s = np.array([CYL['r'], CYL['r'], 0.5 * (CYL['z1'] - CYL['z0'])])
    v = (p - c) / s
    mc = np.maximum(v[:, 0] ** 2 + v[:, 1] ** 2 - 1, np.abs(v[:, 2]) - 1)
    ce = np.array([ELL['cx'], ELL['cy'], ELL['cz']])
    se = np.array([ELL['ax'], ELL['ay'], ELL['az']])
    ve = (p - ce) / se
    me = (ve * ve).sum(axis=1) - 1
    inside = np.maximum(mc, me) < 0
    chord = float(np.linalg.norm(b - a))
    return chord * inside.sum() / (k - 1), chord, 2.0 * chord / (k - 1)


# ---------------------------------------------------------------- comparison
def read_segments(path):
    rows = []
    with open(path, newline='') as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0].isalpha() or line[0] == '#':
                continue
            rows.append([float(x) for x in line.split(',')])
    return np.array(rows, dtype=float)


def read_eval(path):
    with open(path, newline='') as fh:
        rd = csv.reader(fh)
        header = next(rd)
        assert header == ['lo', 'hi', 'len', 'deep', 'graze', 'lc', 'le', 'near'], header
        rows = [[float(x) for x in r] for r in rd]
    return np.array(rows, dtype=float)


def cmd_gen(args):
    segs = gen_segments(args.n, args.seed)
    with open(args.out, 'w', newline='\n') as fh:
        fh.write('ax,ay,az,bx,by,bz\n')
        for row in segs:
            fh.write(','.join(repr(float(x)) for x in row) + '\n')
    print(f'wrote {len(segs)} segments to {args.out} (seed {args.seed})')
    return 0


def cmd_compare(args):
    segs = read_segments(args.segments)
    ev = read_eval(args.eval)
    n = len(segs)
    if len(ev) != n:
        print(f'FAIL: {len(ev)} eval rows for {n} segments')
        return 1
    ref = model(segs)
    fails = 0

    def report(name, bad, worst):
        nonlocal fails
        status = 'PASS' if bad == 0 else 'FAIL'
        if bad:
            fails += 1
        print(f'{status} {name}: {bad} violations, worst {worst:.3g}')

    xseg = coord_scale(segs)   # X of tol(len, X): largest |endpoint coordinate| per segment
    d_len = np.abs(ev[:, 2] - ref['len'])
    tol = tol_len(ref['len'], xseg)
    over = d_len > TOL_M
    report(f'R1 numpy reference: |len - ref| <= tol(len, X) over {n} segments '
           f'({int(over.sum())} exceed the plain {TOL_M:g} m; all of them have len <= '
           f'{float(ref["len"][over].max()) if over.any() else 0.0:.3g} m)',
           int((d_len > tol).sum()), float(d_len.max()))
    d_lc = np.abs(ev[:, 5] - ref['lc'])
    d_le = np.abs(ev[:, 6] - ref['le'])
    report('R1 numpy reference: lc (bare conduit length)',
           int((d_lc > tol_len(ref['lc'], xseg)).sum()), float(d_lc.max()))
    report('R1 numpy reference: le (bare dense-body length)',
           int((d_le > tol_len(ref['le'], xseg)).sum()), float(d_le.max()))
    both = ~np.isnan(ev[:, 0]) & ref['hit'] & (ref['chord'] > 0)
    chord_b = np.maximum(ref['chord'][both], 1e-300)
    tol_t = tol_len(ref['len'][both], xseg[both]) / chord_b
    d_lo = np.abs(ev[both, 0] - ref['lo'][both])
    d_hi = np.abs(ev[both, 1] - ref['hi'][both])
    report('R1 numpy reference: lo / hi parameters (tolerance tol(len) / chord)',
           int(((d_lo > tol_t) | (d_hi > tol_t)).sum()),
           float(max((d_lo * chord_b).max(), (d_hi * chord_b).max())))
    # hit flag: the C++ prints nan for lo/hi when the chord misses Z
    hit_cpp = ~np.isnan(ev[:, 0])
    hit_bad = hit_cpp != ref['hit']
    # a hit of zero length (a tangent point) is numerically fragile: exclude
    fragile = (ref['len'] < 1e-9)
    report('R1 numpy reference: hit flag (segments with reference length >= 1e-9 m)',
           int((hit_bad & ~fragile).sum()), float((hit_bad & ~fragile).sum()))

    # deep / graze flags: only where the reference is not within 1e-11 m of
    # the eps thresholds (the flags are discontinuous functions of the input).
    ci, ei = _shrink(CYL, ELL, -EPS)
    co, eo = _shrink(CYL, ELL, +EPS)

    def margin_len(cy, el):
        a, d = segs[:, :3], segs[:, 3:] - segs[:, :3]
        lo = np.maximum(np.maximum(_cylinder_interval(a, d, cy)[0], _ellipsoid_interval(a, d, el)[0]), 0.0)
        hi = np.minimum(np.minimum(_cylinder_interval(a, d, cy)[1], _ellipsoid_interval(a, d, el)[1]), 1.0)
        return (hi - lo) * ref['chord']

    shrunk_len = margin_len(ci, ei)
    grown_len = margin_len(co, eo)
    amb_deep = np.abs(shrunk_len) < 1e-9
    amb_graze = (np.abs(grown_len) < 1e-9) | amb_deep
    bad_deep = (ev[:, 3].astype(bool) != ref['deep']) & ~amb_deep
    bad_graze = (ev[:, 4].astype(bool) != ref['graze']) & ~amb_graze
    report(f'R1 numpy reference: deep flag ({int(amb_deep.sum())} ambiguous excluded)',
           int(bad_deep.sum()), float(bad_deep.sum()))
    report(f'R1 numpy reference: graze flag ({int(amb_graze.sum())} ambiguous excluded)',
           int(bad_graze.sum()), float(bad_graze.sum()))
    # near flag (chord meets both primitives grown by NEAR_M); flags that change when
    # the margin moves by 1e-9 m are ambiguous
    near_ref = near_flags(segs, NEAR_M)
    amb_near = near_flags(segs, NEAR_M - 1e-9) != near_flags(segs, NEAR_M + 1e-9)
    bad_near = (ev[:, 7].astype(bool) != near_ref) & ~amb_near
    report(f'R1 numpy reference: near flag ({int(amb_near.sum())} ambiguous excluded; '
           f'{int(near_ref.sum())} near)', int(bad_near.sum()), float(bad_near.sum()))
    report('R1 near contains every chord with positive length in Z (numpy and C++)',
           int(((ref['len'] > 0) & ~near_ref).sum() + ((ev[:, 2] > 0) & (ev[:, 7] == 0)).sum()),
           float(((ev[:, 2] > 0) & (ev[:, 7] == 0)).sum()))
    both_out = (ref['len'] > 0) & (ref['lo'] > 0) & (ref['hi'] < 1)
    print(f'INFO segments: {n}; with len > 0: {int((ref["len"] > 0).sum())}; deep: '
          f'{int(ref["deep"].sum())}; graze: {int(ref["graze"].sum())}; '
          f'both ends outside Z (len > 0): {int(both_out.sum())}')

    rng = np.random.default_rng(12345)
    if args.exact > 0:
        pos = np.flatnonzero(ref['len'] > 0)
        sel = np.concatenate([rng.choice(pos, size=min(args.exact // 2, len(pos)), replace=False),
                              rng.choice(n, size=args.exact // 2, replace=False),
                              np.flatnonzero(d_len > 0.5 * TOL_M)])  # plus every doubtful one
        sel = np.unique(sel)
        worst, worst_plain, bad, over_plain = 0.0, 0.0, 0, 0
        for i in sel:
            e = exact_len(segs[i])
            err = abs(ev[i, 2] - e)
            worst = max(worst, err)
            bad += err > float(tol_len(e, xseg[i]))
            over_plain += err > TOL_M
            worst_plain = max(worst_plain, err if e >= 0.05 else 0.0)
        report(f'R1 60-digit reference: |len - exact| <= tol(len, X) on {len(sel)} segments '
               f'({over_plain} exceed the plain {TOL_M:g} m; worst error for len >= 5 cm: '
               f'{worst_plain:.3g} m)', bad, worst)
    if args.dense > 0:
        pos = np.flatnonzero(ref['len'] > 0)
        sel = np.concatenate([rng.choice(pos, size=min(args.dense // 2, len(pos)), replace=False),
                              rng.choice(n, size=args.dense // 2, replace=False)])
        bad, worst = 0, 0.0
        for i in sel:
            est, chord, res = dense_len(segs[i])
            err = abs(ev[i, 2] - est)
            worst = max(worst, err / res if res > 0 else 0.0)
            bad += err > res
        report(f'R1 dense-sampling oracle (K=200001): |len - oracle| <= 2|d|/(K-1) on {len(sel)} segments',
               int(bad), worst)
    print('R1 OVERALL', 'PASS' if fails == 0 else 'FAIL')
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('gen')
    g.add_argument('--out', required=True)
    g.add_argument('--n', type=int, default=100000)
    g.add_argument('--seed', type=int, default=20260930)
    c = sub.add_parser('compare')
    c.add_argument('segments')
    c.add_argument('eval')
    c.add_argument('--dense', type=int, default=2000)
    c.add_argument('--exact', type=int, default=2000)
    args = ap.parse_args()
    return cmd_gen(args) if args.cmd == 'gen' else cmd_compare(args)


if __name__ == '__main__':
    sys.exit(main())
