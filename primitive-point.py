from collections import deque, Counter
from fractions import Fraction
from typing import Optional, Iterable, List, Tuple, Dict
import math

# -------------------- number theory --------------------
def factor_int(n: int) -> Dict[int, int]:
    f: Dict[int, int] = {}
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            f[p] = e
        p = 3 if p == 2 else p + 2
    if x > 1:
        f[x] = f.get(x, 0) + 1
    return f

def vp(n: int, p: int) -> int:
    e = 0
    while n % p == 0:
        n //= p
        e += 1
    return e

def divisors_from_factor(f: Dict[int, int]) -> List[int]:
    ds = [1]
    for p, e in f.items():
        cur = []
        pe = 1
        for _ in range(e + 1):
            for d in ds:
                cur.append(d * pe)
            pe *= p
        ds = cur
    return sorted(ds)

def gl2_fp_order(p: int) -> int:
    return (p * p - 1) * (p * p - p)

def gl2_zpk_order(p: int, k: int) -> int:
    # |GL2(Z/p^k Z)| for k>=0
    if k <= 0:
        return 1
    return (p ** (4 * (k - 1))) * gl2_fp_order(p)

def prime_factors(n: int) -> List[int]:
    return list(factor_int(n).keys())

# -------------------- modular matrices --------------------
def egcd(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return (a, 1, 0)
    g, x, y = egcd(b, a % b)
    return (g, y, x - (a // b) * y)

def inv_mod(a: int, mod: int) -> int:
    a %= mod
    g, x, _ = egcd(a, mod)
    if g != 1:
        raise ValueError("%d not invertible mod %d" % (a, mod))
    return x % mod

def normalize_gens_flat(G_gens_flat: List[List[int]], mod: int) -> List[Tuple[int, int, int, int]]:
    gens: List[Tuple[int, int, int, int]] = []
    for g in G_gens_flat:
        if len(g) != 4:
            raise ValueError("Generator must have 4 entries [a,b,c,d], got %r" % (g,))
        a, b, c, d = g
        gens.append((a % mod, b % mod, c % mod, d % mod))
    return gens

def reduce_gens(G_gens_flat: List[List[int]], mod_from: int, mod_to: int) -> List[List[int]]:
    if mod_from % mod_to != 0:
        raise ValueError("mod_to must divide mod_from")
    return [[g[0] % mod_to, g[1] % mod_to, g[2] % mod_to, g[3] % mod_to] for g in G_gens_flat]

def mat_mul(A: Tuple[int,int,int,int], B: Tuple[int,int,int,int], mod: int) -> Tuple[int,int,int,int]:
    a, b, c, d = A
    e, f, g, h = B
    return (
        (a * e + b * g) % mod,
        (a * f + b * h) % mod,
        (c * e + d * g) % mod,
        (c * f + d * h) % mod,
    )

def mat_inv(A: Tuple[int,int,int,int], mod: int) -> Tuple[int,int,int,int]:
    a, b, c, d = A
    det = (a * d - b * c) % mod
    det_inv = inv_mod(det, mod)
    return (
        ( d * det_inv) % mod,
        (-b * det_inv) % mod,
        (-c * det_inv) % mod,
        ( a * det_inv) % mod,
    )

def subgroup_order_GL2_mod(mod: int, gens_flat: List[List[int]], max_elements: Optional[int] = None) -> int:
    # GL2(Z/1Z) is the trivial group
    if mod == 1:
        return 1

    I = (1 % mod, 0, 0, 1 % mod)
    gens = normalize_gens_flat(gens_flat, mod)

    gens_full: List[Tuple[int,int,int,int]] = []
    for g in gens:
        gens_full.append(g)
        gens_full.append(mat_inv(g, mod))

    # dedup
    gens_full = list(dict.fromkeys(gens_full))

    seen = {I}
    q = deque([I])
    while q:
        x = q.popleft()
        for g in gens_full:
            y = mat_mul(x, g, mod)
            if y not in seen:
                seen.add(y)
                if max_elements is not None and len(seen) > max_elements:
                    raise RuntimeError("Group exceeded max_elements=%d at modulus=%d" % (max_elements, mod))
                q.append(y)
    return len(seen)

# -------------------- Algorithm 2: compute m0 (matches pseudocode) --------------------
def kernel_size_gl2_reduction(n: int, m0: int) -> int:
    """
    #ker(GL2(Z/nZ) -> GL2(Z/m0Z)) for m0|n.
    Works even when m0 drops primes entirely (em0=0).
    """
    if n % m0 != 0:
        raise ValueError("m0 must divide n")
    fn = factor_int(n)
    fm0 = factor_int(m0)
    out = 1
    for p, en in fn.items():
        em0 = fm0.get(p, 0)
        out *= gl2_zpk_order(p, en) // gl2_zpk_order(p, em0)
    return out

def heuristic_SE_from_GN(N: int, G_gens_flat: List[List[int]]) -> List[int]:
    SE = {2, 3}
    for p in factor_int(N).keys():
        gens_p = reduce_gens(G_gens_flat, N, p)
        ordGp = subgroup_order_GL2_mod(p, gens_p)
        if ordGp != gl2_fp_order(p):
            SE.add(p)
    return sorted(SE)

def compute_m0(N: int, G_gens_flat: List[List[int]],
               SE: Optional[Iterable[int]] = None,
               max_elements: Optional[int] = None) -> Tuple[int, int, List[int]]:
    """
    Algorithm 2 (pseudocode):
      1) n = ∏_{ℓ in SE} ℓ^{v_ℓ(N)}
      2) smallest m0 | n with |G(n)| = |G(m0)| * |ker(GL2(Z/nZ)->GL2(Z/m0Z))|
      3) return m0

    Returns (m0, n, SE_used).
    """
    if N <= 0:
        raise ValueError("N must be positive")

    SE_used = heuristic_SE_from_GN(N, G_gens_flat) if SE is None else sorted(set(SE))
    if 2 not in SE_used:
        SE_used = sorted(set(SE_used) | {2})
    if 3 not in SE_used:
        SE_used = sorted(set(SE_used) | {3})

    n = 1
    for ell in SE_used:
        e = vp(N, ell)
        if e > 0:
            n *= ell ** e

    if n == 1:
        return 1, 1, SE_used

    # cache subgroup orders for divisors
    order_cache: Dict[int, int] = {}

    def ord_at(mod: int) -> int:
        if mod == 1:
            return 1
        if mod not in order_cache:
            gens_mod = reduce_gens(G_gens_flat, N, mod)
            order_cache[mod] = subgroup_order_GL2_mod(mod, gens_mod, max_elements=max_elements)
        return order_cache[mod]

    Gn = ord_at(n)

    for m0 in divisors_from_factor(factor_int(n)):  # increasing => smallest m0
        Gm0 = ord_at(m0)
        ker = kernel_size_gl2_reduction(n, m0)
        if Gn == Gm0 * ker:
            return m0, n, SE_used

    raise RuntimeError("No m0 found.")

# -------------------- Primitive points (= sinks) at modulus m0 --------------------
def matvec(A: Tuple[int,int,int,int], v: Tuple[int,int], mod: int) -> Tuple[int,int]:
    a, b, c, d = A
    return ((a * v[0] + b * v[1]) % mod, (c * v[0] + d * v[1]) % mod)

def scalar_mul(k: int, v: Tuple[int,int], mod: int) -> Tuple[int,int]:
    return ((k * v[0]) % mod, (k * v[1]) % mod)

def orbit_of(v: Tuple[int,int], gens: List[Tuple[int,int,int,int]], mod: int):
    seen = {v}
    q = deque([v])
    while q:
        x = q.popleft()
        for A in gens:
            y = matvec(A, x, mod)
            if y not in seen:
                seen.add(y)
                q.append(y)
    return seen

def ord_coord(x: int, mod: int) -> int:
    x %= mod
    if x == 0:
        return 1
    return mod // math.gcd(mod, x)

def vec_order(v: Tuple[int,int], mod: int) -> int:
    oa = ord_coord(v[0], mod)
    ob = ord_coord(v[1], mod)
    return oa * ob // math.gcd(oa, ob)

def divisors(n: int) -> List[int]:
    ds = set()
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            ds.add(i)
            ds.add(n // i)
    return sorted(ds)

def deg_map_X1(n: int, a: int) -> Fraction:
    if n % a != 0:
        raise ValueError("a must divide n")
    b = n // a
    cf = Fraction(1, 2) if (a <= 2 and n > 2) else Fraction(1, 1)
    prod = Fraction(1, 1)
    for p in prime_factors(b):
        if a % p != 0:
            prod *= Fraction(p * p - 1, p * p)
    return cf * b * b * prod

def compute_primitive_points_from_group(m: int, G_gens_flat: List[List[int]]):
    G = normalize_gens_flat(G_gens_flat, m)
    minusI = (m - 1, 0, 0, m - 1)
    H_gens = G + [minusI]

    nonzero = [(a, b) for a in range(m) for b in range(m) if (a, b) != (0, 0)]
    orbit_id = {}
    orbits = []
    for v in nonzero:
        if v in orbit_id:
            continue
        O = orbit_of(v, H_gens, m)
        oid = len(orbits)
        orbits.append(O)
        for u in O:
            orbit_id[u] = oid

    oid_data = {}
    for oid, O in enumerate(orbits):
        rep = min(O)
        n = vec_order(rep, m)
        size = len(O)
        d = size if n <= 2 else size // 2
        oid_data[oid] = {"rep": rep, "n": n, "size": size, "d": d}

    edges = {oid: [] for oid in oid_data}
    for oid, dat in oid_data.items():
        rep, n, d = dat["rep"], dat["n"], dat["d"]
        for a in divisors(n)[:-1]:
            b = n // a
            img = scalar_mul(b, rep, m)
            if img == (0, 0):
                d_target = 1
            else:
                target_oid = orbit_id[img]
                d_target = oid_data[target_oid]["d"]
            degf = deg_map_X1(n, a)
            if Fraction(d, 1) == Fraction(d_target, 1) * degf:
                edges[oid].append(a)

    sink_oids = [oid for oid, outs in edges.items() if len(outs) == 0]

    sinks = []
    for oid in sorted(sink_oids, key=lambda t: (oid_data[t]["n"], oid_data[t]["d"], oid_data[t]["rep"])):
        dat = oid_data[oid]
        sinks.append({"n": dat["n"], "d": dat["d"], "orbit_size": dat["size"], "rep": dat["rep"]})

    deg_multiset = Counter((s["n"], s["d"]) for s in sinks)
    deg_multiset[(1, 1)] += 1

    return {
        "m": m,
        "num_orbits": len(orbits),
        "sinks": sinks,
        "deg_multiset": deg_multiset,
        "sink_count_including_level1": len(sink_oids) + 1,
    }

def compute_m0_and_primitive_points(N: int, G_gens_flat: List[List[int]],
                                   SE: Optional[Iterable[int]] = None,
                                   max_elements: Optional[int] = None):
    m0, n, SE_used = compute_m0(N, G_gens_flat, SE=SE, max_elements=max_elements)
    G_mod_m0 = reduce_gens(G_gens_flat, N, m0)
    prim = compute_primitive_points_from_group(m0, G_mod_m0)
    return m0, n, SE_used, prim

# -------------------- usage --------------------
if __name__ == "__main__":
    N = 544
    G_gens_flat = [[1, 0, 32, 1], [1, 32, 0, 1], [513, 32, 512, 33], [349, 84, 248, 169], [9, 32, 488, 345], [511, 134, 272, 1], [29, 8, 444, 85], [141, 24, 266, 373]]
    m0, n, SE_used, prim = compute_m0_and_primitive_points(N, G_gens_flat)

    print("SE_used =", SE_used)
    print("n =", n)
    print("m0 =", m0)
    print("#orbits (nonzero) =", prim["num_orbits"])
    print("#primitive points (incl level 1) =", prim["sink_count_including_level1"])
    print("Primitive degree multiset (n,d) incl (1,1):")
    for k in sorted(prim["deg_multiset"]):
        print(k, "x", prim["deg_multiset"][k])