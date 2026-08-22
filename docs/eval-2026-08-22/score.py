import json, glob, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
truth = json.load(open('truth.json'))

def approx(mv, tv):
    if isinstance(tv, str) and tv.startswith('~'):
        t = float(tv[1:]); return t != 0 and abs(mv - t)/t <= 0.05
    try: return float(mv) == float(tv)
    except (TypeError, ValueError): return False

print(f"{'model':34} {'match':>5} {'wrong':>5} {'miss':>5} {'invent':>6} {'val-acc':>8} {'precision':>9} {'recall':>7}")
for path in sorted(glob.glob('bench_*.json')):
    model = path[6:-5]
    res = json.load(open(path))
    ma = wr = mi = inv = 0
    wrongs, invents = [], []
    for fname, t in truth.items():
        m = res.get(fname, {})
        for k, tv in t.items():
            if k in m:
                if approx(m[k], tv): ma += 1
                else: wr += 1; wrongs.append((fname, k, m[k], tv))
            else: mi += 1
        for k, mv in m.items():
            if k not in t: inv += 1; invents.append((fname, k, mv))
    rep, pres = ma+wr, ma+wr+inv
    print(f"{model:34} {ma:>5} {wr:>5} {mi:>5} {inv:>6} {ma/rep*100 if rep else 0:>7.1f}% {ma/pres*100 if pres else 0:>8.1f}% {ma/(ma+wr+mi)*100:>6.1f}%")
    for f,k,mv,tv in wrongs[:8]: print(f"    WRONG {f}: {k} model={mv} truth={tv}")
    for f,k,mv in invents[:6]: print(f"    INVENT {f}: {k}={mv}")
