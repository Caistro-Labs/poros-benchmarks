import re, sys, statistics, collections
d = collections.defaultdict(list); pk = {}
for l in open(sys.argv[1]):
    m = re.match(r"SPEED (\w+?)_(72|05)_r(\d+)_(fused|peft)_(\d) (?:rc \d+ )?med ([\d.]+).*peak GB ([\d.]+) ([\d.]+)", l)
    if m:
        _, w, r, c, rep, med, a, res = m.groups(); d[(w, int(r), c)].append(float(med)); pk[(w, int(r), c)] = (a, res)
for (w, r) in sorted({(k[0], k[1]) for k in d}, reverse=True):
    f, p = d[(w, r, "fused")], d[(w, r, "peft")]
    mf, mp = statistics.median(f), statistics.median(p)
    print(f"{w} r{r}: fused {f} med {mf:.3f} | peft {p} med {mp:.3f} | fused/peft {mf/mp:.3f} ({(1-mf/mp)*100:+.1f}% faster) | peak fused {pk[(w,r,'fused')]} peft {pk[(w,r,'peft')]}")
