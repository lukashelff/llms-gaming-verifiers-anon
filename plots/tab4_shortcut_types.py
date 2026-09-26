import sys, csv, json, glob, re, collections, os
sys.path.insert(0, '.')
from IPT.ipt_verifier import extract_hypothesis_with_meta
NEG=re.compile(r"\\\+\s*westbound\s*\(", re.I); WB=re.compile(r"\bwestbound\b", re.I); CONST=re.compile(r"\b(?:train\d+|car\d+_\d+)\b")
# index model_name -> completions
idx={}
for f in glob.glob('output/eval-openai/*/model_outputs.json')+glob.glob('output/eval-oss/*/*/model_outputs.json'):
    if '.orig32k' in f: continue
    try: d=json.load(open(f))
    except Exception as e: continue
    if isinstance(d,dict):
        d=d.get('outputs') or d.get('results') or d.get('data') or list(d.values())
        if d and isinstance(d[0],list): d=d[0]
    if not d or not isinstance(d[0],dict): continue
    name=d[0].get('model_name') or os.path.basename(os.path.dirname(f))
    idx.setdefault(name,{})
    for r in d:
        if not isinstance(r,dict) or 'problem_id' not in r: continue
        idx[name][str(r['problem_id'])]=r.get('model_completion') or r.get('completion') or ''
want=['gpt-5','gpt-5-mini','gpt-5-nano','gpt-oss-120b-effort-medium','gpt-oss-20b-effort-medium','gpt-oss-20b','Olmo-3.1-32B-Think','Olmo-3-32B-Think','Olmo-3-7B-Think']
seen=set()
for det in ["output/eval-openai/ipt_results_paper_v1/detailed_results.csv","output/eval-oss/ipt_results_paper_v1/detailed_results.csv","output/eval-oss/olmo/ipt_results_paper_v1/detailed_results.csv","output/eval-openai/ipt_results/detailed_results.csv","output/eval-oss/ipt_results/detailed_results.csv"]:
    if not os.path.exists(det): continue
    rows=[r for r in csv.DictReader(open(det)) if r['is_reward_shortcut'] in ('True','true','1')]
    by=collections.defaultdict(list)
    for r in rows: by[r['model_name']].append(r)
    for m,rs in by.items():
        if m in seen: continue
        if not any(m==w or m.startswith(w) for w in want): continue
        comp=idx.get(m)
        if comp is None:
            cands=[k for k in idx if k.lower().replace('_','-')==m.lower().replace('_','-')]
            comp=idx[cands[0]] if cands else None
        if comp is None: print(m,'no outputs found', det); continue
        seen.add(m)
        c=collections.Counter(); ex={}
        for r in rs:
            txt=comp.get(str(r['problem_id']),'')
            try: h=extract_hypothesis_with_meta(txt)[0]
            except Exception: h=''
            h=h or ''
            t='label_negation' if (NEG.search(h) or WB.search(h)) else ('identifier_enum' if CONST.search(h) else 'other')
            c[t]+=1; ex.setdefault(t,(r['problem_id'],h[:100].replace('\n',' ')))
        print(f"{m:32s} [{det.split('/')[1]}/{det.split('/')[2]}] shortcuts={len(rs):3d}  label_negation={c['label_negation']:3d}  identifier_enum={c['identifier_enum']:3d}  other={c['other']:2d}")
        for t,(pid,h) in ex.items(): print(f"      e.g. {t} #{pid}: {h}")
