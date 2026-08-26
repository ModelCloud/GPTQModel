"""Build a D300-source-shaped, leakage-safe YAQA mix from unused source rows."""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = Path('/root/qvq-data/divergence300-sources')
D300 = Path('/root/qvq-data/divergence300-v1/divergence300-development.jsonl')
MODEL = '/monster/data/model/Llama-3.2-1B-Instruct'
OUT = ROOT / 'calibration_div300_sources.parquet'

def digest(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def normalized_digest(messages):
    """Hash user prompts after formatting normalization, not just raw JSON."""
    text = "\n".join(str(m.get("content", "")) for m in messages
                      if isinstance(m, dict) and m.get("role") == "user")
    text = re.sub(r"\s+", " ", text.casefold()).strip()
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()

def main():
    sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts'))
    import prepare_divergence300 as prep
    forbidden = set(re.findall(r'"prompt_sha256"\s*:\s*"([0-9a-f]{64})"', D300.read_text()))
    forbidden_normalized = set()
    for line in D300.read_text().splitlines():
        row = json.loads(line)
        forbidden_normalized.add(normalized_digest(row['messages']))
    base = pd.read_parquet(ROOT / 'calibration.parquet')
    for x in base.messages:
        messages = x.tolist() if hasattr(x, 'tolist') else x
        forbidden.add(digest(messages))
        forbidden_normalized.add(normalized_digest(messages))
    builders = [
        ('terminal_bench_2_1', prep._terminal_rows),
        ('swe_bench_verified', prep._swe_rows),
        ('matharena_2025_2026', prep._math_rows),
        ('multi_if_non_english', prep._multi_if_rows),
        ('longbench_v2', prep._longbench_rows),
    ]
    candidates = []
    for group, builder in builders:
        for row in builder(SOURCE_ROOT):
            # YAQA's model-side sequence cap is 16,384 tokens.  Cap source
            # payloads before hashing/counting so one LongBench context cannot
            # consume the entire target or trigger an oversized factor pass.
            messages = [{"role": str(m["role"]), "content": str(m["content"])[:12000]} for m in row['messages']]
            h = digest(messages)
            if h not in forbidden and normalized_digest(messages) not in forbidden_normalized:
                candidates.append({'messages': messages, 'source_group': group, 'source_id': row['source_id'], 'hash': h})
    # Stable interleaving prevents one very long source from consuming the budget.
    candidates.sort(key=lambda r: hashlib.sha256(('qvq-div300-cal\0'+r['source_group']+'\0'+str(r['source_id'])).encode()).hexdigest())
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    chosen, tokens = [], 0
    chosen_normalized = set()
    for row in candidates:
        encoded = tok.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=False)
        ids = encoded['input_ids'] if hasattr(encoded, '__getitem__') and 'input_ids' in encoded else encoded
        n = len(ids)
        if n < 8:
            continue
        nh = normalized_digest(row['messages'])
        if nh in chosen_normalized:
            continue
        chosen.append(row)
        chosen_normalized.add(nh)
        tokens += n
        if tokens >= 500_000:
            break
    pd.DataFrame({'messages': [r['messages'] for r in chosen]}).to_parquet(OUT, index=False)
    info = {'target_tokens':500000,'rows':len(chosen),'tokens':tokens,'candidate_rows':len(candidates),'excluded_d300_or_existing':len(forbidden),'excluded_normalized_d300_or_existing':len(forbidden_normalized),'source_counts':{g:sum(r['source_group']==g for r in chosen) for g,_ in builders},'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'d300_manifest_sha256':hashlib.sha256(D300.read_bytes()).hexdigest()}
    (ROOT/'calibration_div300_sources.json').write_text(json.dumps(info,indent=2,sort_keys=True)+'\n')
    print(json.dumps(info,indent=2))
if __name__ == '__main__': main()
