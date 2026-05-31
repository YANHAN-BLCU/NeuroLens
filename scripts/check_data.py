"""Check available data for panels."""
import json
from collections import Counter

with open(r'F:\NeuroLens2\outputs\Qwen2.5-1.5B-Instruct\test_snip\assessment\evaluation_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data.get('results', [])
print('Total results:', len(results))
print('Fields:', list(results[0].keys()) if results else 'none')

# Count by category
cats = Counter(r.get('category', 'unknown') for r in results)
print('\nCategories:', dict(cats.most_common(10)))

methods = Counter(r.get('method', 'unknown') for r in results)
print('\nMethods:', dict(methods.most_common(10)))

# ASR by category
cat_asr = {}
for r in results:
    cat = r.get('category', 'unknown')
    if cat not in cat_asr:
        cat_asr[cat] = {'total': 0, 'jailbreak': 0}
    cat_asr[cat]['total'] += 1
    if r.get('jailbreak_success'):
        cat_asr[cat]['jailbreak'] += 1

print('\nASR by category:')
for cat, stats in sorted(cat_asr.items(), key=lambda x: x[1]['jailbreak']/max(x[1]['total'],1), reverse=True)[:10]:
    asr = stats['jailbreak'] / max(stats['total'], 1)
    print(f'  {cat}: {asr:.2%} ({stats["jailbreak"]}/{stats["total"]})')

# ASR by method
method_asr = {}
for r in results:
    m = r.get('method', 'unknown')
    if m not in method_asr:
        method_asr[m] = {'total': 0, 'jailbreak': 0}
    method_asr[m]['total'] += 1
    if r.get('jailbreak_success'):
        method_asr[m]['jailbreak'] += 1

print('\nASR by method:')
for m, stats in sorted(method_asr.items(), key=lambda x: x[1]['jailbreak']/max(x[1]['total'],1), reverse=True)[:10]:
    asr = stats['jailbreak'] / max(stats['total'], 1)
    print(f'  {m}: {asr:.2%} ({stats["jailbreak"]}/{stats["total"]})')
