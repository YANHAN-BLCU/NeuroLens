import json, urllib.request

# Check Panel F (layer_similarity)
r = urllib.request.urlopen('http://127.0.0.1:6008/api/layer_similarity')
data = json.loads(r.read())
matrix = data.get('matrix', [])
labels = data.get('layer_labels', [])
print(f'Panel F: {len(matrix)}x{len(matrix[0]) if matrix else 0} matrix')
print(f'  Labels: {len(labels)}, Sample: {labels[:3] if labels else "none"}')
if matrix:
    print(f'  First row sample: {matrix[0][:5]}')
    print(f'  Max value: {max(max(row) for row in matrix):.2f}')

# Check Panel G (attack_paths)
r2 = urllib.request.urlopen('http://127.0.0.1:6008/api/attack_paths')
data2 = json.loads(r2.read())
nodes = data2.get('nodes', [])
links = data2.get('links', [])
print(f'\nPanel G: {len(nodes)} nodes, {len(links)} links')
if nodes: print(f'  Node sample: {nodes[0]}')
if links: print(f'  Link sample: {links[0]}')

# Check Panel H (neuron_activations)
r3 = urllib.request.urlopen('http://127.0.0.1:6008/api/neuron_activations')
data3 = json.loads(r3.read())
print(f'\nPanel H keys: {list(data3.keys())}')
for k, v in data3.items():
    s = len(v.get('successful', []))
    f = len(v.get('failed', []))
    print(f'  {k}: {s} successful, {f} failed')
