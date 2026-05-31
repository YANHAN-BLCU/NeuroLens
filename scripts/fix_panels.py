"""Fix Panel G and H encoding by rewriting from scratch."""
import os

VIS_DIR = r'F:\NeuroLens2\visualization\backend\vis'

# ── Panel G: Sankey ──
panel_g = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sankey - Jailbreak Sample Tracing</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { padding: 16px; background: #fff; height: 100vh; overflow: auto; }
        .panel-header { font-size: 18px; font-weight: 600; margin-bottom: 12px; color: #333; }
        #sankey-chart { width: 100%; height: calc(100vh - 80px); min-height: 300px; }
    </style>
</head>
<body>
    <div class="panel-header">Sample Origin View G</div>
    <div id="sankey-chart"></div>
    <script>
        async function loadData() {
            try {
                const response = await fetch('/api/attack_paths');
                if (!response.ok) throw new Error('HTTP ' + response.status);
                const data = await response.json();
                if (data.error) throw new Error(data.error);
                const nodes = data.nodes || [];
                const links = data.links || [];
                if (nodes.length === 0) { console.log('No nodes'); return; }
                const labels = nodes.map(n => n.label || n.id);
                const nodeColors = nodes.map(n => {
                    if (n.type === 'attack') return '#EEC79F';
                    if (n.type === 'output') return '#A6CDE4';
                    return '#74B69F';
                });
                const sources = links.map(l => {
                    const idx = nodes.findIndex(n => n.id === l.source);
                    return idx >= 0 ? idx : 0;
                });
                const targets = links.map(l => {
                    const idx = nodes.findIndex(n => n.id === l.target);
                    return idx >= 0 ? idx : 0;
                });
                const values = links.map(l => l.value || 1);
                const plotlyData = {
                    type: 'sankey',
                    orientation: 'h',
                    node: { pad: 15, thickness: 20, line: { color: 'black', width: 0.5 }, label: labels, color: nodeColors },
                    link: { source: sources, target: targets, value: values }
                };
                Plotly.newPlot('sankey-chart', [plotlyData], {
                    title: 'Jailbreak Sample Tracing Path',
                    font: { size: 12 },
                    paper_bgcolor: 'white',
                    plot_bgcolor: 'white',
                    margin: { l: 10, r: 10, t: 40, b: 40 }
                }, {responsive: true});
            } catch (e) {
                console.error('Panel G load failed:', e);
            }
        }
        window.onload = function() { loadData(); };
    </script>
</body>
</html>'''

# ── Panel H: Violin ──
panel_h = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Violin - Neuron Activation Distribution</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { padding: 16px; background: #fff; height: 100vh; overflow: auto; }
        .panel-header { font-size: 18px; font-weight: 600; margin-bottom: 12px; color: #333; }
        #violin-chart { width: 100%; height: calc(100vh - 80px); min-height: 300px; }
    </style>
</head>
<body>
    <div class="panel-header">Activation Difference View H</div>
    <div id="violin-chart"></div>
    <script>
        async function loadData() {
            try {
                const response = await fetch('/api/neuron_activations?successful=true&failed=true');
                if (!response.ok) throw new Error('HTTP ' + response.status);
                const data = await response.json();
                if (data.error) throw new Error(data.error);
                const traces = [];
                const quadrants = ['S+A+', 'S-A-', 'S+A-', 'S-A+'];
                const colors = { successful: 'rgba(157, 208, 199, 0.7)', failed: 'rgba(217, 189, 216, 0.7)' };
                quadrants.forEach(q => {
                    const qData = data[q] || { successful: [], failed: [] };
                    const successful = (qData.successful || []).slice(0, 200);
                    const failed = (qData.failed || []).slice(0, 200);
                    if (successful.length > 0) {
                        traces.push({
                            type: 'violin',
                            x: Array(successful.length).fill(q + ' (Success)'),
                            y: successful,
                            name: q + ' - Success',
                            box: { visible: true },
                            meanline: { visible: true },
                            fillcolor: colors.successful,
                            line: { color: '#74B69F' },
                            side: 'negative',
                            width: 0.5
                        });
                    }
                    if (failed.length > 0) {
                        traces.push({
                            type: 'violin',
                            x: Array(failed.length).fill(q + ' (Failed)'),
                            y: failed,
                            name: q + ' - Failed',
                            box: { visible: true },
                            meanline: { visible: true },
                            fillcolor: colors.failed,
                            line: { color: '#D9BDD8' },
                            side: 'positive',
                            width: 0.5
                        });
                    }
                });
                if (traces.length === 0) {
                    console.log('No activation data');
                    return;
                }
                Plotly.newPlot('violin-chart', traces, {
                    title: 'Neuron Activation Distribution',
                    yaxis: { title: 'Activation Value' },
                    showlegend: true,
                    paper_bgcolor: 'white',
                    plot_bgcolor: 'white',
                    margin: { l: 60, r: 30, t: 50, b: 80 }
                }, {responsive: true});
            } catch (e) {
                console.error('Panel H load failed:', e);
            }
        }
        window.onload = function() { loadData(); };
    </script>
</body>
</html>'''

# Write files with correct UTF-8 encoding (no BOM)
with open(os.path.join(VIS_DIR, 'panel_G_sankey.html'), 'w', encoding='utf-8') as f:
    f.write(panel_g)
print('Panel G written')

with open(os.path.join(VIS_DIR, 'panel_H_violin.html'), 'w', encoding='utf-8') as f:
    f.write(panel_h)
print('Panel H written')

# Verify
for name, path in [('G', 'panel_G_sankey.html'), ('H', 'panel_H_violin.html')]:
    full = os.path.join(VIS_DIR, path)
    with open(full, 'r', encoding='utf-8') as f:
        content = f.read()
    import re
    title = re.search(r'<title>(.*?)</title>', content)
    print(f'Panel {name} title: {title.group(1) if title else "NOT FOUND"}')
