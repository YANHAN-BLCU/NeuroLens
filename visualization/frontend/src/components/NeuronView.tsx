// Neuron View Component - Network Visualization with Cytoscape.js

import React, { useEffect, useRef, useState } from 'react';
import cytoscape, { Core, Stylesheet } from 'cytoscape';
import {
  Box,
  Typography,
  Paper,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Drawer,
  IconButton,
  Divider,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

interface NeuronInfo {
  id: string;
  layer: number;
  neuron: number;
  quadrant: string;
  alignment: number;
  activation: number;
}

// Quadrant colors
const quadrantColors: Record<string, string> = {
  'S+A+': '#e74c3c',  // Red - Dangerous
  'S+A-': '#e67e22',  // Orange - To watch
  'S-A+': '#3498db',  // Blue - Protective
  'S-A-': '#27ae60',  // Green - Neutral
};

const MOCK_NEURONS: NeuronInfo[] = [
  { id: 'layer_31_neuron_4062', layer: 31, neuron: 4062, quadrant: 'S+A-', alignment: 0.85, activation: -0.006 },
  { id: 'layer_31_neuron_1200', layer: 31, neuron: 1200, quadrant: 'S-A+', alignment: -0.72, activation: 0.45 },
  { id: 'layer_31_neuron_3500', layer: 31, neuron: 3500, quadrant: 'S-A-', alignment: -0.15, activation: -0.12 },
  { id: 'layer_30_neuron_2000', layer: 30, neuron: 2000, quadrant: 'S+A+', alignment: 0.68, activation: 0.32 },
  { id: 'layer_30_neuron_1500', layer: 30, neuron: 1500, quadrant: 'S-A+', alignment: -0.55, activation: 0.28 },
  { id: 'layer_25_neuron_3000', layer: 25, neuron: 3000, quadrant: 'S-A-', alignment: -0.08, activation: -0.05 },
  { id: 'layer_20_neuron_2500', layer: 20, neuron: 2500, quadrant: 'S+A-', alignment: 0.42, activation: -0.18 },
  { id: 'layer_15_neuron_1800', layer: 15, neuron: 1800, quadrant: 'S-A+', alignment: -0.38, activation: 0.22 },
];

export const NeuronView: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const [selectedNeuron, setSelectedNeuron] = useState<NeuronInfo | null>(null);
  const [filterLayer, setFilterLayer] = useState<string>('all');
  const [filterQuadrant, setFilterQuadrant] = useState<string>('all');

  // Initialize Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;

    // Filter neurons
    let filteredNeurons = MOCK_NEURONS;
    if (filterLayer !== 'all') {
      filteredNeurons = filteredNeurons.filter(n => n.layer === parseInt(filterLayer));
    }
    if (filterQuadrant !== 'all') {
      filteredNeurons = filteredNeurons.filter(n => n.quadrant === filterQuadrant);
    }

    // Create elements
    const elements: any[] = [];
    
    // Add nodes
    filteredNeurons.forEach(neuron => {
      elements.push({
        data: {
          id: neuron.id,
          label: `L${neuron.layer}N${neuron.neuron}`,
          layer: neuron.layer,
          neuron: neuron.neuron,
          quadrant: neuron.quadrant,
        },
      });
    });

    // Add edges (mock gradient connections)
    for (let i = 0; i < filteredNeurons.length - 1; i++) {
      elements.push({
        data: {
          source: filteredNeurons[i].id,
          target: filteredNeurons[i + 1].id,
        },
      });
    }

    // Initialize Cytoscape
    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(quadrant)',
            'label': 'data(label)',
            'font-size': '10px',
            'width': 30,
            'height': 30,
            'color': '#333',
            'border-width': 2,
            'border-color': '#fff',
          },
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#ccc',
            'target-arrow-color': '#ccc',
            'target-arrow-shape': 'triangle',
            'opacity': 0.6,
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 3,
            'border-color': '#000',
          },
        },
      ] as Stylesheet[],
      layout: {
        name: 'concentric',
        concentric: (node: any) => -node.data('layer'),
        levelWidth: () => 1,
        padding: 20,
        animate: true,
      },
      minZoom: 0.5,
      maxZoom: 3,
    });

    // Click event
    cy.on('tap', 'node', (evt) => {
      const nodeId = evt.target.id();
      const neuron = MOCK_NEURONS.find(n => n.id === nodeId);
      if (neuron) setSelectedNeuron(neuron);
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
    };
  }, [filterLayer, filterQuadrant]);

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        Neuron View
      </Typography>

      {/* Controls */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Filter by Layer</InputLabel>
              <Select
                value={filterLayer}
                label="Filter by Layer"
                onChange={(e) => setFilterLayer(e.target.value)}
              >
                <MenuItem value="all">All Layers</MenuItem>
                <MenuItem value="31">Layer 31</MenuItem>
                <MenuItem value="30">Layer 30</MenuItem>
                <MenuItem value="25">Layer 25</MenuItem>
                <MenuItem value="20">Layer 20</MenuItem>
                <MenuItem value="15">Layer 15</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12} sm={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Filter by Quadrant</InputLabel>
              <Select
                value={filterQuadrant}
                label="Filter by Quadrant"
                onChange={(e) => setFilterQuadrant(e.target.value)}
              >
                <MenuItem value="all">All Quadrants</MenuItem>
                <MenuItem value="S+A+">S+A+ (Dangerous)</MenuItem>
                <MenuItem value="S+A-">S+A- (Watch)</MenuItem>
                <MenuItem value="S-A+">S-A+ (Protective)</MenuItem>
                <MenuItem value="S-A-">S-A- (Neutral)</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          {/* Legend */}
          <Grid item xs={12} sm={4}>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {Object.entries(quadrantColors).map(([quad, color]) => (
                <Chip
                  key={quad}
                  label={quad}
                  size="small"
                  sx={{ bgcolor: color, color: 'white' }}
                />
              ))}
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* Cytoscape Container */}
      <Paper sx={{ p: 2, height: 500, mb: 3 }}>
        <Box
          ref={containerRef}
          sx={{
            width: '100%',
            height: '100%',
            minHeight: 450,
          }}
        />
      </Paper>

      {/* Statistics Table */}
      <Typography variant="h6" sx={{ mb: 2 }}>
        Neuron Statistics
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Layer</TableCell>
              <TableCell>Neuron</TableCell>
              <TableCell>Quadrant</TableCell>
              <TableCell>Alignment</TableCell>
              <TableCell>Activation</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {MOCK_NEURONS.map((neuron) => (
              <TableRow
                key={neuron.id}
                hover
                onClick={() => setSelectedNeuron(neuron)}
                sx={{ cursor: 'pointer' }}
              >
                <TableCell>{neuron.id}</TableCell>
                <TableCell>{neuron.layer}</TableCell>
                <TableCell>{neuron.neuron}</TableCell>
                <TableCell>
                  <Chip
                    label={neuron.quadrant}
                    size="small"
                    sx={{ bgcolor: quadrantColors[neuron.quadrant], color: 'white' }}
                  />
                </TableCell>
                <TableCell>{neuron.alignment.toFixed(4)}</TableCell>
                <TableCell>{neuron.activation.toFixed(4)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Detail Drawer */}
      <Drawer
        anchor="right"
        open={!!selectedNeuron}
        onClose={() => setSelectedNeuron(null)}
      >
        <Box sx={{ width: 350, p: 3 }}>
          {selectedNeuron && (
            <>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Neuron Details</Typography>
                <IconButton onClick={() => setSelectedNeuron(null)}>
                  <CloseIcon />
                </IconButton>
              </Box>

              <Divider sx={{ mb: 2 }} />

              <Table size="small">
                <TableBody>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>ID</TableCell>
                    <TableCell>{selectedNeuron.id}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Layer</TableCell>
                    <TableCell>{selectedNeuron.layer}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Neuron</TableCell>
                    <TableCell>{selectedNeuron.neuron}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Quadrant</TableCell>
                    <TableCell>
                      <Chip
                        label={selectedNeuron.quadrant}
                        size="small"
                        sx={{ bgcolor: quadrantColors[selectedNeuron.quadrant], color: 'white' }}
                      />
                    </TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Alignment</TableCell>
                    <TableCell>{selectedNeuron.alignment.toFixed(4)}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell sx={{ fontWeight: 'bold' }}>Activation</TableCell>
                    <TableCell>{selectedNeuron.activation.toFixed(4)}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </>
          )}
        </Box>
      </Drawer>
    </Box>
  );
};

