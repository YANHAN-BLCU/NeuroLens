// Representation View Component - PCA/t-SNE Visualization

import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';
import {
  Box,
  Typography,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Slider,
  Chip,
  CircularProgress,
} from '@mui/material';
import { useStore } from '../store';
import { representationApi } from '../services/api';

const AVAILABLE_LAYERS = [0, 5, 10, 15, 20, 25, 30, 31];

export const RepresentationView: React.FC = () => {
  const { selectedLayer, setSelectedLayer, representationData, setRepresentationData } = useStore();
  const [method, setMethod] = useState<'pca' | 'tsne'>('pca');
  const [perplexity, setPerplexity] = useState<number>(30);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch representation data from API
  useEffect(() => {
    const fetchRepresentation = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const data = await representationApi.getRepresentation({
          layer_idx: selectedLayer,
          method,
          n_components: 2,
        });
        setRepresentationData(data);
      } catch (err) {
        console.error('Failed to fetch representation data:', err);
        setError('Failed to load representation data. Using mock data.');
        // Fallback to mock data on error
        generateMockData();
      } finally {
        setIsLoading(false);
      }
    };

    fetchRepresentation();
  }, [selectedLayer, method, perplexity]);

  // Generate mock data fallback
  const generateMockData = () => {
    const nSamples = 500;
    const x: number[] = [];
    const y: number[] = [];
    const labels: number[] = [];
    
    for (let i = 0; i < nSamples; i++) {
      const isToxic = i > nSamples / 2;
      
      if (method === 'pca') {
        const baseX = isToxic ? 2 + Math.random() * 2 : -2 - Math.random() * 2;
        const baseY = isToxic ? 1 + Math.random() : -1 - Math.random();
        x.push(baseX + (Math.random() - 0.5) * 1.5);
        y.push(baseY + (Math.random() - 0.5) * 1.5);
      } else {
        const angle = Math.random() * Math.PI * 2;
        const radius = isToxic ? 5 + Math.random() * 3 : 2 + Math.random() * 2;
        x.push(Math.cos(angle) * radius + (Math.random() - 0.5) * 0.5);
        y.push(Math.sin(angle) * radius + (Math.random() - 0.5) * 0.5);
      }
      
      labels.push(isToxic ? 1 : 0);
    }
    
    return { x, y, labels };
  };

  // Get data from API response or use mock data
  let data = representationData ? {
    x: representationData.coords.map(c => c[0]),
    y: representationData.coords.map(c => c[1]),
    labels: representationData.labels,
  } : null;

  // If no API data, use mock data
  if (!data && !isLoading) {
    const mock = generateMockData();
    data = mock;
  }

  // Separate safe and toxic points
  const safeX: number[] = [];
  const safeY: number[] = [];
  const toxicX: number[] = [];
  const toxicY: number[] = [];
  
  if (data) {
    data.labels.forEach((label, i) => {
      if (label === 0) {
        safeX.push(data.x[i]);
        safeY.push(data.y[i]);
      } else {
        toxicX.push(data.x[i]);
        toxicY.push(data.y[i]);
      }
    });
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        Representation View
      </Typography>

      {/* Loading State */}
      {isLoading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress />
        </Box>
      )}

      {/* Error State */}
      {error && (
        <Paper sx={{ p: 2, mb: 3, bgcolor: '#fff3e0' }}>
          <Typography color="warning.main">{error}</Typography>
        </Paper>
      )}

      {!isLoading && !data && (
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <Typography variant="h6" color="text.secondary">
            No representation data available.
          </Typography>
        </Box>
      )}

      {!isLoading && data && (
      <>
      <Paper sx={{ p: 2, mb: 3 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Layer</InputLabel>
              <Select
                value={selectedLayer}
                label="Layer"
                onChange={(e) => setSelectedLayer(e.target.value as number)}
              >
                {AVAILABLE_LAYERS.map((layer) => (
                  <MenuItem key={layer} value={layer}>
                    Layer {layer}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} sm={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Method</InputLabel>
              <Select
                value={method}
                label="Method"
                onChange={(e) => setMethod(e.target.value as 'pca' | 'tsne')}
              >
                <MenuItem value="pca">PCA</MenuItem>
                <MenuItem value="tsne">t-SNE</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          {method === 'tsne' && (
            <Grid item xs={12} sm={4}>
              <Typography gutterBottom>Perplexity: {perplexity}</Typography>
              <Slider
                value={perplexity}
                onChange={(_, value) => setPerplexity(value as number)}
                min={5}
                max={100}
                step={5}
                marks={[
                  { value: 5, label: '5' },
                  { value: 50, label: '50' },
                  { value: 100, label: '100' },
                ]}
                valueLabelDisplay="auto"
              />
            </Grid>
          )}
        </Grid>
        
        {/* Legend */}
        <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
          <Chip label="Safe Samples" color="success" variant="outlined" />
          <Chip label="Toxic Samples" color="error" variant="outlined" />
        </Box>
      </Paper>

      {/* Plot */}
      <Paper sx={{ p: 2, height: 600 }}>
        <Plot
          data={[
            {
              x: safeX,
              y: safeY,
              mode: 'markers',
              type: 'scatter',
              name: 'Safe',
              marker: {
                size: 8,
                color: '#4caf50',
                opacity: 0.6,
              },
            },
            {
              x: toxicX,
              y: toxicY,
              mode: 'markers',
              type: 'scatter',
              name: 'Toxic',
              marker: {
                size: 8,
                color: '#f44336',
                opacity: 0.6,
              },
            },
          ]}
          layout={{
            title: {
              text: `Layer ${selectedLayer} - ${method.toUpperCase()} Projection`,
              font: { size: 18, family: 'Times New Roman' },
            },
            xaxis: {
              title: { text: 'Dimension 1', font: { family: 'Times New Roman', size: 14 } },
              showgrid: true,
              zeroline: true,
            },
            yaxis: {
              title: { text: 'Dimension 2', font: { family: 'Times New Roman', size: 14 } },
              showgrid: true,
              zeroline: true,
            },
            hovermode: 'closest' as const,
            showlegend: true,
            legend: {
              x: 1,
              xanchor: 'right' as const,
              y: 1,
            },
            width: undefined,
            height: 550,
            autosize: true,
            paper_bgcolor: '#ffffff',
            plot_bgcolor: '#fafafa',
          }}
          useResizeHandler={true}
          style={{ width: '100%', height: '100%' }}
          config={{
            responsive: true,
            displayModeBar: true,
            toImageButtonOptions: {
              format: 'svg' as const,
              filename: `representation_layer${selectedLayer}_${method}`,
              height: 600,
              width: 800,
              scale: 2,
            },
          }}
        />
      </Paper>
      </>
      )}
    </Box>
  );
};

