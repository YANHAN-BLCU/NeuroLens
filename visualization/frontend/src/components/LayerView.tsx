// Layer View Component - Sankey Diagram and Layer Evolution

import React, { useEffect, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Card,
  CardContent,
  CircularProgress,
} from '@mui/material';
import { useStore } from '../store';
import { layerApi } from '../services/api';
import { LayerEvolutionData } from '../types';

interface LayerData {
  layer: number;
  safe_count: number;
  toxic_count: number;
  safe_ratio: number;
  val_acc: number;
  val_roc_auc?: number;
  mean_projection_safe?: number;
  mean_projection_toxic?: number;
}

const MOCK_LAYER_DATA: LayerData[] = [
  { layer: 0, safe_count: 1000, toxic_count: 500, safe_ratio: 0.67, val_acc: 0.72 },
  { layer: 5, safe_count: 980, toxic_count: 520, safe_ratio: 0.65, val_acc: 0.78 },
  { layer: 10, safe_count: 960, toxic_count: 540, safe_ratio: 0.64, val_acc: 0.82 },
  { layer: 15, safe_count: 950, toxic_count: 550, safe_ratio: 0.63, val_acc: 0.85 },
  { layer: 20, safe_count: 940, toxic_count: 560, safe_ratio: 0.63, val_acc: 0.88 },
  { layer: 25, safe_count: 930, toxic_count: 570, safe_ratio: 0.62, val_acc: 0.91 },
  { layer: 30, safe_count: 920, toxic_count: 580, safe_ratio: 0.61, val_acc: 0.93 },
  { layer: 31, safe_count: 915, toxic_count: 585, safe_ratio: 0.61, val_acc: 0.94 },
];

export const LayerView: React.FC = () => {
  const { layerEvolution, setLayerEvolution } = useStore();
  const [layerData, setLayerData] = useState<LayerData[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Fetch layer evolution data from API
  useEffect(() => {
    const fetchLayerEvolution = async () => {
      setIsLoading(true);
      try {
        const data = await layerApi.getLayerEvolution();
        setLayerEvolution(data);
        
        // Transform API response to array format
        const layers = Object.entries(data).map(([key, value]) => {
          const layerNum = parseInt(key.replace('layer_', ''));
          return {
            layer: layerNum,
            safe_count: value.safe_count,
            toxic_count: value.toxic_count,
            safe_ratio: value.safe_ratio,
            val_acc: value.val_acc || 0,
            val_roc_auc: value.val_roc_auc,
            mean_projection_safe: value.mean_projection_safe,
            mean_projection_toxic: value.mean_projection_toxic,
          };
        }).sort((a, b) => a.layer - b.layer);
        
        setLayerData(layers);
      } catch (error) {
        console.error('Failed to fetch layer evolution data:', error);
        // Fallback to mock data on error
        setLayerData(MOCK_LAYER_DATA);
      } finally {
        setIsLoading(false);
      }
    };

    if (!layerEvolution) {
      fetchLayerEvolution();
    } else {
      // Transform cached data
      const layers = Object.entries(layerEvolution).map(([key, value]) => {
        const layerNum = parseInt(key.replace('layer_', ''));
        return {
          layer: layerNum,
          safe_count: value.safe_count,
          toxic_count: value.toxic_count,
          safe_ratio: value.safe_ratio,
          val_acc: value.val_acc || 0,
          val_roc_auc: value.val_roc_auc,
          mean_projection_safe: value.mean_projection_safe,
          mean_projection_toxic: value.mean_projection_toxic,
        };
      }).sort((a, b) => a.layer - b.layer);
      setLayerData(layers);
    }
  }, [layerEvolution, setLayerEvolution]);

  // Sankey diagram option
  const getSankeyOption = () => {
    const nodes = layerData.map(d => ({
      name: `Layer ${d.layer}`,
    }));
    
    const links: Array<{
      source: string;
      target: string;
      value: number;
    }> = [];
    
    for (let i = 0; i < layerData.length - 1; i++) {
      links.push({
        source: `Layer ${layerData[i].layer}`,
        target: `Layer ${layerData[i + 1].layer}`,
        value: layerData[i].safe_count,
      });
    }

    return {
      title: {
        text: 'Sample Flow Across Layers',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontFamily: 'Times New Roman',
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'item',
        triggerOn: 'mousemove',
      },
      series: [
        {
          type: 'sankey',
          layout: 'none',
          emphasis: {
            focus: 'adjacency',
          },
          data: nodes,
          links: links,
          orient: 'horizontal',
          label: {
            position: 'right',
            fontFamily: 'Times New Roman',
          },
          lineStyle: {
            color: 'gradient',
            curveness: 0.5,
          },
          itemStyle: {
            color: '#5470c6',
          },
        },
      ],
    };
  };

  // Line chart option for safe/toxic ratio
  const getLineChartOption = () => {
    return {
      title: {
        text: 'Safe/Toxic Ratio Across Layers',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontFamily: 'Times New Roman',
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'axis',
      },
      legend: {
        bottom: 10,
        data: ['Safe Ratio', 'Validation Accuracy'],
        textStyle: {
          fontFamily: 'Times New Roman',
        },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: layerData.map(d => `L${d.layer}`),
        axisLabel: {
          fontFamily: 'Times New Roman',
        },
      },
      yAxis: [
        {
          type: 'value',
          name: 'Ratio',
          max: 1,
          nameTextStyle: {
            fontFamily: 'Times New Roman',
          },
        },
        {
          type: 'value',
          name: 'Accuracy',
          max: 1,
          nameTextStyle: {
            fontFamily: 'Times New Roman',
          },
        },
      ],
      series: [
        {
          name: 'Safe Ratio',
          type: 'line',
          data: layerData.map(d => d.safe_ratio),
          smooth: true,
          lineStyle: { width: 2 },
          itemStyle: { color: '#4caf50' },
          areaStyle: { opacity: 0.2 },
        },
        {
          name: 'Validation Accuracy',
          type: 'line',
          yAxisIndex: 1,
          data: layerData.map(d => d.val_acc),
          smooth: true,
          lineStyle: { width: 2 },
          itemStyle: { color: '#2196f3' },
        },
      ],
    };
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        Layer View
      </Typography>

      {isLoading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {!isLoading && layerData.length === 0 && (
        <Typography variant="body1" color="text.secondary" sx={{ py: 4 }}>
          No layer evolution data available. Please run the pipeline first.
        </Typography>
      )}

      {!isLoading && layerData.length > 0 && (
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <Card sx={{ bgcolor: '#e3f2fd' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Total Layers
              </Typography>
              <Typography variant="h4" color="primary">
                {layerData.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card sx={{ bgcolor: '#e8f5e9' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Best Accuracy
              </Typography>
              <Typography variant="h4" color="success.main">
                {(layerData[layerData.length - 1].val_acc * 100).toFixed(1)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card sx={{ bgcolor: '#fff3e0' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Avg Safe Ratio
              </Typography>
              <Typography variant="h4" color="warning.main">
                {(layerData.reduce((a, b) => a + b.safe_ratio, 0) / layerData.length * 100).toFixed(1)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2, height: 400 }}>
            <ReactECharts option={getLineChartOption()} style={{ height: '100%', width: '100%' }} />
          </Paper>
        </Grid>
        <Grid item xs={12}>
          <Paper sx={{ p: 2, height: 400 }}>
            <ReactECharts option={getSankeyOption()} style={{ height: '100%', width: '100%' }} />
          </Paper>
        </Grid>
      </Grid>

      {/* Data Table */}
      <Typography variant="h6" sx={{ mt: 4, mb: 2 }}>
        Layer Statistics
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Layer</TableCell>
              <TableCell>Safe Samples</TableCell>
              <TableCell>Toxic Samples</TableCell>
              <TableCell>Safe Ratio</TableCell>
              <TableCell>Val Accuracy</TableCell>
              <TableCell>Status</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {layerData.map((row) => (
              <TableRow key={row.layer}>
                <TableCell>Layer {row.layer}</TableCell>
                <TableCell>{row.safe_count}</TableCell>
                <TableCell>{row.toxic_count}</TableCell>
                <TableCell>{(row.safe_ratio * 100).toFixed(1)}%</TableCell>
                <TableCell>{(row.val_acc * 100).toFixed(1)}%</TableCell>
                <TableCell>
                  <Chip 
                    label={row.val_acc > 0.9 ? 'Best' : row.val_acc > 0.8 ? 'Good' : 'Low'} 
                    color={row.val_acc > 0.9 ? 'success' : row.val_acc > 0.8 ? 'primary' : 'default'} 
                    size="small" 
                  />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      )}
    </Box>
  );
};

