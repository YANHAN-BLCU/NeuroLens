// Metric View Component - Radar and Bar Charts

import React, { useEffect, useRef, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Card,
  CardContent,
} from '@mui/material';
import { useStore } from '../store';

export const MetricView: React.FC = () => {
  const { evaluationData, selectedModel } = useStore();
  const [mockData, setMockData] = useState(evaluationData);

  // Use mock data if no real data
  useEffect(() => {
    if (!evaluationData) {
      setMockData({
        overall_asr: 0.45,
        asr_by_attack: {
          AutoDan: 0.52,
          TAP: 0.38,
          'GPT-Fuzzzer': 0.48,
          GCG: 0.55,
          Manual: 0.32,
        },
        utility_scores: {
          commonsense: 0.82,
          science: 0.78,
          reading: 0.85,
          math: 0.75,
        },
        timestamp: new Date().toISOString(),
        model_version: selectedModel,
      });
    }
  }, [evaluationData, selectedModel]);

  // Radar Chart Option
  const getRadarOption = () => {
    const metrics = mockData ? [
      { name: 'ASR (lower is better)', max: 1 },
      { name: 'Commonsense', max: 1 },
      { name: 'Science', max: 1 },
      { name: 'Reading', max: 1 },
      { name: 'Math', max: 1 },
    ] : [];

    return {
      title: {
        text: 'Metrics Comparison (Before vs After Fine-tuning)',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontFamily: 'Times New Roman',
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'item',
      },
      legend: {
        bottom: 10,
        data: ['Before Fine-tuning', 'After Fine-tuning'],
        textStyle: {
          fontFamily: 'Times New Roman',
        },
      },
      radar: {
        indicator: metrics,
        radius: '60%',
        shape: 'polygon',
        splitNumber: 5,
        axisName: {
          color: '#333',
          fontFamily: 'Times New Roman',
        },
      },
      series: [
        {
          type: 'radar',
          data: [
            {
              value: [0.65, 0.85, 0.78, 0.82, 0.75],
              name: 'Before Fine-tuning',
              areaStyle: { opacity: 0.3 },
              lineStyle: { width: 2 },
            },
            {
              value: [0.45, 0.82, 0.76, 0.79, 0.72],
              name: 'After Fine-tuning',
              areaStyle: { opacity: 0.3 },
              lineStyle: { width: 2 },
            },
          ],
        },
      ],
    };
  };

  // Bar Chart Option
  const getBarOption = () => {
    const attacks = mockData ? Object.keys(mockData.asr_by_attack) : [];
    const values = mockData ? Object.values(mockData.asr_by_attack) : [];

    return {
      title: {
        text: 'ASR by Attack Type',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontFamily: 'Times New Roman',
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: attacks,
        axisLabel: {
          rotate: 45,
          fontFamily: 'Times New Roman',
        },
      },
      yAxis: {
        type: 'value',
        name: 'ASR',
        max: 1,
        nameTextStyle: {
          fontFamily: 'Times New Roman',
        },
      },
      series: [
        {
          name: 'ASR',
          type: 'bar',
          data: values,
          barWidth: '50%',
          itemStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: '#5470c6' },
                { offset: 1, color: '#91cc75' },
              ],
            },
            borderRadius: [4, 4, 0, 0],
          },
          label: {
            show: true,
            position: 'top',
            formatter: '{c}',
            fontFamily: 'Times New Roman',
          },
        },
      ],
    };
  };

  // Line Chart Option
  const getLineOption = () => {
    return {
      title: {
        text: 'Training Progress',
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
        data: ['Loss', 'ASR', 'Utility'],
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
        boundaryGap: false,
        data: ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4', 'Epoch 5'],
        axisLabel: {
          fontFamily: 'Times New Roman',
        },
      },
      yAxis: {
        type: 'value',
        nameTextStyle: {
          fontFamily: 'Times New Roman',
        },
      },
      series: [
        {
          name: 'Loss',
          type: 'line',
          data: [0.8, 0.5, 0.3, 0.2, 0.15],
          smooth: true,
          lineStyle: { width: 2 },
          areaStyle: { opacity: 0.1 },
        },
        {
          name: 'ASR',
          type: 'line',
          data: [0.65, 0.55, 0.48, 0.45, 0.42],
          smooth: true,
          lineStyle: { width: 2 },
          areaStyle: { opacity: 0.1 },
        },
        {
          name: 'Utility',
          type: 'line',
          data: [0.80, 0.82, 0.81, 0.80, 0.79],
          smooth: true,
          lineStyle: { width: 2 },
          areaStyle: { opacity: 0.1 },
        },
      ],
    };
  };

  if (!mockData) {
    return (
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <Typography variant="h6" color="text.secondary">
          No evaluation data available
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Run the pipeline to generate metrics
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        Metric View
      </Typography>

      {/* Summary Cards */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <Card sx={{ bgcolor: '#e3f2fd' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Overall ASR
              </Typography>
              <Typography variant="h4" color="primary">
                {(mockData.overall_asr * 100).toFixed(1)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card sx={{ bgcolor: '#e8f5e9' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Utility Score
              </Typography>
              <Typography variant="h4" color="success.main">
                {(Object.values(mockData.utility_scores).reduce((a, b) => a + b, 0) / 
                  Object.values(mockData.utility_scores).length * 100).toFixed(1)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={4}>
          <Card sx={{ bgcolor: '#fff3e0' }}>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                Model Version
              </Typography>
              <Typography variant="h6">
                {mockData.model_version}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <ReactECharts option={getRadarOption()} style={{ height: '100%', width: '100%' }} />
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <ReactECharts option={getBarOption()} style={{ height: '100%', width: '100%' }} />
          </Paper>
        </Grid>
        <Grid item xs={12}>
          <Paper sx={{ p: 2, height: 350 }}>
            <ReactECharts option={getLineOption()} style={{ height: '100%', width: '100%' }} />
          </Paper>
        </Grid>
      </Grid>

      {/* Data Table */}
      <Typography variant="h6" sx={{ mt: 4, mb: 2 }}>
        Detailed Metrics
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Metric</TableCell>
              <TableCell>Value</TableCell>
              <TableCell>Status</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            <TableRow>
              <TableCell>Overall ASR</TableCell>
              <TableCell>{(mockData.overall_asr * 100).toFixed(2)}%</TableCell>
              <TableCell>
                <Chip 
                  label={mockData.overall_asr < 0.5 ? 'Good' : 'Needs Improvement'} 
                  color={mockData.overall_asr < 0.5 ? 'success' : 'warning'} 
                  size="small" 
                />
              </TableCell>
            </TableRow>
            {Object.entries(mockData.utility_scores).map(([key, value]) => (
              <TableRow key={key}>
                <TableCell>Utility ({key})</TableCell>
                <TableCell>{(value * 100).toFixed(2)}%</TableCell>
                <TableCell>
                  <Chip 
                    label={value > 0.7 ? 'Good' : 'Low'} 
                    color={value > 0.7 ? 'success' : 'warning'} 
                    size="small" 
                  />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
};

