// Control Panel Component

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Slider,
  FormLabel,
  RadioGroup,
  Radio,
  Button,
  Divider,
  Chip,
  Stack,
  Alert,
  CircularProgress,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  AutoFixHigh as TuneIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';

import { useStore, AVAILABLE_MODELS, AVAILABLE_ATTACKS } from '../store';
import { pipelineApi, finetuneApi } from '../services/api';

export const ControlPanel: React.FC = () => {
  const {
    selectedModel,
    selectedAttacks,
    threshold,
    finetuneMethod,
    isRunning,
    isFinetuning,
    setSelectedModel,
    setSelectedAttacks,
    setThreshold,
    setFinetuneMethod,
    setIsRunning,
    setIsFinetuning,
    setError,
    setEvaluationData,
    setFinetuneTask,
    getPipelineConfig,
  } = useStore();

  const handleAttackToggle = (attack: string) => {
    if (selectedAttacks.includes(attack)) {
      setSelectedAttacks(selectedAttacks.filter((a) => a !== attack));
    } else {
      setSelectedAttacks([...selectedAttacks, attack]);
    }
  };

  const handleRunPipeline = async () => {
    setIsRunning(true);
    setError(null);
    try {
      const config = getPipelineConfig();
      const result = await pipelineApi.runPipeline(config);
      console.log('Pipeline started:', result.task_id);
      
      // For demo, set mock evaluation data
      setEvaluationData({
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
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to run pipeline');
    } finally {
      setIsRunning(false);
    }
  };

  const handleStartFinetune = async () => {
    if (finetuneMethod === 'none') return;
    
    setIsFinetuning(true);
    setError(null);
    try {
      const result = await finetuneApi.startFinetune({
        method: finetuneMethod as 'tsft' | 'va_tsft',
        config: {
          model: selectedModel,
          epochs: 3,
          learning_rate: 1e-5,
        },
      });
      setFinetuneTask(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start fine-tuning');
    } finally {
      setIsFinetuning(false);
    }
  };

  return (
    <Paper
      elevation={2}
      sx={{
        p: 3,
        backgroundColor: 'white',
        borderRadius: 2,
      }}
    >
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        Control Panel
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Configure model, attack types, and parameters for analysis
      </Typography>

      <Stack spacing={3}>
        {/* Model Selection */}
        <FormControl fullWidth>
          <InputLabel>Model</InputLabel>
          <Select
            value={selectedModel}
            label="Model"
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {AVAILABLE_MODELS.map((model) => (
              <MenuItem key={model} value={model}>
                {model}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Divider />

        {/* Attack Types */}
        <Box>
          <FormLabel component="legend" sx={{ mb: 1, fontWeight: 500 }}>
            Attack Types
          </FormLabel>
          <FormGroup row>
            {AVAILABLE_ATTACKS.map((attack) => (
              <FormControlLabel
                key={attack}
                control={
                  <Checkbox
                    checked={selectedAttacks.includes(attack)}
                    onChange={() => handleAttackToggle(attack)}
                    color="primary"
                  />
                }
                label={attack}
              />
            ))}
          </FormGroup>
          <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
            <Button
              size="small"
              onClick={() => setSelectedAttacks(AVAILABLE_ATTACKS)}
            >
              Select All
            </Button>
            <Button
              size="small"
              onClick={() => setSelectedAttacks([])}
            >
              Clear
            </Button>
          </Stack>
        </Box>

        <Divider />

        {/* Threshold Slider */}
        <Box>
          <Typography gutterBottom>
            Detection Threshold: <strong>{threshold}</strong>
          </Typography>
          <Slider
            value={threshold}
            onChange={(_, value) => setThreshold(value as number)}
            min={0}
            max={1}
            step={0.05}
            marks={[
              { value: 0, label: '0' },
              { value: 0.5, label: '0.5' },
              { value: 1, label: '1' },
            ]}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => value.toFixed(2)}
          />
        </Box>

        <Divider />

        {/* Fine-tune Method */}
        <FormControl component="fieldset">
          <FormLabel component="legend" sx={{ mb: 1, fontWeight: 500 }}>
            Fine-tune Method
          </FormLabel>
          <RadioGroup
            row
            value={finetuneMethod}
            onChange={(e) => setFinetuneMethod(e.target.value as typeof finetuneMethod)}
          >
            <FormControlLabel value="none" control={<Radio />} label="None" />
            <FormControlLabel value="tsft" control={<Radio />} label="TSFT" />
            <FormControlLabel value="va_tsft" control={<Radio />} label="VA-TSFT" />
          </RadioGroup>
        </FormControl>

        <Divider />

        {/* Action Buttons */}
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            color="primary"
            size="large"
            startIcon={isRunning ? <CircularProgress size={20} color="inherit" /> : <PlayIcon />}
            onClick={handleRunPipeline}
            disabled={isRunning || selectedAttacks.length === 0}
            sx={{ minWidth: 160 }}
          >
            {isRunning ? 'Running...' : 'Run Pipeline'}
          </Button>
          
          <Button
            variant="outlined"
            color="secondary"
            size="large"
            startIcon={isFinetuning ? <CircularProgress size={20} /> : <TuneIcon />}
            onClick={handleStartFinetune}
            disabled={isFinetuning || finetuneMethod === 'none'}
            sx={{ minWidth: 160 }}
          >
            {isFinetuning ? 'Fine-tuning...' : 'Start Fine-tune'}
          </Button>
        </Box>

        {/* Selected Config Summary */}
        <Alert severity="info" sx={{ mt: 2 }}>
          <Typography variant="body2">
            <strong>Configuration:</strong> {selectedModel} | 
            Attacks: {selectedAttacks.length > 0 ? selectedAttacks.join(', ') : 'None selected'} |
            Threshold: {threshold} | 
            Fine-tune: {finetuneMethod.toUpperCase()}
          </Typography>
        </Alert>
      </Stack>
    </Paper>
  );
};

