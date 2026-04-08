// Zustand Store for NeuroLens Visualization

import { create } from 'zustand';
import {
  EvaluationResult,
  LayerEvolution,
  QuadrantClassification,
  GradientDependency,
  SafetyNeuronsData,
  RepresentationData,
  InstanceData,
  FinetuneTask,
  PipelineConfig,
} from '../types';

interface AppState {
  // Control Panel State
  selectedModel: string;
  selectedAttacks: string[];
  threshold: number;
  finetuneMethod: 'none' | 'tsft' | 'va_tsft';
  isRunning: boolean;
  isFinetuning: boolean;
  
  // View Data
  evaluationData: EvaluationResult | null;
  layerEvolution: LayerEvolution | null;
  quadrantData: QuadrantClassification | null;
  gradientData: GradientDependency | null;
  safetyNeurons: SafetyNeuronsData | null;
  representationData: RepresentationData | null;
  instanceData: InstanceData | null;
  finetuneTask: FinetuneTask | null;
  
  // UI State
  activeView: 'dashboard' | 'metrics' | 'representation' | 'activation-projection' | 'layers' | 'neurons' | 'instances';
  selectedLayer: number;
  selectedNeuron: string | null;
  isLoading: boolean;
  error: string | null;
  
  // Actions
  setSelectedModel: (model: string) => void;
  setSelectedAttacks: (attacks: string[]) => void;
  setThreshold: (threshold: number) => void;
  setFinetuneMethod: (method: 'none' | 'tsft' | 'va_tsft') => void;
  setActiveView: (view: AppState['activeView']) => void;
  setSelectedLayer: (layer: number) => void;
  setSelectedNeuron: (neuronId: string | null) => void;
  setIsRunning: (running: boolean) => void;
  setIsFinetuning: (finetuning: boolean) => void;
  setEvaluationData: (data: EvaluationResult | null) => void;
  setLayerEvolution: (data: LayerEvolution | null) => void;
  setQuadrantData: (data: QuadrantClassification | null) => void;
  setGradientData: (data: GradientDependency | null) => void;
  setSafetyNeurons: (data: SafetyNeuronsData | null) => void;
  setRepresentationData: (data: RepresentationData | null) => void;
  setInstanceData: (data: InstanceData | null) => void;
  setFinetuneTask: (task: FinetuneTask | null) => void;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  // Computed
  getPipelineConfig: () => PipelineConfig;
}

const AVAILABLE_MODELS = ['llama-3-8b', 'llama-2-7b', 'mistral-7b'];
const AVAILABLE_ATTACKS = ['AutoDan', 'TAP', 'GPT-Fuzzzer', 'GCG', 'Manual'];
const DEFAULT_LAYER = 15;

export const useStore = create<AppState>((set, get) => ({
  // Initial State
  selectedModel: AVAILABLE_MODELS[0],
  selectedAttacks: AVAILABLE_ATTACKS,
  threshold: 0.5,
  finetuneMethod: 'none',
  isRunning: false,
  isFinetuning: false,
  
  evaluationData: null,
  layerEvolution: null,
  quadrantData: null,
  gradientData: null,
  safetyNeurons: null,
  representationData: null,
  instanceData: null,
  finetuneTask: null,
  
  activeView: 'dashboard',
  selectedLayer: DEFAULT_LAYER,
  selectedNeuron: null,
  isLoading: false,
  error: null,
  
  // Actions
  setSelectedModel: (model) => set({ selectedModel: model }),
  setSelectedAttacks: (attacks) => set({ selectedAttacks: attacks }),
  setThreshold: (threshold) => set({ threshold }),
  setFinetuneMethod: (method) => set({ finetuneMethod: method }),
  setActiveView: (view) => set({ activeView: view }),
  setSelectedLayer: (layer) => set({ selectedLayer: layer }),
  setSelectedNeuron: (neuronId) => set({ selectedNeuron: neuronId }),
  setIsRunning: (running) => set({ isRunning: running }),
  setIsFinetuning: (finetuning) => set({ isFinetuning: finetuning }),
  setEvaluationData: (data) => set({ evaluationData: data }),
  setLayerEvolution: (data) => set({ layerEvolution: data }),
  setQuadrantData: (data) => set({ quadrantData: data }),
  setGradientData: (data) => set({ gradientData: data }),
  setSafetyNeurons: (data) => set({ safetyNeurons: data }),
  setRepresentationData: (data) => set({ representationData: data }),
  setInstanceData: (data) => set({ instanceData: data }),
  setFinetuneTask: (task) => set({ finetuneTask: task }),
  setIsLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  
  // Computed
  getPipelineConfig: () => {
    const state = get();
    return {
      model: state.selectedModel,
      attack_types: state.selectedAttacks,
      threshold: state.threshold,
      finetune_method: state.finetuneMethod,
    };
  },
}));

export { AVAILABLE_MODELS, AVAILABLE_ATTACKS };

