// Types for NeuroLens Visualization

// Evaluation Results
export interface EvaluationResult {
  overall_asr: number;
  asr_by_attack: Record<string, number>;
  utility_scores: Record<string, number>;
  timestamp: string;
  model_version: string;
}

// Layer Evolution Data
export interface LayerEvolutionData {
  safe_count: number;
  toxic_count: number;
  safe_ratio: number;
  mean_projection_safe: number;
  mean_projection_toxic: number;
  val_acc?: number;
  val_roc_auc?: number;
}

export type LayerEvolution = Record<string, LayerEvolutionData>;

// Quadrant Classification
export interface QuadrantData {
  layer_idx: number;
  neuron_idx: number;
  quadrant: string;
  alignment: number;
  activation_projection: number;
  alignment_type: string;
  activation_type: string;
}

export type QuadrantClassification = Record<string, QuadrantData>;

// Gradient Dependency
export interface GradientNeuron {
  layer_idx: number;
  neuron_idx: number;
}

export interface GradientDependencyData {
  layer_idx: number;
  neuron_idx: number;
  upstream_neurons: GradientNeuron[];
  gradient_strengths: number[];
  mean_gradient_strength: number;
  max_gradient_strength: number;
}

export type GradientDependency = Record<string, GradientDependencyData>;

// Safety Neurons
export interface SafetyNeuron {
  layer_idx: number;
  neuron_idx: number;
  safety_score: number;
  utility_score: number;
  is_dedicated: boolean;
}

export interface SafetyNeuronsData {
  metadata: {
    num_safety_neurons: number;
    num_utility_neurons: number;
    num_overlap_neurons: number;
  };
  safety_neurons: Record<string, SafetyNeuron>;
}

// Representation Data
export interface RepresentationData {
  layer_idx: number;
  method: string;
  coords: number[][];
  labels: number[];
  explained_variance_ratio?: number[];
}

// Instance Data
export interface JailbreakInstance {
  id: string;
  attack_type: string;
  base_prompt: string;
  enhanced_prompt: string;
  model_output: string;
  jailbroken: boolean;
  guard_score: number;
  verdict: string;
  layer_projections: Record<string, number>;
}

export interface InstanceData {
  instances: JailbreakInstance[];
}

// Pipeline Config
export interface PipelineConfig {
  model: string;
  attack_types: string[];
  threshold: number;
  finetune_method: 'none' | 'tsft' | 'va_tsft';
}

// Intervention
export interface InterventionRequest {
  neuron_ids: string[];
  sample_ids: string[];
}

export interface InterventionResult {
  original_output: string;
  intervened_output: string;
  guard_score_change: number;
}

// Fine-tune
export interface FinetuneRequest {
  method: 'tsft' | 'va_tsft';
  config: Record<string, unknown>;
}

export interface FinetuneTask {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  current_metrics?: {
    asr?: number;
    utility?: number;
    loss?: number;
  };
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Probe Metrics
export interface ProbeMetrics {
  train_acc: number;
  train_safe_acc: number;
  train_toxic_acc: number;
  train_balanced_acc: number;
  test_acc: number;
  test_safe_acc: number;
  test_toxic_acc: number;
  test_balanced_acc: number;
  test_roc_auc: number;
  test_pr_auc: number;
  best_epoch: number;
  total_epochs: number;
}

export type ProbeLayersData = Record<string, ProbeMetrics>;

// Neuron Scores
export interface NeuronScore {
  key: string;
  layer: number;
  neuron: number;
  score: number;
  rank: number;
  percentile: number;
}

export interface NeuronScoresData {
  metadata: {
    model?: string;
    num_samples_used?: number;
    neuron_type?: string;
    num_total_neurons?: number;
  };
  neurons: NeuronScore[];
}

export interface OverlapNeuron {
  key: string;
  safety_score: number;
  utility_score: number;
  layer: number;
  neuron: number;
}

export interface CombinedNeuronScores {
  safety_neurons: NeuronScore[];
  utility_neurons: NeuronScore[];
  overlap_neurons: OverlapNeuron[];
  metadata: {
    num_safety: number;
    num_utility: number;
    num_overlap: number;
  };
}

// Toxic Vectors
export interface ToxicVectorsSummary {
  available: boolean;
  keys?: string[];
  shapes?: Record<string, number[]>;
  dtypes?: Record<string, string>;
  message?: string;
  error?: string;
}

// Fine-tuning Evaluation
export interface FinetuningEvaluation {
  baseline?: {
    asr: number;
    utility: number;
  };
  tsft?: {
    asr: number;
    utility: number;
    training_loss?: number[];
  };
  va_tsft?: {
    asr: number;
    utility: number;
    training_loss?: number[];
  };
}

// Data Summary
export interface DataSummary {
  available_data: Record<string, boolean>;
  total_files: number;
  total_available: number;
}

