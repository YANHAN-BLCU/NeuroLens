// API Services for NeuroLens Visualization

import axios from 'axios';
import {
  EvaluationResult,
  LayerEvolution,
  QuadrantClassification,
  GradientDependency,
  SafetyNeuronsData,
  RepresentationData,
  InstanceData,
  InterventionRequest,
  InterventionResult,
  FinetuneRequest,
  FinetuneTask,
  PipelineConfig,
} from '../types';

const API_BASE = '/api';

// Create axios instance with defaults
const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// ============= Metrics API =============

export const metricsApi = {
  /**
   * Get evaluation metrics
   */
  getMetrics: async (params?: {
    model_version?: string;
    attack_type?: string;
    time_range?: string;
  }): Promise<EvaluationResult> => {
    const response = await apiClient.get<EvaluationResult>('/metrics', { params });
    return response.data;
  },

  /**
   * Get ASR by attack type
   */
  getAsrByAttack: async (): Promise<Record<string, number>> => {
    const response = await apiClient.get('/metrics/asr-by-attack');
    return response.data;
  },

  /**
   * Get utility scores
   */
  getUtilityScores: async (): Promise<Record<string, number>> => {
    const response = await apiClient.get('/metrics/utility-scores');
    return response.data;
  },
};

// ============= Representation API =============

export const representationApi = {
  /**
   * Get representation data for a specific layer
   */
  getRepresentation: async (params: {
    layer_idx: number;
    method?: 'pca' | 'tsne';
    sample_ids?: string[];
    n_components?: 2 | 3;
  }): Promise<RepresentationData> => {
    const response = await apiClient.get<RepresentationData>('/representation', { params });
    return response.data;
  },
};

// ============= Layer API =============

export const layerApi = {
  /**
   * Get layer evolution data
   */
  getLayerEvolution: async (): Promise<LayerEvolution> => {
    const response = await apiClient.get<LayerEvolution>('/layers/evolution');
    return response.data;
  },

  /**
   * Get gradient dependencies
   */
  getGradientDependencies: async (params?: {
    layer_idx?: number;
    neuron_ids?: string[];
  }): Promise<GradientDependency> => {
    const response = await apiClient.get<GradientDependency>('/layers/gradients', { params });
    return response.data;
  },

  /**
   * Get all available layers
   */
  getAvailableLayers: async (): Promise<number[]> => {
    const response = await apiClient.get<number[]>('/layers');
    return response.data;
  },
};

// ============= Neuron API =============

export const neuronApi = {
  /**
   * Get quadrant classification data
   */
  getQuadrants: async (params?: {
    layer_idx?: number;
    quadrant?: string;
  }): Promise<QuadrantClassification> => {
    const response = await apiClient.get<QuadrantClassification>('/neurons/quadrants', { params });
    return response.data;
  },

  /**
   * Get gradient dependency for neurons
   */
  getGradientDependency: async (params?: {
    neuron_id?: string;
    depth?: number;
  }): Promise<GradientDependency> => {
    const response = await apiClient.get<GradientDependency>('/neurons/gradient-dependency', { params });
    return response.data;
  },

  /**
   * Get safety neurons data
   */
  getSafetyNeurons: async (): Promise<SafetyNeuronsData> => {
    const response = await apiClient.get<SafetyNeuronsData>('/neurons/safety');
    return response.data;
  },

  /**
   * Get parameter alignment data
   */
  getParameterAlignment: async (): Promise<Record<string, unknown>> => {
    const response = await apiClient.get('/neurons/parameter-alignment');
    return response.data;
  },

  /**
   * Get activation projection data
   */
  getActivationProjection: async (): Promise<Record<string, unknown>> => {
    const response = await apiClient.get('/neurons/activation-projection');
    return response.data;
  },
};

// ============= Instance API =============

export const instanceApi = {
  /**
   * Get jailbreak instances
   */
  getInstances: async (params?: {
    attack_type?: string;
    jailbroken?: boolean;
    limit?: number;
    offset?: number;
  }): Promise<InstanceData> => {
    const response = await apiClient.get<InstanceData>('/instances', { params });
    return response.data;
  },

  /**
   * Get single instance by ID
   */
  getInstanceById: async (id: string): Promise<import('../types').JailbreakInstance> => {
    const response = await apiClient.get<import('../types').JailbreakInstance>(`/instances/${id}`);
    return response.data;
  },
};

// ============= Pipeline API =============

export const pipelineApi = {
  /**
   * Run the inference pipeline
   */
  runPipeline: async (config: PipelineConfig): Promise<{ task_id: string }> => {
    const response = await apiClient.post<{ task_id: string }>('/pipeline/run', config);
    return response.data;
  },

  /**
   * Get pipeline status
   */
  getPipelineStatus: async (taskId: string): Promise<{ status: string; result?: unknown }> => {
    const response = await apiClient.get(`/pipeline/status/${taskId}`);
    return response.data;
  },
};

// ============= Intervention API =============

export const interventionApi = {
  /**
   * Perform neuron intervention
   */
  intervene: async (request: InterventionRequest): Promise<InterventionResult> => {
    const response = await apiClient.post<InterventionResult>('/intervene', request);
    return response.data;
  },
};

// ============= Fine-tune API =============

export const finetuneApi = {
  /**
   * Start fine-tuning
   */
  startFinetune: async (request: FinetuneRequest): Promise<FinetuneTask> => {
    const response = await apiClient.post<FinetuneTask>('/fine_tune', request);
    return response.data;
  },

  /**
   * Get fine-tuning task status
   */
  getFinetuneStatus: async (taskId: string): Promise<FinetuneTask> => {
    const response = await apiClient.get<FinetuneTask>(`/fine_tune/${taskId}`);
    return response.data;
  },

  /**
   * Cancel fine-tuning task
   */
  cancelFinetune: async (taskId: string): Promise<void> => {
    await apiClient.delete(`/fine_tune/${taskId}`);
  },
};

// ============= Health API =============

export const healthApi = {
  /**
   * Check API health
   */
  checkHealth: async (): Promise<{ status: string; version: string }> => {
    const response = await apiClient.get('/health');
    return response.data;
  },
};

// Export default client
export default apiClient;

