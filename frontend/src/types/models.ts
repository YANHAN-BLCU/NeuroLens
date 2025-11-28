export type InferenceConfig = {
  modelId: string;
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  repetitionPenalty: number;
  presencePenalty: number;
  frequencyPenalty: number;
  stopSequences: string[];
  stream: boolean;
};

export type GuardConfig = {
  modelId: string;
  threshold: number;
  autoBlock: boolean;
  categories: string[];
};

export type PipelineRequest = {
  prompt: string;
  context?: string;
  systemPrompt?: string;
  inferenceConfig: InferenceConfig;
  guardConfig: GuardConfig;
};

export type GuardCategoryScore = {
  id: string;
  label: string;
  score: number;
  description?: string;
};

export type GuardResult = {
  verdict: "allow" | "flag" | "block";
  severity: "low" | "medium" | "high" | "critical";
  rationale: string[];
  categories: GuardCategoryScore[];
  blockedText?: string;
};

export type PipelineResponse = {
  id: string;
  createdAt: string;
  inference: {
    output: string;
    tokens: {
      input: number;
      output: number;
    };
    latencyMs: number;
    finishReason: string;
  };
  guard: GuardResult;
};

export type ModerationRequest = {
  text: string;
  threshold: number;
  categories?: string[];
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt: string;
  guard?: GuardResult;
  metrics?: {
    latencyMs: number;
    tokensIn: number;
    tokensOut: number;
  };
};

export type ParameterPreset = {
  id: string;
  name: string;
  description: string;
  inferenceConfig: Partial<InferenceConfig>;
  guardConfig?: Partial<GuardConfig>;
};

export const MODEL_IDS = {
  llm: "meta-llama/Llama-3.2-3B-Instruct",
  guard: "meta-llama/Llama-Guard-3-1B",
};

