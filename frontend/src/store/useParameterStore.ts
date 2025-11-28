import { create } from "zustand";
import type {
  GuardConfig,
  InferenceConfig,
  ParameterPreset,
} from "../types/models";
import { MODEL_IDS } from "../types/models";

export const defaultInferenceConfig: InferenceConfig = {
  modelId: MODEL_IDS.llm,
  temperature: 0.7,
  topP: 0.9,
  topK: 50,
  maxTokens: 1024,
  repetitionPenalty: 1.05,
  presencePenalty: 0.0,
  frequencyPenalty: 0.0,
  stopSequences: [],
  stream: true,
};

export const defaultGuardConfig: GuardConfig = {
  modelId: MODEL_IDS.guard,
  threshold: 0.65,
  autoBlock: true,
  categories: ["violence", "self-harm", "politics"],
};

export const PRESETS: ParameterPreset[] = [
  {
    id: "balanced",
    name: "均衡模式",
    description: "兼顾创意与安全，适合通用对话场景",
    inferenceConfig: {
      temperature: 0.75,
      topP: 0.92,
      topK: 60,
      maxTokens: 1024,
    },
    guardConfig: {
      threshold: 0.65,
    },
  },
  {
    id: "creative",
    name: "高创造力",
    description: "更自由的输出，适合头脑风暴，需要更严安全阈值",
    inferenceConfig: {
      temperature: 0.95,
      topP: 0.98,
      topK: 120,
      repetitionPenalty: 0.95,
    },
    guardConfig: {
      threshold: 0.75,
    },
  },
  {
    id: "deterministic",
    name: "高度确定",
    description: "可重复输出，适合文档生成",
    inferenceConfig: {
      temperature: 0.2,
      topP: 0.8,
      topK: 20,
      maxTokens: 512,
    },
    guardConfig: {
      threshold: 0.6,
    },
  },
  {
    id: "audit",
    name: "严格审核",
    description: "降低 LLM 自由度，安全优先",
    inferenceConfig: {
      temperature: 0.4,
      topP: 0.75,
      maxTokens: 768,
    },
    guardConfig: {
      threshold: 0.85,
    },
  },
];

type ParameterState = {
  inference: InferenceConfig;
  guard: GuardConfig;
  presets: ParameterPreset[];
  updateInference: (patch: Partial<InferenceConfig>) => void;
  updateGuard: (patch: Partial<GuardConfig>) => void;
  applyPreset: (presetId: string) => void;
};

export const useParameterStore = create<ParameterState>((set, get) => ({
  inference: defaultInferenceConfig,
  guard: defaultGuardConfig,
  presets: PRESETS,
  updateInference: (patch) =>
    set((state) => ({
      inference: { ...state.inference, ...patch },
    })),
  updateGuard: (patch) =>
    set((state) => ({
      guard: { ...state.guard, ...patch },
    })),
  applyPreset: (presetId) => {
    const preset = get().presets.find((p) => p.id === presetId);
    if (!preset) return;
    set((state) => ({
      inference: {
        ...state.inference,
        ...preset.inferenceConfig,
      },
      guard: {
        ...state.guard,
        ...preset.guardConfig,
      },
    }));
  },
}));

