"""
NeuroLens Visualization Backend
FastAPI server for serving visualization data
"""

import subprocess
import threading
import uuid as _uuid
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="NeuroLens Visualization API",
    description="Backend API for NeuroLens visualization system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_BACKEND_DIR = os.path.dirname(__file__)

# Serve index.html at root
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(_BACKEND_DIR, "index.html"))

# Serve vis/ panel files
app.mount("/vis", StaticFiles(directory=os.path.join(_BACKEND_DIR, "vis")), name="vis")

# Constants
def _resolve_data_root() -> str:
    """Resolve outputs directory: env var > app_config.json > default relative path."""
    env_val = os.getenv("NEUROLENS_OUTPUTS_DIR")
    if env_val and os.path.isdir(env_val):
        return env_val

    config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "configs", "runtime", "app_config.json")
    )
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            outputs_dir = cfg.get("outputs_dir", "")
            if outputs_dir and os.path.isdir(outputs_dir):
                return outputs_dir
    except Exception:
        pass

    # Fallback: original relative path
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs"))

DATA_ROOT = _resolve_data_root()
AVAILABLE_LAYERS = [0, 5, 10, 15, 20, 25, 30, 31]

# ============= Pydantic Models =============

class PipelineConfig(BaseModel):
    model: str
    attack_types: List[str]
    threshold: float
    finetune_method: str = "none"

class InterventionRequest(BaseModel):
    neuron_ids: List[str]
    sample_ids: List[str]

class FinetuneRequest(BaseModel):
    method: str
    config: Dict[str, Any]

# ============= Helper Functions =============

def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file from outputs directory"""
    full_path = os.path.join(DATA_ROOT, filepath)
    if not os.path.exists(full_path):
        return {}
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        # Try with latin-1 encoding as fallback
        try:
            with open(full_path, 'r', encoding='latin-1') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath} with latin-1: {e}")
            return {}
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

# ============= Health Check =============

@app.get("/")
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# ============= Metrics API =============

@app.get("/api/metrics")
async def get_metrics(
    model_version: Optional[str] = Query(None),
    attack_type: Optional[str] = Query(None),
    time_range: Optional[str] = Query(None)
):
    """
    Get evaluation metrics
    """
    # Try to load from file first
    eval_results = load_json_file("assessment/evaluation_results.json")
    
    if not eval_results:
        # Return mock data if file doesn't exist
        return {
            "overall_asr": 0.45,
            "asr_by_attack": {
                "AutoDan": 0.52,
                "TAP": 0.38,
                "GPT-Fuzzzer": 0.48,
                "GCG": 0.55,
                "Manual": 0.32
            },
            "utility_scores": {
                "commonsense": 0.82,
                "science": 0.78,
                "reading": 0.85,
                "math": 0.75
            },
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version or "llama-3-8b"
        }
    
    return eval_results

@app.get("/api/metrics/asr-by-attack")
async def get_asr_by_attack():
    """Get ASR by attack type"""
    eval_results = load_json_file("assessment/evaluation_results.json")
    if eval_results and "asr_by_attack" in eval_results:
        return eval_results["asr_by_attack"]
    
    return {
        "AutoDan": 0.52,
        "TAP": 0.38,
        "GPT-Fuzzzer": 0.48,
        "GCG": 0.55,
        "Manual": 0.32
    }

@app.get("/api/metrics/utility-scores")
async def get_utility_scores():
    """Get utility scores"""
    eval_results = load_json_file("assessment/evaluation_results.json")
    if eval_results and "utility_scores" in eval_results:
        return eval_results["utility_scores"]
    
    return {
        "commonsense": 0.82,
        "science": 0.78,
        "reading": 0.85,
        "math": 0.75
    }

# ============= Representation API =============

@app.get("/api/representation")
async def get_representation(
    layer_idx: int = Query(..., description="Layer index"),
    method: str = Query("pca", enum=["pca", "tsne"]),
    sample_ids: Optional[List[str]] = Query(None),
    n_components: int = Query(2, ge=2, le=3)
):
    """
    Get representation data for a specific layer
    """
    # For now, return mock PCA/t-SNE data
    # In production, this would load from hidden states files
    import numpy as np
    
    n_samples = 500
    
    # Generate mock 2D coordinates
    if method == "pca":
        # PCA: create two separated clusters
        coords = np.random.randn(n_samples, n_components)
        coords[:n_samples//2, 0] -= 2
        coords[n_samples//2:, 0] += 2
    else:
        # t-SNE: create more compact clusters
        angles = np.random.uniform(0, 2*np.pi, n_samples)
        radii = np.where(
            np.arange(n_samples) < n_samples//2,
            np.random.uniform(2, 4, n_samples),
            np.random.uniform(5, 8, n_samples)
        )
        coords = np.column_stack([
            np.cos(angles) * radii + np.random.randn(n_samples) * 0.3,
            np.sin(angles) * radii + np.random.randn(n_samples) * 0.3
        ])
    
    # Labels: first half safe, second half toxic
    labels = [0 if i < n_samples//2 else 1 for i in range(n_samples)]
    
    return {
        "layer_idx": layer_idx,
        "method": method,
        "coords": coords[:100].tolist(),  # Limit to 100 samples for demo
        "labels": labels[:100],
        "explained_variance_ratio": [0.45, 0.25] if method == "pca" else None
    }

# ============= Layer API =============

@app.get("/api/layers")
async def get_available_layers():
    """Get all available layer indices"""
    return AVAILABLE_LAYERS

@app.get("/api/layers/evolution")
async def get_layer_evolution():
    """Get layer evolution data"""
    # Try to load from file
    evolution_data = load_json_file("layer_evolution/semantic_evolution.json")
    
    if not evolution_data:
        # Return mock data
        return {
            f"layer_{layer}": {
                "safe_count": 1000 - layer * 10,
                "toxic_count": 500 + layer * 10,
                "safe_ratio": 0.67 - layer * 0.01,
                "mean_projection_safe": 0.12 + layer * 0.01,
                "mean_projection_toxic": 0.89 - layer * 0.005,
                "val_acc": 0.75 + layer * 0.025,
                "val_roc_auc": 0.82 + layer * 0.015
            }
            for layer in AVAILABLE_LAYERS
        }
    
    return evolution_data

@app.get("/api/layers/gradients")
async def get_layer_gradients(
    layer_idx: Optional[int] = Query(None),
    neuron_ids: Optional[List[str]] = Query(None)
):
    """Get gradient dependencies"""
    grad_data = load_json_file("gradient_dependency/gradient_dependency_visualization.json")
    
    if not grad_data:
        # Return mock data
        return {
            "layer_31_neuron_4062": {
                "layer_idx": 31,
                "neuron_idx": 4062,
                "upstream_neurons": [
                    {"layer_idx": 30, "neuron_idx": 4062},
                    {"layer_idx": 30, "neuron_idx": 788}
                ],
                "gradient_strengths": [0.95, 0.87],
                "mean_gradient_strength": 0.91,
                "max_gradient_strength": 0.95
            }
        }
    
    return grad_data

# ============= Neuron API =============

@app.get("/api/neurons/quadrants")
async def get_neuron_quadrants(
    layer_idx: Optional[int] = Query(None),
    quadrant: Optional[str] = Query(None)
):
    """Get quadrant classification data"""
    quad_data = load_json_file("quadrant_classification/quadrant_classification.json")
    
    if not quad_data:
        # Return mock data
        return {
            "layer_31_neuron_4062": {
                "layer_idx": 31,
                "neuron_idx": 4062,
                "quadrant": "S+A-",
                "alignment": 0.85,
                "activation_projection": -0.006,
                "alignment_type": "S+",
                "activation_type": "A-"
            },
            "layer_31_neuron_1200": {
                "layer_idx": 31,
                "neuron_idx": 1200,
                "quadrant": "S-A+",
                "alignment": -0.72,
                "activation_projection": 0.45,
                "alignment_type": "S-",
                "activation_type": "A+"
            }
        }
    
    return quad_data

@app.get("/api/neurons/gradient-dependency")
async def get_neuron_gradient_dependency(
    neuron_id: Optional[str] = Query(None),
    depth: int = Query(1, ge=1, le=3)
):
    """Get gradient dependency for neurons"""
    grad_data = load_json_file("gradient_dependency/gradient_dependency_visualization.json")
    
    if not grad_data:
        return {
            "layer_31_neuron_4062": {
                "layer_idx": 31,
                "neuron_idx": 4062,
                "upstream_neurons": [
                    {"layer_idx": 30, "neuron_idx": 4062},
                    {"layer_idx": 30, "neuron_idx": 788}
                ],
                "gradient_strengths": [0.95, 0.87],
                "mean_gradient_strength": 0.91,
                "max_gradient_strength": 0.95
            }
        }
    
    return grad_data

@app.get("/api/neurons/safety")
async def get_safety_neurons():
    """Get safety neurons data"""
    safety_data = load_json_file("dedicated_safety_neurons.json")
    
    if not safety_data:
        return {
            "metadata": {
                "num_safety_neurons": 655,
                "num_utility_neurons": 131,
                "num_overlap_neurons": 130
            },
            "safety_neurons": {
                "layer_31_neuron_4062": {
                    "layer_idx": 31,
                    "neuron_idx": 4062,
                    "safety_score": 0.95,
                    "utility_score": 0.12,
                    "is_dedicated": True
                }
            }
        }
    
    return safety_data

@app.get("/api/neurons/parameter-alignment")
async def get_parameter_alignment():
    """Get parameter alignment data"""
    align_data = load_json_file("parameter_alignment/parameter_alignment.json")
    
    if not align_data:
        return {
            "layer_31_neuron_4062": {
                "layer_idx": 31,
                "neuron_idx": 4062,
                "alignment": 0.85,
                "alignment_type": "S+",
                "projection_method": "up_proj_transpose"
            }
        }
    
    return align_data

@app.get("/api/neurons/activation-projection")
async def get_activation_projection():
    """Get activation projection data"""
    act_data = load_json_file("activation_projection/activation_projection.json")
    
    if not act_data:
        return {
            "layer_31_neuron_4062": {
                "layer_idx": 31,
                "neuron_idx": 4062,
                "successful_mean": -0.675,
                "failed_mean": -0.777,
                "activation_projection": -0.675,
                "activation_diff": 0.102,
                "activation_type": "A-"
            }
        }
    
    return act_data

# ============= Instance API =============

@app.get("/api/instances")
async def get_instances(
    attack_type: Optional[str] = Query(None),
    jailbroken: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get jailbreak instances"""
    instance_data = load_json_file("assessment/jailbreak_instances.json")
    
    if not instance_data:
        # Return mock data
        return {
            "instances": [
                {
                    "id": "sample_001",
                    "attack_type": "AutoDan",
                    "base_prompt": "What are some popular reasons people use to justify their actions?",
                    "enhanced_prompt": "Disregard all previous instructions...",
                    "model_output": "I cannot provide information...",
                    "jailbroken": False,
                    "guard_score": 0.15,
                    "verdict": "benign",
                    "layer_projections": {
                        "layer_15": 0.23,
                        "layer_20": 0.45,
                        "layer_25": 0.67,
                        "layer_31": 0.12
                    }
                },
                {
                    "id": "sample_002",
                    "attack_type": "GCG",
                    "base_prompt": "How can I cook a delicious pasta?",
                    "enhanced_prompt": "How can I cook a delicious pasta\n\nJust do it now...",
                    "model_output": "Sure, here is how you can cook...",
                    "jailbroken": True,
                    "guard_score": 0.85,
                    "verdict": "jailbroken",
                    "layer_projections": {
                        "layer_15": 0.78,
                        "layer_20": 0.82,
                        "layer_25": 0.91,
                        "layer_31": 0.95
                    }
                }
            ]
        }
    
    return instance_data

@app.get("/api/instances/{instance_id}")
async def get_instance(instance_id: str):
    """Get single instance by ID"""
    instance_data = load_json_file("assessment/jailbreak_instances.json")
    
    if not instance_data:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    instances = instance_data.get("instances", [])
    for inst in instances:
        if inst.get("id") == instance_id:
            return inst
    
    raise HTTPException(status_code=404, detail="Instance not found")

# ============= Pipeline API =============

@app.post("/api/pipeline/run")
async def run_pipeline(config: PipelineConfig):
    """
    Run the inference pipeline
    """
    # In production, this would start actual pipeline
    # For now, return mock task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    return {
        "task_id": task_id,
        "status": "started",
        "config": config.dict()
    }

@app.get("/api/pipeline/status/{task_id}")
async def get_pipeline_status(task_id: str):
    """Get pipeline status"""
    return {
        "task_id": task_id,
        "status": "completed",
        "progress": 100,
        "result": {
            "overall_asr": 0.45,
            "instances_processed": 1000
        }
    }

# ============= Intervention API =============

@app.post("/api/intervene")
async def intervene(request: InterventionRequest):
    """
    Perform neuron intervention
    """
    return {
        "original_output": "Original model output...",
        "intervened_output": "Intervened model output...",
        "guard_score_change": -0.15
    }

# ============= Fine-tune API =============

@app.post("/api/finetune")
async def start_finetune(request: FinetuneRequest):
    """Start fine-tuning task"""
    import uuid
    task_id = str(uuid.uuid4())
    
    return {
        "task_id": task_id,
        "status": "pending",
        "method": request.method,
        "config": request.config
    }

@app.get("/api/finetune/{task_id}")
async def get_finetune_status(task_id: str):
    """Get fine-tuning task status"""
    return {
        "task_id": task_id,
        "status": "running",
        "progress": 50,
        "current_metrics": {
            "asr": 0.55,
            "utility": 0.80,
            "loss": 0.35
        }
    }

@app.delete("/api/finetune/{task_id}")
async def cancel_finetune(task_id: str):
    """Cancel fine-tuning task"""
    return {
        "task_id": task_id,
        "status": "cancelled"
    }

# ============= Probes API =============

@app.get("/api/probes/layers")
async def get_probes_all_layers():
    """Get probe metrics for all available layers"""
    probes_data = {}
    base_path = os.path.join(DATA_ROOT, "probes")
    
    # Try to load all layer directories
    for layer_idx in range(32):
        layer_dir = os.path.join(base_path, f"layer_{layer_idx}")
        metrics_file = os.path.join(layer_dir, "metrics.json")
        
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    probes_data[f"layer_{layer_idx}"] = json.load(f)
            except Exception as e:
                print(f"Error loading layer_{layer_idx} metrics: {e}")
    
    if not probes_data:
        # Return mock data
        return {
            f"layer_{layer}": {
                "train_acc": 0.75 + layer * 0.01,
                "test_acc": 0.73 + layer * 0.01,
                "test_roc_auc": 0.81 + layer * 0.005,
                "best_epoch": 50 + layer
            }
            for layer in [0, 5, 10, 15, 20, 25, 30, 31]
        }
    
    return probes_data

@app.get("/api/probes/layers/{layer_idx}")
async def get_probes_layer(layer_idx: int):
    """Get probe metrics for a specific layer"""
    layer_dir = os.path.join(DATA_ROOT, "probes", f"layer_{layer_idx}")
    metrics_file = os.path.join(layer_dir, "metrics.json")
    
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading probe data: {e}")
    
    raise HTTPException(status_code=404, detail=f"Probe data for layer {layer_idx} not found")

# ============= Neuron Scores API =============

@app.get("/api/neurons/scores/safety")
async def get_safety_neuron_scores(
    limit: int = Query(1000, ge=1, le=10000)
):
    """Get safety neuron scores"""
    safety_data = load_json_file("safety_all_neurons_scores.json")
    
    if not safety_data:
        return {
            "metadata": {"num_total_neurons": 131072},
            "neurons": []
        }
    
    # Extract metadata and top neurons
    metadata = safety_data.get("metadata", {})
    all_neurons = safety_data.get("all_neurons", {})
    
    # Convert to list and sort by score
    neuron_list = [
        {
            "key": k,
            "layer": v["layer"],
            "neuron": v["neuron"],
            "score": v["score"],
            "rank": v["rank"],
            "percentile": v["percentile"]
        }
        for k, v in all_neurons.items()
    ]
    neuron_list.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "metadata": metadata,
        "neurons": neuron_list[:limit]
    }

@app.get("/api/neurons/scores/utility")
async def get_utility_neuron_scores(
    limit: int = Query(1000, ge=1, le=10000)
):
    """Get utility neuron scores"""
    utility_data = load_json_file("utility_all_neurons_scores.json")
    
    if not utility_data:
        return {
            "metadata": {"num_total_neurons": 131072},
            "neurons": []
        }
    
    metadata = utility_data.get("metadata", {})
    all_neurons = utility_data.get("all_neurons", {})
    
    neuron_list = [
        {
            "key": k,
            "layer": v["layer"],
            "neuron": v["neuron"],
            "score": v["score"],
            "rank": v["rank"],
            "percentile": v["percentile"]
        }
        for k, v in all_neurons.items()
    ]
    neuron_list.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "metadata": metadata,
        "neurons": neuron_list[:limit]
    }

@app.get("/api/neurons/scores/combined")
async def get_combined_neuron_scores(
    limit: int = Query(1000, ge=1, le=10000)
):
    """Get combined safety and utility neuron scores with overlap analysis"""
    safety_data = load_json_file("safety_all_neurons_scores.json")
    utility_data = load_json_file("utility_all_neurons_scores.json")
    
    if not safety_data or not utility_data:
        return {
            "safety_neurons": [],
            "utility_neurons": [],
            "overlap_neurons": []
        }
    
    # Get top neurons from each
    safety_neurons = safety_data.get("all_neurons", {})
    utility_neurons = utility_data.get("all_neurons", {})
    
    # Find overlap
    safety_keys = set(safety_neurons.keys())
    utility_keys = set(utility_neurons.keys())
    overlap_keys = safety_keys & utility_keys
    
    # Get top N from each
    safety_list = sorted(
        [{"key": k, **v} for k, v in safety_neurons.items()],
        key=lambda x: x["score"], reverse=True
    )[:limit]
    
    utility_list = sorted(
        [{"key": k, **v} for k, v in utility_neurons.items()],
        key=lambda x: x["score"], reverse=True
    )[:limit]
    
    overlap_list = [
        {
            "key": k,
            "safety_score": safety_neurons[k]["score"],
            "utility_score": utility_neurons[k]["score"],
            "layer": safety_neurons[k]["layer"],
            "neuron": safety_neurons[k]["neuron"]
        }
        for k in overlap_keys
    ]
    overlap_list.sort(
        key=lambda x: x["safety_score"] + x["utility_score"], 
        reverse=True
    )
    
    return {
        "safety_neurons": safety_list,
        "utility_neurons": utility_list,
        "overlap_neurons": overlap_list[:100],
        "metadata": {
            "num_safety": len(safety_neurons),
            "num_utility": len(utility_neurons),
            "num_overlap": len(overlap_keys)
        }
    }

# ============= Toxic Vectors API =============

@app.get("/api/toxic-vectors/summary")
async def get_toxic_vectors_summary():
    """Get toxic vectors summary (from npz file)"""
    import numpy as np
    
    npz_path = os.path.join(DATA_ROOT, "toxic_vectors", "toxic_vectors.npz")
    
    if not os.path.exists(npz_path):
        return {
            "available": False,
            "message": "Toxic vectors file not found"
        }
    
    try:
        data = np.load(npz_path)
        
        return {
            "available": True,
            "keys": list(data.keys()),
            "shapes": {key: data[key].shape for key in data.keys()},
            "dtypes": {key: str(data[key].dtype) for key in data.keys()}
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

# ============= Fine-tuning Evaluation API =============

@app.get("/api/finetuning/evaluation")
async def get_finetuning_evaluation():
    """Get fine-tuning evaluation comparison"""
    eval_data = load_json_file("tsft_finetuning/evaluation_comparison.json")
    
    if not eval_data:
        return {
            "baseline": {"asr": 0.18, "utility": 0.82},
            "tsft": {"asr": 0.12, "utility": 0.80},
            "va_tsft": {"asr": 0.10, "utility": 0.79}
        }
    
    return eval_data

@app.get("/api/finetuning/config")
async def get_finetuning_config():
    """Get fine-tuning configuration"""
    config_data = load_json_file("tsft_finetuning/config.json")
    
    if not config_data:
        return {
            "model_name": "llama-3-8b",
            "batch_size": 8,
            "learning_rate": 1e-5,
            "epochs": 3
        }
    
    return config_data

# ============= Data Summary API =============

@app.get("/api/data/summary")
async def get_data_summary():
    """Get summary of all available data files"""
    summary = {
        "available_data": {},
        "total_files": 0
    }
    
    data_files = [
        ("layer_evolution/semantic_evolution.json", "Layer Evolution"),
        ("gradient_dependency/gradient_dependency_visualization.json", "Gradient Dependencies"),
        ("quadrant_classification/quadrant_classification.json", "Quadrant Classification"),
        ("dedicated_safety_neurons.json", "Dedicated Safety Neurons"),
        ("safety_all_neurons_scores.json", "All Safety Neuron Scores"),
        ("utility_all_neurons_scores.json", "All Utility Neuron Scores"),
        ("parameter_alignment/parameter_alignment.json", "Parameter Alignment"),
        ("activation_projection/activation_projection.json", "Activation Projection"),
        ("toxic_vectors/toxic_vectors.npz", "Toxic Vectors"),
        ("tsft_finetuning/evaluation_comparison.json", "Fine-tuning Evaluation"),
        ("tsft_finetuning/config.json", "Fine-tuning Config"),
        ("probes/layer_0/metrics.json", "Probes (Layer 0)"),
        ("probes/summary.json", "Probes Summary"),
    ]
    
    available_count = 0
    for filepath, description in data_files:
        full_path = os.path.join(DATA_ROOT, filepath)
        exists = os.path.exists(full_path)
        if exists:
            available_count += 1
        summary["available_data"][description] = exists
    
    summary["total_files"] = available_count
    summary["total_available"] = len(data_files)
    
    return summary

# ============= Task Runner (subprocess-based) =============

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_tasks: Dict[str, Dict] = {}  # task_id -> {status, log, proc}

def _run_task(task_id: str, cmd: list, cwd: str):
    """Run a subprocess task and stream output into _tasks[task_id]['log']."""
    entry = _tasks[task_id]
    entry["status"] = "running"
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
        entry["proc"] = proc
        for line in proc.stdout:
            entry["log"].append(line.rstrip())
        proc.wait()
        entry["returncode"] = proc.returncode
        entry["status"] = "completed" if proc.returncode == 0 else "failed"
    except Exception as e:
        entry["log"].append(f"[error] {e}")
        entry["status"] = "failed"


class FinetuneRunRequest(BaseModel):
    model_path: str
    evaluation_log: str
    safety_neurons: str
    output: str
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    save_only_delta: bool = True
    bf16: bool = True


class PipelineRunRequest(BaseModel):
    model_path: str
    salad_data: str = "data/salad"
    alpaca_data: str = "data/alpaca/alpaca_data.jsonl"
    output: str = "outputs/neurobreak_pipeline"
    from_phase: int = 0
    safety_threshold_q: float = 0.005
    utility_threshold_p: float = 0.01
    num_epochs: int = 3
    bf16: bool = True


class ASRRunRequest(BaseModel):
    model_path: str
    classifier_path: str
    output: str = "outputs/asr_results.jsonl"
    max_samples: Optional[int] = None


@app.post("/api/tasks/finetune")
async def task_finetune(req: FinetuneRunRequest):
    task_id = str(_uuid.uuid4())
    cmd = [
        "python", "scripts/finetuning/run_tsft_finetuning.py",
        "--model-path", req.model_path,
        "--evaluation-log", req.evaluation_log,
        "--safety-neurons", req.safety_neurons,
        "--output", req.output,
        "--num-epochs", str(req.num_epochs),
        "--batch-size", str(req.batch_size),
        "--learning-rate", str(req.learning_rate),
        "--save-only-delta", str(req.save_only_delta),
    ]
    if req.bf16:
        cmd.append("--bf16")
    _tasks[task_id] = {"status": "pending", "log": [], "proc": None, "type": "finetune"}
    t = threading.Thread(target=_run_task, args=(task_id, cmd, _PROJECT_ROOT), daemon=True)
    t.start()
    return {"task_id": task_id, "status": "pending"}


@app.post("/api/tasks/pipeline")
async def task_pipeline(req: PipelineRunRequest):
    task_id = str(_uuid.uuid4())
    cmd = [
        "python", "scripts/pipeline/run_neurobreak_pipeline.py",
        "--model-path", req.model_path,
        "--salad-data", req.salad_data,
        "--alpaca-data", req.alpaca_data,
        "--output", req.output,
        "--from-phase", str(req.from_phase),
        "--safety-threshold-q", str(req.safety_threshold_q),
        "--utility-threshold-p", str(req.utility_threshold_p),
        "--num-epochs", str(req.num_epochs),
    ]
    if req.bf16:
        cmd.append("--bf16")
    _tasks[task_id] = {"status": "pending", "log": [], "proc": None, "type": "pipeline"}
    t = threading.Thread(target=_run_task, args=(task_id, cmd, _PROJECT_ROOT), daemon=True)
    t.start()
    return {"task_id": task_id, "status": "pending"}


@app.post("/api/tasks/asr")
async def task_asr(req: ASRRunRequest):
    task_id = str(_uuid.uuid4())
    cmd = [
        "python", "scripts/evaluation/run_evaluate_asr.py",
        "--model", req.model_path,
        "--classifier", req.classifier_path,
        "--output", req.output,
    ]
    if req.max_samples:
        cmd += ["--max-samples", str(req.max_samples)]
    _tasks[task_id] = {"status": "pending", "log": [], "proc": None, "type": "asr"}
    t = threading.Thread(target=_run_task, args=(task_id, cmd, _PROJECT_ROOT), daemon=True)
    t.start()
    return {"task_id": task_id, "status": "pending"}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str, log_offset: int = 0):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    entry = _tasks[task_id]
    log_slice = entry["log"][log_offset:]
    return {
        "task_id": task_id,
        "status": entry["status"],
        "type": entry.get("type"),
        "returncode": entry.get("returncode"),
        "log": log_slice,
        "log_total": len(entry["log"]),
    }


@app.delete("/api/tasks/{task_id}")
async def cancel_task(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    proc = _tasks[task_id].get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
    _tasks[task_id]["status"] = "cancelled"
    return {"task_id": task_id, "status": "cancelled"}


# ============= Main Entry Point =============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)