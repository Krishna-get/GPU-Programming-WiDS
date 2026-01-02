import time
import numpy as np

def run_cpu_baseline():
    # NPU Simulation Parameters (Week 1 Project)
    # LLaMA-sized layer
    M = 1       # Batch size (Edge Inference)
    K = 4096    # Input Features
    N = 4096    # Output Features

    print(f"Running CPU Baseline for Edge AI NPU Simulation...")
    print(f"Dimensions: Input({M}x{K}) * Weights({K}x{N})")

    # Initialize data
    X = np.random.randn(M, K).astype(np.float32)
    W = np.random.randn(K, N).astype(np.float32)

    # Measure Runtime
    start_time = time.perf_counter()
    
    # The actual NPU Operation
    Y = np.dot(X, W)
    Y = np.maximum(Y, 0) # ReLU
    
    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000

    print(f"CPU Runtime: {duration_ms:.4f} ms")
    return duration_ms

if __name__ == "__main__":
    run_cpu_baseline()
