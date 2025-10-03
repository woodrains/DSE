#!/usr/bin/env python3
"""
Complete FlexAction vs CGA Comparison on Real Hardware
Runs on NVIDIA A100 GPU0 using conda environment 'llmulator'
Tests GEMM workload with actual performance measurements
"""

import sys
import os

# Add paths
sys.path.append('/root/Heron')
sys.path.append('/root/Heron104')

import json
import time
import numpy as np
from typing import Dict, List
import argparse

# Heron imports
from Heron.environment import Env
from Heron.config import Config
from Heron.runner.runner import MeasureOption
from Heron.task.task import Task

# TVM imports
import tvm
from tvm import te

# FlexAction imports
from flex_tuner import FlexActionTuner


def setup_gpu_environment():
    """Configure GPU and conda environment"""
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU0

    # Verify GPU availability
    import torch
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✗ Warning: No GPU detected!")

    return torch.cuda.is_available()


def create_experiment_config(method: str, log_dir: str, max_trials: int = 100) -> Config:
    """Create configuration for either FlexAction or CGA"""
    config = Config()

    # Basic settings
    config.task_name = f"gemm_tensorcore_{method}"
    config.log_dir = log_dir
    config.codegen_type = 'cuda'
    config.target_name = 'cuda'

    # Set optimization method
    config.opt_method = method  # 'FLEXACTION' or 'CGA'

    # Search parameters
    config.max_trials = max_trials
    config.measure_time_per_round = 20
    config.search_generations = max_trials // 20

    # Population parameters
    config.pop_num = 50
    config.select_num = 20
    config.iter_walks = 5
    config.history_topk = 5
    config.crossover_key_ratio = 0.3

    # Parallel execution
    config.parallel = True
    config.parallel_num = 4

    # Cost model settings
    config.use_cost_model = True
    config.n_estimators = 300
    config.early_stopping = 30
    config.metric = 'r2'

    # Feasible file path
    config.feasible_file_path = os.path.join(log_dir, 'feasible.json')

    return config


def create_gemm_workload(M: int, N: int, K: int):
    """Create GEMM computation task"""
    def gemm_compute():
        A = te.placeholder((M, K), name='A', dtype='float16')
        B = te.placeholder((K, N), name='B', dtype='float16')

        # Reduction axis
        k = te.reduce_axis((0, K), name='k')

        # Matrix multiplication
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(
                A[i, k].astype('float32') * B[k, j].astype('float32'),
                axis=k
            ),
            name='C'
        )

        return [A, B, C]

    return gemm_compute, (M, K, N)


def register_flexaction_tuner():
    """Register FlexAction tuner in Heron environment"""
    # Patch the environment.py to support FLEXACTION
    from Heron import environment

    original_createTask = environment.Env.createTask

    def createTask_with_flexaction(self, name, opfunc, args, target,
                                   target_host=None, dump_const_desc=False):
        """Extended createTask that supports FlexAction"""
        assert self.task == None
        self.config.task_name = name
        task = Task(name, opfunc, args, target, target_host)
        task.knob_manager.dump_descs = dump_const_desc

        if self.build_kwargs == None:
            self.get_build_kwargs(task)

        task.build_kwargs = self.build_kwargs
        task.config = self.config
        self.task = task

        # Initialize tuner based on method
        self.config.setEnv(self)

        if self.config.opt_method == 'FLEXACTION':
            # Use FlexAction tuner
            from flex_tuner import FlexActionTuner
            self.tuner = FlexActionTuner(self.config)
        elif self.config.opt_method == 'CGA':
            from Heron.tuner.ga_tuner import CGATuner
            self.tuner = CGATuner(self.config)
        elif self.config.opt_method == 'GA':
            from Heron.tuner.ga_tuner import GATuner
            self.tuner = GATuner(self.config)
        else:
            # Use original implementation
            return original_createTask(self, name, opfunc, args, target,
                                      target_host, dump_const_desc)

        # Build cost model if enabled
        if self.config.use_cost_model:
            self.tuner.buildCostModel(self.task)

        # Initialize performance buffer
        from Heron.perf.perfBuffer import perfBuffer
        self.perf_buffer = perfBuffer(self.config)

        return task

    # Apply patch
    environment.Env.createTask = createTask_with_flexaction
    print("✓ FlexAction tuner registered")


def run_single_experiment(method: str, workload_size: tuple,
                          max_trials: int = 100) -> Dict:
    """Run a single optimization experiment"""
    M, N, K = workload_size

    # Create log directory
    log_dir = f'/root/Heron104/logs/{method}_M{M}_N{N}_K{K}'
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Running {method} on GEMM {M}x{N}x{K}")
    print(f"Log directory: {log_dir}")
    print(f"{'='*70}\n")

    # Create config
    config = create_experiment_config(method, log_dir, max_trials)

    # Set TVM target for A100 (sm_80)
    target = tvm.target.cuda(arch='sm_80')
    target_host = tvm.target.Target("llvm")

    # Create measure option
    measure_option = MeasureOption(
        builder='local',
        runner='local',
        number=10,  # Run 10 times per measurement
        repeat=3,   # Repeat 3 times
        timeout=10  # 10 second timeout
    )

    # Create environment
    env = Env(measure_option, config)

    # Create task
    opfunc, args = create_gemm_workload(M, N, K)
    task_name = f"gemm_M{M}_N{N}_K{K}"

    env.createTask(task_name, opfunc, args, target, target_host)

    # Run tuning
    print(f"Starting {method} optimization...")
    start_time = time.time()

    try:
        population, stat = env.tune(task_name, pretrained=False)
        elapsed_time = time.time() - start_time

        # Collect results
        best_perf = env.perf_buffer.best_perf if env.perf_buffer.best_perf else 0

        # Calculate theoretical peak
        # A100: 312 TFLOPS for FP16 with TensorCore
        theoretical_peak_tflops = 312
        theoretical_peak_gflops = theoretical_peak_tflops * 1000

        # Calculate actual FLOPS for GEMM: 2*M*N*K
        actual_flops = 2 * M * N * K

        # If best_perf is latency (seconds), convert to GFLOPS
        if best_perf > 0 and best_perf < 1:  # Likely latency in seconds
            achieved_gflops = (actual_flops / best_perf) / 1e9
        else:
            achieved_gflops = best_perf

        efficiency = (achieved_gflops / theoretical_peak_gflops) * 100

        results = {
            'method': method,
            'workload': f'{M}x{N}x{K}',
            'M': M, 'N': N, 'K': K,
            'max_trials': max_trials,
            'elapsed_time': elapsed_time,
            'best_performance_gflops': achieved_gflops,
            'efficiency_percent': efficiency,
            'theoretical_peak_gflops': theoretical_peak_gflops,
            'num_measurements': len(env.perf_buffer.data_y) if env.perf_buffer else 0,
            'log_dir': log_dir
        }

        # Save results
        result_file = os.path.join(log_dir, 'results.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✓ {method} Completed")
        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  Best Performance: {achieved_gflops:.2f} GFLOPS")
        print(f"  Efficiency: {efficiency:.2f}% of peak")
        print(f"  Measurements: {results['num_measurements']}")
        print(f"{'='*70}\n")

        return results

    except Exception as e:
        print(f"\n✗ Error during {method} optimization: {e}")
        import traceback
        traceback.print_exc()

        return {
            'method': method,
            'workload': f'{M}x{N}x{K}',
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }


def compare_methods(workload_sizes: List[tuple], max_trials: int = 100):
    """Compare FlexAction vs CGA on multiple workloads"""

    print("\n" + "="*70)
    print("FlexAction vs CGA Comparison Study")
    print("GPU: NVIDIA A100")
    print(f"Workloads: {len(workload_sizes)} GEMM tasks")
    print(f"Max trials per method: {max_trials}")
    print("="*70)

    all_results = []

    for workload in workload_sizes:
        M, N, K = workload

        # Run CGA
        cga_results = run_single_experiment('CGA', (M, N, K), max_trials)
        all_results.append(cga_results)

        # Run FlexAction
        flex_results = run_single_experiment('FLEXACTION', (M, N, K), max_trials)
        all_results.append(flex_results)

        # Compare
        print(f"\n{'='*70}")
        print(f"Comparison for GEMM {M}x{N}x{K}")
        print(f"{'='*70}")

        if 'error' not in cga_results and 'error' not in flex_results:
            cga_perf = cga_results['best_performance_gflops']
            flex_perf = flex_results['best_performance_gflops']

            improvement = ((flex_perf - cga_perf) / cga_perf * 100) if cga_perf > 0 else 0

            print(f"CGA Performance:        {cga_perf:.2f} GFLOPS")
            print(f"FlexAction Performance: {flex_perf:.2f} GFLOPS")
            print(f"Improvement:            {improvement:+.2f}%")

            cga_time = cga_results['elapsed_time']
            flex_time = flex_results['elapsed_time']
            print(f"\nCGA Time:               {cga_time:.2f}s")
            print(f"FlexAction Time:        {flex_time:.2f}s")

        print(f"{'='*70}\n")

    # Save comparison summary
    summary_file = '/root/Heron104/logs/comparison_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Complete comparison saved to: {summary_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='FlexAction vs CGA Comparison')
    parser.add_argument('--method', type=str, default='both',
                       choices=['flexaction', 'cga', 'both'],
                       help='Which method to run')
    parser.add_argument('--workload', type=str, default='64,64,64',
                       help='GEMM workload size as M,N,K')
    parser.add_argument('--trials', type=int, default=100,
                       help='Maximum number of trials')
    parser.add_argument('--compare', action='store_true',
                       help='Run full comparison on multiple workloads')

    args = parser.parse_args()

    # Setup environment
    print("Setting up environment...")
    gpu_available = setup_gpu_environment()

    if not gpu_available:
        print("Warning: No GPU detected. Results may not be accurate.")

    # Register FlexAction tuner
    register_flexaction_tuner()

    if args.compare:
        # Run full comparison
        workloads = [
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
        ]
        compare_methods(workloads, args.trials)
    else:
        # Parse workload
        M, N, K = map(int, args.workload.split(','))

        if args.method in ['cga', 'both']:
            run_single_experiment('CGA', (M, N, K), args.trials)

        if args.method in ['flexaction', 'both']:
            run_single_experiment('FLEXACTION', (M, N, K), args.trials)

    print("\n✓ All experiments completed!")


if __name__ == "__main__":
    main()
