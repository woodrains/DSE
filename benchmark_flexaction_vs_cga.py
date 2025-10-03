#!/usr/bin/env python3
"""
FlexAction vs CGA Comparison on Real GPU Workloads
Using A100 GPU and actual TVM compilation
"""

import os
import sys
import time
import json
import argparse
import numpy as np

# Setup paths
sys.path.insert(0, '/root/Heron')
sys.path.insert(0, '/root/Heron104')

# GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU0
os.environ['TVM_NUM_THREADS'] = '16'

print("Initializing environment...")

# Import modules after path setup
import tvm
from tvm import te, auto_scheduler
import torch

# Import Heron
from Heron.environment import Env
from Heron.config import Config
from Heron.runner.runner import MeasureOption, Runner
from Heron.perf.perfBuffer import perfBuffer

# Import FlexAction
from real_flexaction_integration import register_flexaction_to_heron

# Register FlexAction
register_flexaction_to_heron()


class BenchmarkRunner:
    """Run benchmarks comparing FlexAction and CGA"""

    def __init__(self, workload_name, log_base_dir='/root/Heron104/benchmark_results'):
        self.workload_name = workload_name
        self.log_base_dir = log_base_dir
        self.results = {}

        # Create result directories
        os.makedirs(log_base_dir, exist_ok=True)
        self.flexaction_log = os.path.join(log_base_dir, f'{workload_name}_flexaction')
        self.cga_log = os.path.join(log_base_dir, f'{workload_name}_cga')
        os.makedirs(self.flexaction_log, exist_ok=True)
        os.makedirs(self.cga_log, exist_ok=True)

    def create_config(self, method='FLEXACTION', trials=100):
        """Create configuration for optimization"""
        config = Config()

        # Basic settings
        config.task_name = self.workload_name
        config.codegen_type = 'cuda'
        config.target_name = 'gpu'
        config.opt_method = method  # 'FLEXACTION' or 'CGA'

        # Optimization settings
        config.max_trials = trials
        config.measure_time_per_round = 20
        config.search_generations = max(1, trials // 20)

        # GA/FlexAction parameters
        config.pop_num = 50
        config.select_num = 20
        config.iter_walks = 5
        config.crossover_key_ratio = 0.3

        # Parallel settings
        config.parallel = True
        config.parallel_num = 4

        # Cost model
        config.use_cost_model = True
        config.n_estimators = 300
        config.early_stopping = 30
        config.metric = 'r2'

        # Logging
        if method == 'FLEXACTION':
            config.log_dir = self.flexaction_log
        else:
            config.log_dir = self.cga_log

        return config

    def create_workload(self, workload_type='gemm'):
        """Create workload function"""
        if workload_type == 'gemm':
            M, N, K = 512, 512, 512

            def gemm():
                A = te.placeholder((M, K), name='A', dtype='float16')
                B = te.placeholder((K, N), name='B', dtype='float16')
                k = te.reduce_axis((0, K), name='k')
                C = te.compute(
                    (M, N),
                    lambda i, j: te.sum(
                        A[i, k].astype('float32') * B[k, j].astype('float32'),
                        axis=k
                    ),
                    name='C'
                )
                return [A, B, C]

            return gemm, (M, K, N), 'gemm'

        elif workload_type == 'conv2d':
            # Conv2d workload
            batch, in_channel, height, width = 1, 64, 56, 56
            out_channel, kernel = 64, 3
            pad, stride = 1, 1

            def conv2d():
                A = te.placeholder((batch, in_channel, height, width), name='A', dtype='float16')
                W = te.placeholder((out_channel, in_channel, kernel, kernel), name='W', dtype='float16')

                # Compute output dimensions
                out_h = (height + 2 * pad - kernel) // stride + 1
                out_w = (width + 2 * pad - kernel) // stride + 1

                # Pad input
                A_pad = te.compute(
                    (batch, in_channel, height + 2 * pad, width + 2 * pad),
                    lambda n, c, h, w: te.if_then_else(
                        te.all(h >= pad, h < height + pad, w >= pad, w < width + pad),
                        A[n, c, h - pad, w - pad],
                        0.0
                    ),
                    name='A_pad'
                )

                # Convolution
                rc = te.reduce_axis((0, in_channel), name='rc')
                ry = te.reduce_axis((0, kernel), name='ry')
                rx = te.reduce_axis((0, kernel), name='rx')

                B = te.compute(
                    (batch, out_channel, out_h, out_w),
                    lambda n, f, y, x: te.sum(
                        A_pad[n, rc, y * stride + ry, x * stride + rx].astype('float32') *
                        W[f, rc, ry, rx].astype('float32'),
                        axis=[rc, ry, rx]
                    ),
                    name='B'
                )
                return [A, W, B]

            return conv2d, (batch, in_channel, height, width, out_channel, kernel), 'conv2d'

        elif workload_type == 'batch_matmul':
            batch, M, N, K = 8, 256, 256, 256

            def batch_matmul():
                A = te.placeholder((batch, M, K), name='A', dtype='float16')
                B = te.placeholder((batch, K, N), name='B', dtype='float16')
                k = te.reduce_axis((0, K), name='k')
                C = te.compute(
                    (batch, M, N),
                    lambda b, i, j: te.sum(
                        A[b, i, k].astype('float32') * B[b, k, j].astype('float32'),
                        axis=k
                    ),
                    name='C'
                )
                return [A, B, C]

            return batch_matmul, (batch, M, K, N), 'batch_matmul'

    def run_optimization(self, method='FLEXACTION', workload_type='gemm', trials=100):
        """Run optimization with specified method"""
        print(f"\n{'='*60}")
        print(f"Running {method} on {workload_type}")
        print(f"{'='*60}")

        # Create configuration
        config = self.create_config(method=method, trials=trials)

        # Set target
        if torch.cuda.is_available():
            # Get GPU compute capability
            device_props = torch.cuda.get_device_properties(0)
            if 'A100' in device_props.name:
                target = tvm.target.cuda(arch='sm_80')  # A100
            elif 'V100' in device_props.name:
                target = tvm.target.cuda(arch='sm_70')  # V100
            else:
                target = tvm.target.cuda()
        else:
            print("Warning: CUDA not available, using CPU target")
            target = tvm.target.Target("llvm")

        target_host = tvm.target.Target("llvm")

        print(f"Target: {target}")

        # Create measurement options
        measure_option = MeasureOption(
            builder='local',
            runner='local',
            number=10,
            repeat=3,
            min_repeat_ms=0,
            timeout=10
        )

        # Create environment
        env = Env(measure_option, config)

        # Initialize performance buffer
        env.perf_buffer = perfBuffer()

        # Create workload
        opfunc, args, name = self.create_workload(workload_type)
        task_name = f"{name}_{args}"

        # Create task
        env.createTask(task_name, opfunc, args, target, target_host)

        # Run optimization
        start_time = time.time()
        try:
            population, stat = env.tune(task_name, pretrained=False)
            elapsed = time.time() - start_time

            # Get results
            best_perf = env.perf_buffer.best_perf if env.perf_buffer.best_perf else 0
            best_latency = 1000.0 / best_perf if best_perf > 0 else float('inf')

            # Calculate FLOPS
            if workload_type == 'gemm':
                M, K, N = args[0], args[1], args[2]
                flop = 2 * M * K * N  # Matrix multiplication FLOPs
            elif workload_type == 'conv2d':
                batch, in_c, h, w, out_c, k = args
                out_h = h  # Simplified
                out_w = w  # Simplified
                flop = 2 * batch * out_c * out_h * out_w * in_c * k * k
            elif workload_type == 'batch_matmul':
                batch, M, K, N = args
                flop = 2 * batch * M * K * N

            gflops = (flop / best_latency) / 1e9 if best_latency < float('inf') else 0

            result = {
                'method': method,
                'workload': workload_type,
                'trials': trials,
                'time': elapsed,
                'best_perf': best_perf,
                'best_latency_ms': best_latency,
                'gflops': gflops
            }

            print(f"\nResults:")
            print(f"  Time: {elapsed:.2f} seconds")
            print(f"  Best latency: {best_latency:.3f} ms")
            print(f"  Performance: {gflops:.2f} GFLOPS")

            # Print Lambda statistics if FlexAction
            if method == 'FLEXACTION' and hasattr(env.tuner, 'print_lambda_stats'):
                env.tuner.print_lambda_stats()

            return result

        except Exception as e:
            print(f"Error during optimization: {e}")
            import traceback
            traceback.print_exc()
            return {
                'method': method,
                'workload': workload_type,
                'error': str(e)
            }

    def run_comparison(self, workload_type='gemm', trials=100):
        """Run comparison between FlexAction and CGA"""
        print(f"\n{'='*70}")
        print(f"Comparing FlexAction vs CGA on {workload_type}")
        print(f"Trials: {trials}")
        print(f"{'='*70}")

        # Run FlexAction
        flexaction_result = self.run_optimization('FLEXACTION', workload_type, trials)
        self.results['flexaction'] = flexaction_result

        # Run CGA
        cga_result = self.run_optimization('CGA', workload_type, trials)
        self.results['cga'] = cga_result

        # Compare results
        self.print_comparison()

        # Save results
        self.save_results()

        return self.results

    def print_comparison(self):
        """Print comparison results"""
        print(f"\n{'='*70}")
        print("COMPARISON RESULTS")
        print(f"{'='*70}")

        if 'flexaction' in self.results and 'cga' in self.results:
            fa = self.results['flexaction']
            cga = self.results['cga']

            if 'error' not in fa and 'error' not in cga:
                # Calculate speedup
                speedup = cga.get('best_latency_ms', float('inf')) / fa.get('best_latency_ms', 1)
                time_speedup = cga.get('time', float('inf')) / fa.get('time', 1)

                print(f"\nPerformance Comparison:")
                print(f"{'Method':<15} {'Latency (ms)':<15} {'GFLOPS':<15} {'Time (s)':<15}")
                print("-" * 60)
                print(f"{'FlexAction':<15} {fa.get('best_latency_ms', 0):<15.3f} "
                      f"{fa.get('gflops', 0):<15.2f} {fa.get('time', 0):<15.2f}")
                print(f"{'CGA':<15} {cga.get('best_latency_ms', 0):<15.3f} "
                      f"{cga.get('gflops', 0):<15.2f} {cga.get('time', 0):<15.2f}")
                print("-" * 60)
                print(f"{'Speedup':<15} {speedup:<15.2f}x {'':15} {time_speedup:<15.2f}x")

                # Winner
                if speedup > 1:
                    print(f"\n✓ FlexAction is {speedup:.2f}x faster than CGA")
                else:
                    print(f"\n✓ CGA is {1/speedup:.2f}x faster than FlexAction")

    def save_results(self):
        """Save results to JSON file"""
        result_file = os.path.join(self.log_base_dir, f'{self.workload_name}_comparison.json')
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {result_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='FlexAction vs CGA Benchmark')
    parser.add_argument('--workload', type=str, default='gemm',
                       choices=['gemm', 'conv2d', 'batch_matmul'],
                       help='Workload type to benchmark')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials for optimization')
    parser.add_argument('--method', type=str, default='both',
                       choices=['flexaction', 'cga', 'both'],
                       help='Method to run')

    args = parser.parse_args()

    print("=" * 70)
    print("FlexAction vs CGA Real GPU Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available'}")
    print(f"TVM: {tvm.__version__ if hasattr(tvm, '__version__') else 'Unknown'}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 70)

    # Create benchmark runner
    runner = BenchmarkRunner(args.workload)

    # Run benchmark
    if args.method == 'both':
        runner.run_comparison(args.workload, args.trials)
    elif args.method == 'flexaction':
        runner.run_optimization('FLEXACTION', args.workload, args.trials)
    elif args.method == 'cga':
        runner.run_optimization('CGA', args.workload, args.trials)

    print("\n" + "=" * 70)
    print("Benchmark completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()