#!/usr/bin/env python3
"""
Simplified FlexAction Test with Heron
Direct integration test without complex imports
"""

import os
import sys
import time
import json

# Setup environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/Heron104')

import numpy as np

# Import from Heron
from Heron.config import Config
from Heron.environment import Env
from Heron.tuner.ga_tuner import CGATuner

# Import our FlexAction components
from real_flexaction_integration import RealFlexActionTuner, register_flexaction_to_heron

def create_simple_config(method='CGA'):
    """Create a simple configuration"""
    config_data = {
        'config': {
            'out_name': 'test_output',
            'method': method,
            'max_trials': 20,
            'runner_number': 2,
            'runner_repeat': 3,
            'runner_timeout': 10,
            'build_timeout': 10,
            'in_dtype': 'float16',
            'out_dtype': 'float32',
            'cases': [
                {'M': 64, 'K': 64, 'N': 64}
            ]
        }
    }

    config = Config()
    config.initialize(config_data)
    config.opt_method = method
    config.log_dir = f'/root/Heron104/test_{method.lower()}'
    os.makedirs(config.log_dir, exist_ok=True)

    # Additional parameters
    config.pop_num = 20
    config.select_num = 10
    config.iter_walks = 3
    config.measure_time_per_round = 10
    config.search_generations = 2
    config.crossover_key_ratio = 0.3
    config.parallel = False  # Simplify for testing
    config.use_cost_model = True

    return config

def test_flexaction_simple():
    """Run a simple test of FlexAction"""
    print("=" * 60)
    print("Simple FlexAction Test")
    print("=" * 60)

    # Register FlexAction
    register_flexaction_to_heron()

    # Test with both methods
    methods = ['FLEXACTION', 'CGA']
    results = {}

    for method in methods:
        print(f"\n--- Testing {method} ---")

        # Create config
        config = create_simple_config(method)

        # Create environment
        from tvm.autotvm.measure.measure import MeasureOption
        measure_option = MeasureOption(
            builder='local',
            runner='local',
            number=5,
            repeat=1,
            timeout=10
        )

        env = Env(measure_option, config)

        # Simple GEMM workload
        import tvm
        from tvm import te

        # Get TVM target
        if 'A100' in str(os.popen('nvidia-smi --query-gpu=name --format=csv,noheader -i 0').read()):
            target = tvm.target.cuda(arch='sm_80')
        else:
            target = tvm.target.cuda()
        target_host = tvm.target.Target("llvm")

        # Define simple GEMM
        M, N, K = 64, 64, 64
        def simple_gemm():
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

        # Create task
        opfunc, args = simple_gemm, (M, K, N)
        task_name = f"gemm_{M}_{K}_{N}"

        # Initialize performance buffer
        from Heron.perf.perfBuffer import perfBuffer
        env.perf_buffer = perfBuffer()

        try:
            # Create task
            env.createTask(task_name, opfunc, args, target, target_host)

            # Run tuning
            start_time = time.time()
            population, stat = env.tune(task_name, pretrained=False)
            elapsed = time.time() - start_time

            # Get results
            best_perf = env.perf_buffer.best_perf if env.perf_buffer.best_perf else 0

            result = {
                'method': method,
                'time': elapsed,
                'best_perf': best_perf,
                'trials': config.max_trials
            }

            print(f"\n{method} Results:")
            print(f"  Time: {elapsed:.2f}s")
            print(f"  Best perf: {best_perf:.4f}")

            # Print Lambda stats if FlexAction
            if method == 'FLEXACTION' and hasattr(env.tuner, 'lambda_usage_stats'):
                print("\n  Lambda Usage:")
                for name, stats in list(env.tuner.lambda_usage_stats.items())[:5]:
                    if stats['count'] > 0:
                        print(f"    {name}: count={stats['count']}, avg_reward={stats['avg_reward']:.3f}")

            results[method] = result

        except Exception as e:
            print(f"Error with {method}: {e}")
            results[method] = {'error': str(e)}

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    if 'FLEXACTION' in results and 'CGA' in results:
        if 'error' not in results['FLEXACTION'] and 'error' not in results['CGA']:
            fa_perf = results['FLEXACTION'].get('best_perf', 0)
            cga_perf = results['CGA'].get('best_perf', 0)

            if fa_perf > 0 and cga_perf > 0:
                speedup = fa_perf / cga_perf
                print(f"Performance speedup: {speedup:.2f}x")

                if speedup > 1:
                    print(f"✓ FlexAction is {speedup:.2f}x better than CGA")
                else:
                    print(f"✓ CGA is {1/speedup:.2f}x better than FlexAction")

            fa_time = results['FLEXACTION'].get('time', float('inf'))
            cga_time = results['CGA'].get('time', float('inf'))
            time_ratio = cga_time / fa_time
            print(f"Time efficiency: {time_ratio:.2f}x")

    # Save results
    result_file = '/root/Heron104/simple_test_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_file}")

    return results

if __name__ == "__main__":
    print("Starting Simple FlexAction Test")
    print("GPU: A100")
    print("=" * 60)

    results = test_flexaction_simple()

    print("\nTest completed!")