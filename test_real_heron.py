#!/usr/bin/env python3
"""
FlexAction vs CGA Test Using Heron's Real Infrastructure
Following the patterns from Heron's actual test scripts
"""

import os
import sys
import time
import json
import argparse

# Setup paths
sys.path.insert(0, '/root')

import numpy as np
import tvm
import tvm.autotvm as autotvm
from tvm import te

# Heron imports
from Heron.environment import Env
import Heron.runner as HeronRunner
from Heron.config import Config
import Heron.ops.cuda as heron_cuda

# Setup GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def makeConfig(method='CGA', trials=20):
    """Create configuration following Heron's pattern"""
    config_data = {
        'config': {
            'out_name': f'test_{method.lower()}',
            'method': method,
            'max_trials': trials,
            'runner_number': 4,
            'runner_repeat': 3,
            'runner_timeout': 10,
            'build_timeout': 10,
            'in_dtype': 'float16',
            'out_dtype': 'float32'
        }
    }

    config = Config()
    config.initialize(config_data)

    # Set target for A100
    config.target_name = 'cuda'
    config.codegen_type = 'GPU_TENSOR_CORE'
    config.get_op = heron_cuda.getOpFromName
    config.device_id = 0

    # Log directory
    config.log_dir = f'/root/Heron104/results_{method.lower()}'
    os.makedirs(config.log_dir, exist_ok=True)

    return config

def run_test(op_name, params, method='CGA', trials=20):
    """Run test with specified method"""
    print(f"\n{'='*60}")
    print(f"Testing {method} with {op_name}")
    print(f"Parameters: {params}")
    print(f"Trials: {trials}")
    print(f"{'='*60}")

    # Create config
    config = makeConfig(method, trials)

    # Create measure option
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=5,
            repeat=3,
            timeout=10
        )
    )

    # Create environment
    env = Env(measure_option, config)

    # Set target
    target = tvm.target.Target(config.target_name)

    # Get operation
    op = config.get_op(op_name)
    task_name = f"{op_name}_{params}"

    # Create task
    start_time = time.time()
    task = env.createTask(task_name, op, params, target)
    task.device_id = config.device_id

    # Run tuning
    env.tune(task_name)

    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.2f} seconds")

    # Get best result
    if hasattr(env, 'perf_buffer') and env.perf_buffer.best_perf:
        best_perf = env.perf_buffer.best_perf
        print(f"Best performance: {best_perf:.4f}")
    else:
        best_perf = 0

    return {
        'method': method,
        'op': op_name,
        'params': params,
        'trials': trials,
        'time': elapsed,
        'best_perf': best_perf
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', type=str, default='dense',
                       choices=['dense', 'conv2d', 'batch_matmul'],
                       help='Operation to test')
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of trials')
    parser.add_argument('--compare', action='store_true',
                       help='Compare FlexAction vs CGA')

    args = parser.parse_args()

    print("=" * 70)
    print("FlexAction vs CGA Test on A100 GPU")
    print("=" * 70)

    # Define parameters based on operation
    if args.op == 'dense':
        params = (128, 128, 128)  # M, K, N
    elif args.op == 'conv2d':
        params = (1, 64, 56, 56, 64, 3, 1, 1)  # batch, in_c, h, w, out_c, kernel, stride, pad
    else:
        params = (4, 128, 128, 128)  # batch, M, K, N

    results = {}

    if args.compare:
        # First, integrate FlexAction
        try:
            from real_flexaction_integration import register_flexaction_to_heron
            register_flexaction_to_heron()
            print("✓ FlexAction registered")

            # Run FlexAction
            results['flexaction'] = run_test(args.op, params, 'FLEXACTION', args.trials)
        except Exception as e:
            print(f"FlexAction test failed: {e}")
            results['flexaction'] = {'error': str(e)}

        # Run CGA
        results['cga'] = run_test(args.op, params, 'CGA', args.trials)

        # Compare
        print("\n" + "=" * 70)
        print("COMPARISON RESULTS")
        print("=" * 70)

        if 'flexaction' in results and 'cga' in results:
            if 'error' not in results['flexaction'] and 'error' not in results['cga']:
                fa = results['flexaction']
                cga = results['cga']

                print(f"{'Method':<15} {'Time (s)':<15} {'Best Perf':<15}")
                print("-" * 45)
                print(f"{'FlexAction':<15} {fa['time']:<15.2f} {fa['best_perf']:<15.4f}")
                print(f"{'CGA':<15} {cga['time']:<15.2f} {cga['best_perf']:<15.4f}")

                if fa['best_perf'] > 0 and cga['best_perf'] > 0:
                    speedup = fa['best_perf'] / cga['best_perf']
                    print(f"\nPerformance ratio: {speedup:.2f}x")

    else:
        # Run single test
        results = run_test(args.op, params, 'CGA', args.trials)

    # Save results
    result_file = f'/root/Heron104/test_results_{args.op}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_file}")

if __name__ == "__main__":
    main()