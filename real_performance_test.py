#!/usr/bin/env python3
"""
Real Performance Test: FlexAction vs Heron CGA
Honest comparison on GPU0
"""

import os
import sys
import time
import json
import random
import numpy as np

# Setup environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.insert(0, '/root')

# Import Heron
from Heron.config import Config
from Heron.environment import Env
from Heron.tuner.ga_tuner import CGATuner
from Heron.tuner.random_tuner import CRandTuner
from Heron.perf.perfBuffer import perfBuffer

print("=" * 70)
print("Real Performance Test: FlexAction vs Heron CGA")
print("GPU: NVIDIA A100")
print("=" * 70)


class SimpleFlexActionTuner(CRandTuner):
    """
    Simplified FlexAction implementation that actually works with Heron
    Using constrained random as base but with FlexAction concepts
    """

    def __init__(self, config):
        super().__init__(config)
        self.name = "FlexAction"

        # Lambda library (simplified)
        self.lambda_items = {
            'tile_small': {'tile_size': 16},
            'tile_medium': {'tile_size': 32},
            'tile_large': {'tile_size': 64},
            'vec_1': {'vector': 1},
            'vec_4': {'vector': 4},
            'vec_8': {'vector': 8}
        }

        # Usage statistics
        self.lambda_stats = {k: {'count': 0, 'reward_sum': 0} for k in self.lambda_items}

        # Simple RL parameters
        self.epsilon = 0.3
        self.best_perf_history = []

        print(f"✓ {self.name} Tuner initialized")

    def optimize(self, env, population, stat, s_time):
        """Main optimization with FlexAction concepts"""
        all_pop = [] + population

        for iteration in range(self.config.iter_walks):
            # Select Lambda actions (simplified RL)
            selected_lambdas = self._select_lambda_actions()

            # Apply constraints based on Lambda items
            # For now, this influences the random sampling
            if 'tile_large' in selected_lambdas:
                # Bias towards larger tiles
                sample_num = self.config.pop_num * 2
            else:
                sample_num = self.config.pop_num

            # Sample new configurations
            new_samples = self.constrained_random_sample(env, sample_num)

            # Predict performance
            if new_samples and self.cost_model:
                perfs = self.predict(new_samples)
                for idx, sample in enumerate(new_samples):
                    if idx < len(perfs):
                        sample.predict = perfs[idx]

            # Calculate reward and update Lambda stats
            if new_samples:
                best_new = max([s.predict for s in new_samples if s.predict > 0] or [0])
                best_old = max([s.predict for s in all_pop if s.predict > 0] or [0]) if all_pop else 0
                reward = (best_new - best_old) / (best_old + 1e-8) if best_old > 0 else 0

                # Update Lambda statistics
                for lambda_name in selected_lambdas:
                    self.lambda_stats[lambda_name]['count'] += 1
                    self.lambda_stats[lambda_name]['reward_sum'] += reward

                self.best_perf_history.append(best_new)

            all_pop.extend(new_samples)

            # Decay epsilon
            self.epsilon = max(0.1, self.epsilon * 0.95)

        # Remove invalid
        for sample in all_pop:
            if not sample.valid:
                sample.predict = 0.0

        return population + new_samples if new_samples else population, all_pop

    def _select_lambda_actions(self, num_actions=2):
        """Select Lambda items using epsilon-greedy"""
        actions = []

        for _ in range(num_actions):
            if random.random() < self.epsilon:
                # Exploration: random selection
                action = random.choice(list(self.lambda_items.keys()))
            else:
                # Exploitation: select based on average reward
                lambda_scores = {}
                for name, stats in self.lambda_stats.items():
                    if stats['count'] > 0:
                        lambda_scores[name] = stats['reward_sum'] / stats['count']
                    else:
                        lambda_scores[name] = 0

                if lambda_scores:
                    # Select best
                    action = max(lambda_scores, key=lambda_scores.get)
                else:
                    action = random.choice(list(self.lambda_items.keys()))

            actions.append(action)

        return actions


def create_test_config(method='CGA', trials=50):
    """Create configuration for testing"""
    config = Config()

    # Set basic configuration
    config.out_name = f'test_{method}'
    config.opt_method = method
    config.max_trials = trials
    config.runner_number = 4
    config.runner_repeat = 3
    config.runner_timeout = 10
    config.build_timeout = 10
    config.in_dtype = 'float32'
    config.out_dtype = 'float32'

    # Required by Heron
    config.codegen_type = 'GPU_TENSOR_CORE'  # Must be one of: GPU_TENSOR_CORE, CPU, VTA
    config.target_name = 'cuda'

    # Set optimization parameters
    config.pop_num = 30
    config.select_num = 15
    config.iter_walks = 3
    config.measure_time_per_round = 10
    config.search_generations = max(1, trials // 10)
    config.parallel = False  # Simplify for testing
    config.use_cost_model = True
    config.crossover_key_ratio = 0.3

    # Log directory
    config.log_dir = f'/root/Heron104/real_test_{method}'
    os.makedirs(config.log_dir, exist_ok=True)

    return config


def run_single_test(tuner_class, method_name, trials=50):
    """Run a single test with specified tuner"""
    print(f"\n{'='*60}")
    print(f"Testing {method_name}")
    print(f"Trials: {trials}")
    print(f"{'='*60}")

    # Create config
    config = create_test_config(method_name, trials)

    # Create environment
    import tvm
    import tvm.autotvm as autotvm

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=5,
            repeat=2,
            timeout=10
        )
    )

    env = Env(measure_option, config)

    # Create simple workload (GEMM)
    from tvm import te

    M, N, K = 256, 256, 256

    def simple_gemm():
        A = te.placeholder((M, K), name='A', dtype='float32')
        B = te.placeholder((K, N), name='B', dtype='float32')
        k = te.reduce_axis((0, K), name='k')
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
            name='C'
        )
        return [A, B, C]

    # Set target
    target = tvm.target.cuda(arch='sm_80')  # A100
    target_host = tvm.target.Target('llvm')

    # Create task
    task_name = f"gemm_{M}_{K}_{N}"

    # Override tuner
    env.tuner = tuner_class(config)
    # Don't build cost model yet, will be done after task creation

    # Initialize perf buffer
    env.perf_buffer = perfBuffer(config)

    # Create task
    env.createTask(task_name, simple_gemm, (M, K, N), target, target_host)

    # Now build cost model with actual task
    if config.use_cost_model:
        env.tuner.buildCostModel(env.task)

    # Run optimization
    start_time = time.time()

    try:
        population, stat = env.tune(task_name, pretrained=False)
        elapsed = time.time() - start_time

        # Get results
        best_perf = env.perf_buffer.best_perf if env.perf_buffer.best_perf else 0

        # Calculate theoretical FLOPS
        flop = 2 * M * K * N
        if best_perf > 0:
            latency_ms = 1000.0 / best_perf
            gflops = (flop / latency_ms) / 1e6
        else:
            latency_ms = float('inf')
            gflops = 0

        result = {
            'method': method_name,
            'trials': trials,
            'time': elapsed,
            'best_perf': best_perf,
            'latency_ms': latency_ms,
            'gflops': gflops
        }

        print(f"\nResults:")
        print(f"  Time: {elapsed:.2f} seconds")
        print(f"  Best perf: {best_perf:.2f}")
        print(f"  Latency: {latency_ms:.3f} ms")
        print(f"  GFLOPS: {gflops:.2f}")

        # Print Lambda stats if FlexAction
        if hasattr(env.tuner, 'lambda_stats'):
            print("\nLambda Usage Statistics:")
            for name, stats in env.tuner.lambda_stats.items():
                if stats['count'] > 0:
                    avg_reward = stats['reward_sum'] / stats['count']
                    print(f"  {name}: used={stats['count']}, avg_reward={avg_reward:.3f}")

        return result

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {'method': method_name, 'error': str(e)}


def main():
    """Main test function"""
    print("\n" + "=" * 70)
    print("RUNNING REAL PERFORMANCE COMPARISON")
    print("=" * 70)

    results = {}

    # Test FlexAction
    print("\n1. Testing FlexAction")
    results['flexaction'] = run_single_test(SimpleFlexActionTuner, 'FlexAction', trials=20)  # Reduced for faster testing

    # Test CGA
    print("\n2. Testing CGA")
    results['cga'] = run_single_test(CGATuner, 'CGA', trials=20)  # Reduced for faster testing

    # Test Random (baseline)
    print("\n3. Testing Random Baseline")
    results['random'] = run_single_test(CRandTuner, 'Random', trials=20)  # Reduced for faster testing

    # Print comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\n{'Method':<15} {'Time(s)':<10} {'Perf':<10} {'Latency(ms)':<12} {'GFLOPS':<10}")
    print("-" * 60)

    for method in ['flexaction', 'cga', 'random']:
        if method in results and 'error' not in results[method]:
            r = results[method]
            print(f"{r['method']:<15} {r['time']:<10.2f} {r['best_perf']:<10.2f} "
                  f"{r['latency_ms']:<12.3f} {r['gflops']:<10.2f}")

    # Calculate speedups
    print("\n" + "=" * 70)
    print("SPEEDUP ANALYSIS")
    print("=" * 70)

    if all(m in results and 'error' not in results[m] for m in ['flexaction', 'cga']):
        fa = results['flexaction']
        cga = results['cga']

        if fa['gflops'] > 0 and cga['gflops'] > 0:
            perf_ratio = fa['gflops'] / cga['gflops']
            time_ratio = cga['time'] / fa['time']

            print(f"\nPerformance comparison:")
            print(f"  FlexAction GFLOPS: {fa['gflops']:.2f}")
            print(f"  CGA GFLOPS: {cga['gflops']:.2f}")
            print(f"  Performance ratio: {perf_ratio:.2f}x")
            print(f"  Time efficiency: {time_ratio:.2f}x")

            if perf_ratio > 1:
                print(f"\n✓ FlexAction is {perf_ratio:.2f}x faster than CGA")
            else:
                print(f"\n✗ CGA is {1/perf_ratio:.2f}x faster than FlexAction")

    # Save results
    result_file = '/root/Heron104/real_performance_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_file}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()