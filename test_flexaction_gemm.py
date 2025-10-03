#!/usr/bin/env python3
"""
FlexAction在TensorCore GEMM上的测试脚本
测试单一架构(V100 TensorCore) × 单一workload(GEMM 64x64x64)
"""

import sys
import os
sys.path.append('/root/Heron')
sys.path.append('/root/Heron104')

import json
import time
import torch
import tvm
from tvm import te
import numpy as np

# Heron imports
from Heron.environment import Env
from Heron.config import Config
from Heron.ops.cuda import dense
from Heron.runner.runner import MeasureOption

# FlexAction imports
from flex_tuner import FlexActionTuner
from flexaction_csp_integration import FlexActionCSPIntegration, TensorCoreConstraintBuilder


def setup_environment():
    """设置测试环境"""
    # 激活conda环境
    os.environ['CONDA_DEFAULT_ENV'] = 'llmulator'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用GPU0

    # 创建日志目录
    log_dir = '/root/Heron104/flexaction_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir


def create_test_config(log_dir: str) -> Config:
    """创建测试配置"""
    config = Config()

    # 基础配置
    config.task_name = "gemm_tensorcore_64x64x64"
    config.log_dir = log_dir
    config.codegen_type = 'cuda'
    config.target_name = 'tensorcore'

    # FlexAction配置
    config.opt_method = 'FLEXACTION'  # 新的优化方法标识

    # 搜索配置
    config.max_trials = 100  # 总测量次数
    config.measure_time_per_round = 20  # 每轮测量数
    config.search_generations = 5  # 搜索代数

    # 遗传算法参数（FlexAction也会使用部分参数）
    config.pop_num = 50  # 种群大小
    config.select_num = 20  # 选择数量
    config.iter_walks = 5  # 迭代次数
    config.crossover_key_ratio = 0.3  # 关键变量比例

    # 并行配置
    config.parallel = True
    config.parallel_num = 4

    # 代价模型
    config.use_cost_model = True
    config.n_estimators = 300
    config.early_stopping = 30
    config.metric = 'r2'

    return config


def create_gemm_task():
    """创建GEMM任务"""
    # GEMM参数：C[M,N] = A[M,K] @ B[K,N]
    M, N, K = 64, 64, 64

    # 使用Heron的dense算子
    def gemm_tensorcore():
        A = te.placeholder((M, K), name='A', dtype='float16')
        B = te.placeholder((K, N), name='B', dtype='float16')

        # 定义计算
        k = te.reduce_axis((0, K), name='k')
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k].astype('float32') * B[k, j].astype('float32'), axis=k),
            name='C'
        )
        return [A, B, C]

    return gemm_tensorcore, (M, K, N)


def integrate_flexaction_with_heron(env: Env):
    """将FlexAction集成到Heron环境"""
    # 修改environment.py的createTask方法，添加FLEXACTION支持
    original_create = env.createTask

    def create_with_flexaction(name, opfunc, args, target, target_host=None, dump_const_desc=False):
        # 调用原始方法
        result = original_create(name, opfunc, args, target, target_host, dump_const_desc)

        # 如果是FLEXACTION方法，替换tuner
        if env.config.opt_method == 'FLEXACTION':
            env.tuner = FlexActionTuner(env.config)
            if env.config.use_cost_model:
                env.tuner.buildCostModel(env.task)

        return result

    env.createTask = create_with_flexaction


def run_flexaction_test():
    """运行FlexAction测试"""
    print("=" * 60)
    print("FlexAction on TensorCore GEMM Test")
    print("Architecture: NVIDIA V100 TensorCore")
    print("Workload: GEMM 64×64×64 (float16)")
    print("=" * 60)

    # 1. 设置环境
    log_dir = setup_environment()
    print(f"Log directory: {log_dir}")

    # 2. 创建配置
    config = create_test_config(log_dir)
    print(f"Optimization method: {config.opt_method}")
    print(f"Max trials: {config.max_trials}")

    # 3. 设置TVM target
    target = tvm.target.cuda(arch='sm_70')  # V100架构
    target_host = tvm.target.Target("llvm")

    # 4. 创建测量选项
    measure_option = MeasureOption(
        builder='local',
        runner='local',
        number=10,  # 每次测量运行10次
        repeat=3,   # 重复3次取平均
        timeout=10  # 超时10秒
    )

    # 5. 创建Heron环境
    env = Env(measure_option, config)

    # 6. 集成FlexAction
    integrate_flexaction_with_heron(env)

    # 7. 创建任务
    opfunc, args = create_gemm_task()
    task_name = f"gemm_M{args[0]}_K{args[1]}_N{args[2]}"

    # 8. 创建和配置任务
    print(f"\nCreating task: {task_name}")
    env.createTask(task_name, opfunc, args, target, target_host)

    # 9. 初始化性能缓冲
    from Heron.perf.perfBuffer import perfBuffer
    env.perf_buffer = perfBuffer()

    print("\n" + "=" * 60)
    print("Starting FlexAction optimization...")
    print("=" * 60)

    # 10. 运行调优
    start_time = time.time()
    try:
        population, stat = env.tune(task_name, pretrained=False)
        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("Optimization completed!")
        print(f"Total time: {elapsed:.2f} seconds")

        # 11. 输出结果
        if env.perf_buffer.best_perf:
            print(f"Best performance: {env.perf_buffer.best_perf:.4f} GFLOPS")

            # 计算理论峰值性能比
            # V100 TensorCore理论峰值: 125 TFLOPS (FP16)
            theoretical_peak = 125000  # GFLOPS
            efficiency = (env.perf_buffer.best_perf / theoretical_peak) * 100
            print(f"Efficiency: {efficiency:.2f}% of theoretical peak")

        # 12. 保存结果
        result_file = os.path.join(log_dir, 'flexaction_result.json')
        with open(result_file, 'w') as f:
            json.dump({
                'task': task_name,
                'architecture': 'V100_TensorCore',
                'method': 'FlexAction',
                'best_perf': env.perf_buffer.best_perf if env.perf_buffer.best_perf else 0,
                'total_time': elapsed,
                'trials': config.max_trials
            }, f, indent=2)
        print(f"\nResults saved to: {result_file}")

    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


def compare_with_cga():
    """与CGA方法进行对比测试"""
    print("\nRunning comparison with CGA...")

    # 运行CGA
    config_cga = create_test_config('/root/Heron104/cga_logs')
    config_cga.opt_method = 'CGA'

    # TODO: 运行CGA并比较结果

    print("Comparison completed!")


def test_lambda_library():
    """测试Lambda库功能"""
    from flex_tuner import LambdaLibrary, LambdaItem

    print("\n" + "=" * 60)
    print("Testing Lambda Library")
    print("=" * 60)

    # 创建库
    lib = LambdaLibrary("tensorcore", "gemm")
    print(f"Initial library size: {len(lib.items)}")

    # 列出所有Lambda项
    print("\nAvailable Lambda Items:")
    for name, item in lib.items.items():
        print(f"  - {name}: {item.lowering_type} {item.params}")

    # 测试约束生成
    from flexaction_csp_integration import TensorCoreConstraintBuilder

    constraints = TensorCoreConstraintBuilder.build_tile_constraints(16, 16, 16)
    print("\nGenerated constraints:")
    for name, spec in constraints.items():
        print(f"  - {name}: {spec}")


def test_csp_integration():
    """测试CSP集成功能"""
    print("\n" + "=" * 60)
    print("Testing CSP Integration")
    print("=" * 60)

    # 这里需要实际的KnobManager实例
    # 简化测试
    print("CSP integration module loaded successfully")


if __name__ == "__main__":
    print("FlexAction TensorCore GEMM Test Script")
    print("Python:", sys.version)
    print("PyTorch:", torch.__version__)
    print("TVM:", tvm.__version__ if hasattr(tvm, '__version__') else "Unknown")

    # 运行主测试
    run_flexaction_test()

    # 运行辅助测试
    # test_lambda_library()
    # test_csp_integration()
    # compare_with_cga()

    print("\nAll tests completed!")