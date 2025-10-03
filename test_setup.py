#!/usr/bin/env python3
"""
Quick test to verify FlexAction setup works
"""

import sys
import os
sys.path.append('/root/Heron')
sys.path.append('/root/Heron104')

print("Testing FlexAction Setup")
print("="*60)

# Test 1: GPU availability
print("\n1. Testing GPU...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA Version: {torch.version.cuda}")
    else:
        print("   ✗ No GPU detected")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: TVM
print("\n2. Testing TVM...")
try:
    import tvm
    from tvm import te
    print(f"   ✓ TVM imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Heron imports
print("\n3. Testing Heron imports...")
try:
    from Heron.environment import Env
    from Heron.config import Config
    from Heron.task.task import Task
    print("   ✓ Heron modules imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: FlexAction imports
print("\n4. Testing FlexAction imports...")
try:
    from flex_tuner import FlexActionTuner, LambdaLibrary, FlexActionPolicy
    print("   ✓ FlexAction tuner imported successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Lambda library
print("\n5. Testing Lambda Library...")
try:
    from flex_tuner import LambdaLibrary
    lib = LambdaLibrary("cuda", "gemm")
    print(f"   ✓ Lambda library created with {len(lib.items)} items")
    for name in list(lib.items.keys())[:3]:
        print(f"      - {name}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 6: Policy network
print("\n6. Testing Policy Network...")
try:
    from flex_tuner import FlexActionPolicy
    policy = FlexActionPolicy(state_dim=64, max_vocab_size=100)
    print(f"   ✓ Policy network created")
    print(f"      - State dim: 64")
    print(f"      - Max vocab size: 100")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 7: Create simple config
print("\n7. Testing Config creation...")
try:
    from Heron.config import Config
    config = Config()
    config.opt_method = 'FLEXACTION'
    config.max_trials = 10
    config.pop_num = 5
    config.parallel = False
    print(f"   ✓ Config created")
    print(f"      - Method: {config.opt_method}")
    print(f"      - Max trials: {config.max_trials}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("Setup test completed!")
print("\nIf all tests passed, you can run:")
print("  python /root/Heron104/run_flexaction_vs_cga.py --workload 64,64,64 --method flexaction --trials 50")
