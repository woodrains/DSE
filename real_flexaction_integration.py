#!/usr/bin/env python3
"""
Real FlexAction Integration with Heron
在真实GPU架构上运行并与CGA进行对比
"""

import os
import sys
import time
import json
import copy

# Set up paths
sys.path.insert(0, '/root/Heron')
sys.path.insert(0, '/root/Heron104')

# Environment setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU0
os.environ['TVM_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple

# Import TVM
import tvm
from tvm import te, tir, auto_scheduler
from tvm.autotvm.measure import measure_option
from tvm.autotvm.tuner import XGBTuner

# Import Heron modules
from Heron.environment import Env
from Heron.config import Config
from Heron.tuner.tuner import Tuner
from Heron.tuner.ga_tuner import CGATuner
from Heron.sample import Sample, Code
from Heron.runner.runner import Runner
from Heron.perf.perfBuffer import perfBuffer
from Heron.task.task import Task
from Heron.utils import anaCostModel

# Import FlexAction modules
from flexaction_csp_integration import FlexActionCSPIntegration, TensorCoreConstraintBuilder


Experience = namedtuple('Experience',
    ['state', 'action', 'reward', 'next_state', 'done'])


class RealFlexActionTuner(Tuner):
    """Real FlexAction Tuner integrated with Heron"""

    def __init__(self, config):
        super().__init__(config)

        # Lambda library
        self.lambda_library = {}
        self.lambda_usage_stats = {}

        # RL components
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.replay_buffer = deque(maxlen=10000)

        # Hyperparameters
        self.batch_size = 32
        self.learning_rate = 5e-4
        self.gamma = 0.95
        self.epsilon = 0.3
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.target_update_freq = 10

        # Performance tracking
        self.episode_rewards = []
        self.best_configs = []

        print("✓ Real FlexAction Tuner initialized")

    def initialize(self, env):
        """Initialize FlexAction components with real environment"""
        # Initialize Lambda library based on target architecture
        self._init_lambda_library(env)

        # Initialize neural networks
        state_dim = self._get_state_dim(env)
        action_dim = len(self.lambda_library)

        self.policy_net = self._build_network(state_dim, action_dim).cuda()
        self.target_net = self._build_network(state_dim, action_dim).cuda()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        print(f"✓ Initialized RL components: state_dim={state_dim}, action_dim={action_dim}")

    def _init_lambda_library(self, env):
        """Initialize Lambda library based on target architecture"""
        target = str(env.task.target)

        if 'cuda' in target or 'gpu' in target:
            # A100/V100 GPU Lambda items
            self.lambda_library = {
                # TensorCore configurations
                'tc_8x8x8': {
                    'type': 'tensorcore',
                    'params': {'m': 8, 'n': 8, 'k': 8},
                    'constraints': [('IN', 'm', [8]), ('IN', 'n', [8]), ('IN', 'k', [8])]
                },
                'tc_16x16x16': {
                    'type': 'tensorcore',
                    'params': {'m': 16, 'n': 16, 'k': 16},
                    'constraints': [('IN', 'm', [16]), ('IN', 'n', [16]), ('IN', 'k', [16])]
                },
                'tc_32x32x8': {
                    'type': 'tensorcore',
                    'params': {'m': 32, 'n': 32, 'k': 8},
                    'constraints': [('IN', 'm', [32]), ('IN', 'n', [32]), ('IN', 'k', [8])]
                },

                # Memory hierarchy
                'shared_small': {
                    'type': 'memory',
                    'params': {'size': 16*1024},
                    'constraints': [('LE', 'shared_mem', 16*1024)]
                },
                'shared_medium': {
                    'type': 'memory',
                    'params': {'size': 32*1024},
                    'constraints': [('LE', 'shared_mem', 32*1024)]
                },
                'shared_large': {
                    'type': 'memory',
                    'params': {'size': 48*1024},
                    'constraints': [('LE', 'shared_mem', 48*1024)]
                },

                # Vectorization
                'vec_1': {
                    'type': 'vectorize',
                    'params': {'length': 1},
                    'constraints': [('IN', 'vector_length', [1])]
                },
                'vec_4': {
                    'type': 'vectorize',
                    'params': {'length': 4},
                    'constraints': [('IN', 'vector_length', [4])]
                },
                'vec_8': {
                    'type': 'vectorize',
                    'params': {'length': 8},
                    'constraints': [('IN', 'vector_length', [8])]
                },

                # Tiling strategies
                'tile_small': {
                    'type': 'tiling',
                    'params': {'tile_i': 16, 'tile_j': 16},
                    'constraints': [('EQ', 'tile_i', 16), ('EQ', 'tile_j', 16)]
                },
                'tile_medium': {
                    'type': 'tiling',
                    'params': {'tile_i': 32, 'tile_j': 32},
                    'constraints': [('EQ', 'tile_i', 32), ('EQ', 'tile_j', 32)]
                },
                'tile_large': {
                    'type': 'tiling',
                    'params': {'tile_i': 64, 'tile_j': 64},
                    'constraints': [('EQ', 'tile_i', 64), ('EQ', 'tile_j', 64)]
                }
            }

        # Initialize usage statistics
        for key in self.lambda_library:
            self.lambda_usage_stats[key] = {
                'count': 0,
                'total_reward': 0,
                'avg_reward': 0
            }

    def _get_state_dim(self, env):
        """Get state dimension from environment"""
        # Extract features from CSP
        return 128  # Fixed dimension for neural network

    def _build_network(self, input_dim, output_dim):
        """Build neural network for policy"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def extract_state(self, env, population=None):
        """Extract state features from environment"""
        features = []

        # Task features
        if hasattr(env, 'task'):
            task = env.task
            # Get knob manager info
            if hasattr(task, 'knob_manager'):
                km = task.knob_manager
                if hasattr(km, 'solver') and hasattr(km.solver, 'vals'):
                    features.append(len(km.solver.vals))  # Number of variables
                else:
                    features.append(0)

                if hasattr(km, 'solver') and hasattr(km.solver, 'constraints'):
                    features.append(len(km.solver.constraints))  # Number of constraints
                else:
                    features.append(0)
            else:
                features.extend([0, 0])

            # Workload features
            if hasattr(task, 'args'):
                features.extend(list(task.args)[:10])  # First 10 args

        # Performance features
        if hasattr(env, 'perf_buffer'):
            if env.perf_buffer.best_perf:
                features.append(env.perf_buffer.best_perf)
            else:
                features.append(0)

            features.append(len(env.perf_buffer.data_x))  # Number of measurements
        else:
            features.extend([0, 0])

        # Population features
        if population:
            valid_samples = [s for s in population if s.valid]
            features.append(len(valid_samples))
            if valid_samples:
                perfs = [s.predict for s in valid_samples if s.predict > 0]
                if perfs:
                    features.append(np.mean(perfs))
                    features.append(np.std(perfs))
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0])

        # Pad to fixed dimension
        while len(features) < 128:
            features.append(0)

        return np.array(features[:128], dtype=np.float32)

    def select_actions(self, state, num_actions=3):
        """Select multiple Lambda actions using epsilon-greedy"""
        actions = []
        action_names = list(self.lambda_library.keys())

        state_tensor = torch.FloatTensor(state).unsqueeze(0).cuda()

        for _ in range(num_actions):
            if np.random.random() < self.epsilon:
                # Exploration: random action
                action_idx = np.random.randint(len(action_names))
            else:
                # Exploitation: best action from Q-network
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                    action_idx = q_values.argmax(dim=1).item()

            actions.append(action_idx)
            action_name = action_names[action_idx]

            # Update usage stats
            self.lambda_usage_stats[action_name]['count'] += 1

        return actions

    def apply_lambda_actions(self, env, action_indices):
        """Apply selected Lambda actions to CSP"""
        action_names = list(self.lambda_library.keys())

        # Get current knob manager
        km = copy.deepcopy(env.task.knob_manager)

        # Apply each Lambda action's constraints
        for idx in action_indices:
            action_name = action_names[idx]
            lambda_item = self.lambda_library[action_name]

            # Apply constraints
            for constraint in lambda_item.get('constraints', []):
                self._apply_constraint_to_km(km, constraint)

        return km

    def _apply_constraint_to_km(self, km, constraint):
        """Apply a single constraint to knob manager"""
        constraint_type = constraint[0]

        if constraint_type == 'IN':
            var_name, values = constraint[1], constraint[2]
            # Update variable domain in knob manager
            if hasattr(km, 'solver') and hasattr(km.solver, 'vals'):
                for key in km.solver.vals.keys():
                    if var_name in str(key):
                        km.solver.vals[key] = values
                        break

        elif constraint_type == 'EQ':
            var_name, value = constraint[1], constraint[2]
            # Set variable to specific value
            if hasattr(km, 'solver') and hasattr(km.solver, 'vals'):
                for key in km.solver.vals.keys():
                    if var_name in str(key):
                        km.solver.vals[key] = [value]
                        break

        elif constraint_type == 'LE':
            var_name, limit = constraint[1], constraint[2]
            # Add upper bound constraint
            # This requires more complex handling in actual CSP solver
            pass

    def optimize(self, env, population, stat, s_time):
        """Main optimization loop - replaces CGA"""

        # Initialize if first call
        if self.policy_net is None:
            self.initialize(env)

        all_pop = [] + population

        for iteration in range(self.config.iter_walks):
            print(f"\n=== FlexAction Iteration {iteration+1}/{self.config.iter_walks} ===")

            # Extract state
            state = self.extract_state(env, all_pop)

            # Select Lambda actions
            action_indices = self.select_actions(state, num_actions=3)
            action_names = [list(self.lambda_library.keys())[idx] for idx in action_indices]
            print(f"Selected Lambda actions: {action_names}")

            # Apply Lambda actions to get modified CSP
            modified_km = self.apply_lambda_actions(env, action_indices)

            # Sample from modified CSP
            new_samples = self.sample_from_modified_csp(env, modified_km, self.config.pop_num)

            # Predict performance
            if new_samples:
                perfs = self.predict(new_samples)
                for idx, sample in enumerate(new_samples):
                    if idx < len(perfs):
                        sample.predict = perfs[idx]

                # Calculate reward
                best_new = max(new_samples, key=lambda x: x.predict)
                if all_pop:
                    best_old = max(all_pop, key=lambda x: x.predict)
                    reward = (best_new.predict - best_old.predict) / (best_old.predict + 1e-8)
                else:
                    reward = best_new.predict / 100.0

                print(f"Best new performance: {best_new.predict:.2f}, Reward: {reward:.3f}")

                # Update Lambda usage stats
                for idx in action_indices:
                    action_name = list(self.lambda_library.keys())[idx]
                    stats = self.lambda_usage_stats[action_name]
                    stats['total_reward'] += reward
                    stats['avg_reward'] = stats['total_reward'] / stats['count']

                # Store experience
                next_state = self.extract_state(env, all_pop + new_samples)
                self.replay_buffer.append(
                    Experience(state, action_indices, reward, next_state, False)
                )

                # Train policy network
                if len(self.replay_buffer) >= self.batch_size:
                    self.train_step()

                # Update target network
                if iteration % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                all_pop.extend(new_samples)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Record statistics
            self.recordStat(all_pop, env, s_time, stat)

        # Remove invalid samples
        for sample in all_pop:
            if not sample.valid:
                sample.predict = 0.0

        return population + new_samples if new_samples else population, all_pop

    def sample_from_modified_csp(self, env, modified_km, num_samples):
        """Sample from modified CSP using Heron's infrastructure"""
        # Temporarily replace task's knob manager
        original_km = env.task.knob_manager
        env.task.knob_manager = modified_km

        # Use Heron's constrained random sampling
        samples = self.constrained_random_sample(env, num_samples)

        # Restore original knob manager
        env.task.knob_manager = original_km

        return samples

    def train_step(self):
        """Train the policy network"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        experiences = [self.replay_buffer[i] for i in batch]

        states = torch.FloatTensor([e.state for e in experiences]).cuda()
        actions = torch.LongTensor([e.action[0] if isinstance(e.action, list) else e.action
                                   for e in experiences]).cuda()
        rewards = torch.FloatTensor([e.reward for e in experiences]).cuda()
        next_states = torch.FloatTensor([e.next_state for e in experiences]).cuda()

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q

        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def recordStat(self, all_pop, env, start_time, stat):
        """Record statistics"""
        if all_pop:
            valid_samples = [s for s in all_pop if s.valid and s.predict > 0]
            if valid_samples:
                best = max(valid_samples, key=lambda x: x.predict)
                avg_perf = np.mean([s.predict for s in valid_samples])
                print(f"  Stats: best={best.predict:.2f}, avg={avg_perf:.2f}, "
                      f"valid={len(valid_samples)}/{len(all_pop)}, epsilon={self.epsilon:.3f}")

    def print_lambda_stats(self):
        """Print Lambda usage statistics"""
        print("\n=== Lambda Usage Statistics ===")
        sorted_items = sorted(self.lambda_usage_stats.items(),
                            key=lambda x: x[1]['avg_reward'], reverse=True)
        for name, stats in sorted_items:
            if stats['count'] > 0:
                print(f"{name:20s}: count={stats['count']:3d}, avg_reward={stats['avg_reward']:+.3f}")


def register_flexaction_to_heron():
    """Register FlexAction as an optimization method in Heron"""
    # Import and modify Heron's environment
    import Heron.environment as heron_env

    # Add FlexAction to the createTask method
    original_create_task = heron_env.Env.createTask

    def create_task_with_flexaction(self, name, opfunc, args, target,
                                   target_host=None, dump_const_desc=False):
        # Call original method
        result = original_create_task(self, name, opfunc, args, target,
                                    target_host, dump_const_desc)

        # Check if FlexAction is selected
        if hasattr(self.config, 'opt_method') and self.config.opt_method == 'FLEXACTION':
            self.tuner = RealFlexActionTuner(self.config)
            if self.config.use_cost_model:
                self.tuner.buildCostModel(self.task)
            print("✓ FlexAction tuner registered")

        return result

    # Replace the method
    heron_env.Env.createTask = create_task_with_flexaction
    print("✓ FlexAction registered to Heron environment")


if __name__ == "__main__":
    print("Real FlexAction Integration Module")
    print("=" * 60)
    register_flexaction_to_heron()
    print("Module loaded successfully!")