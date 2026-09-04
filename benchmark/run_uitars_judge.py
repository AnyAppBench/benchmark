"""
Test script for judge-assisted UI-TARS agent on Android World tasks.

This script runs single tasks with the judge-assisted agent for testing
and validation before full benchmark runs.
"""

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from agents.uitars.adapters.android_world_judge import JudgeAssistedUITARSAgent
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        'Missing local adapter module: agents.uitars.adapters.android_world_judge. '
        'Place copied agent code under <project_root>/agents/... or install '
        'a package that provides this module.'
    ) from exc
from benchmarks.online.android_world.android_world.suite_utils import (
    load_and_initialize_task_suite,
)
from benchmarks.online.android_world.android_world import registry


def main():
    parser = argparse.ArgumentParser(description="Run judge-assisted UI-TARS on a single Android World task")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., ContactsAddContact)")
    parser.add_argument("--agent_device", type=str, default="cuda:0", help="Device for agent model")
    parser.add_argument("--judge_device", type=str, default="cuda:1", help="Device for judge model")
    parser.add_argument("--debug_screenshots", action="store_true", help="Save debug screenshots")
    parser.add_argument("--max_steps", type=int, default=15, help="Maximum steps per task")
    parser.add_argument("--enable_judge", action="store_true", default=True, help="Enable judge evaluation")
    
    args = parser.parse_args()
    
    # Load task
    print(f"\n{'='*60}")
    print(f"Testing Judge-Assisted UI-TARS on: {args.task}")
    print(f"Agent device: {args.agent_device}")
    print(f"Judge device: {args.judge_device}")
    print(f"Judge enabled: {args.enable_judge}")
    print(f"{'='*60}\n")
    
    # Get task class
    task_registry = registry.TaskRegistry()
    task_cls = task_registry.get_task(args.task)
    
    if not task_cls:
        print(f"Error: Task '{args.task}' not found in registry")
        sys.exit(1)
    
    # Load environment and task suite
    env, task_suite = load_and_initialize_task_suite(
        suite_family="android_world",
        task_names=[args.task],
        max_steps_per_task=args.max_steps,
    )
    
    # Create judge-assisted agent
    agent = JudgeAssistedUITARSAgent(
        env=env,
        agent_device=args.agent_device,
        judge_device=args.judge_device,
        debug_save_screenshots=args.debug_screenshots,
        enable_judge=args.enable_judge,
    )
    
    # Run task
    print(f"\nRunning task: {args.task}")
    print("-" * 60)
    
    task = task_suite[0]
    task.setup(env)
    
    done = False
    step = 0
    max_steps = args.max_steps
    
    while not done and step < max_steps:
        step += 1
        result = agent.step(task.goal)
        done = result.done
        
        if done:
            print(f"\nAgent signaled completion at step {step}")
            break
    
    # Evaluate
    success = task.is_successful(env)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Steps: {step}/{max_steps}")
    print(f"Success: {'✓' if success else '✗'}")
    
    # Print judge statistics
    if args.enable_judge:
        stats = agent.get_stats()
        print(f"\nJudge evaluations: {stats['total_evaluations']}")
        print(f"Failures detected: {stats['failures_detected']}")
        print(f"Average score: {stats['avg_score']:.2f}/5.0")
    
    print("="*60 + "\n")
    
    # Cleanup
    agent.cleanup()
    task.teardown(env)
    env.close()


if __name__ == "__main__":
    main()

