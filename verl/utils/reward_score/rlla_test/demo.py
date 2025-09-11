#!/usr/bin/env python3

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Demonstration script for the argument classification system.

This script shows how the system evaluates tool call predictions against ground truth.
"""

import os
import sys
from unittest.mock import patch

# Add current directory to path to ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add the reward_score directory to the path for direct rlla import
reward_score_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if reward_score_dir not in sys.path:
    sys.path.insert(0, reward_score_dir)


def demo_basic_usage():
    """Demonstrate basic usage of the argument classification system."""
    print("Argument Classification System Demo")
    print("=" * 50)

    # Try to import the module and handle import errors gracefully
    try:
        # Mock the necessary dependencies first
        with patch("rlla.load_argument_config") as mock_config, patch("rlla.cosine_similarity_text") as mock_similarity:
            # Import after patching to avoid dependency issues
            from rlla import compute_tool_call_reward

            # Set up configuration
            mock_config.return_value = {
                "parameter_types": {
                    "exact_match": {"type": 1, "arguments": ["index", "url"]},
                    "coordinate": {"type": 2, "arguments": ["x", "y"], "tolerance": 20},
                    "semantic_similarity": {"type": 3, "arguments": ["text", "userSidePrompt"], "threshold": 0.8},
                    "enum_boolean": {
                        "type": 4,
                        "arguments": ["button", "enter"],
                        "enum_values": {"button": ["left", "right"], "enter": [True, False]},
                    },
                    "array": {"type": 5, "arguments": ["file_paths"]},
                },
                "tool_specifications": {
                    "navigate_to": {"url": "exact_match", "userSidePrompt": "semantic_similarity"},
                    "click_element": {
                        "index": "exact_match",
                        "button": "enum_boolean",
                        "userSidePrompt": "semantic_similarity",
                    },
                    "input_text": {"text": "semantic_similarity", "enter": "enum_boolean"},
                    "click_by_point": {"x": "coordinate", "y": "coordinate", "button": "enum_boolean"},
                    "click_upload_file": {"index": "exact_match", "file_paths": "array"},
                },
            }

            _run_demo_examples(compute_tool_call_reward, mock_similarity)

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nThis appears to be a dependency issue. Common solutions:")
        print("1. Install/reinstall numpy: pip install --upgrade numpy")
        print("2. Check for conflicting modules in your environment")
        print("3. Create a fresh virtual environment")
        print("4. Try running: pip install --force-reinstall numpy")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print("Please check your Python environment and dependencies.")
        return False

    return True


def _run_demo_examples(compute_tool_call_reward, mock_similarity):
    print("\n1. Perfect Match Example")
    print("-" * 30)

    gt_tools = [
        {"name": "navigate_to", "arguments": {"url": "https://example.com", "userSidePrompt": "Go to example website"}}
    ]

    pred_tools = [
        {"name": "navigate_to", "arguments": {"url": "https://example.com", "userSidePrompt": "Go to example website"}}
    ]

    mock_similarity.return_value = 1.0  # Perfect semantic match
    reward = compute_tool_call_reward(gt_tools, pred_tools, 3.0, -3.0)
    print(f"Ground Truth: {gt_tools[0]}")
    print(f"Prediction:   {pred_tools[0]}")
    print(f"Reward Score: {reward:.3f}")

    print("\n2. Partial Match Example")
    print("-" * 30)

    pred_tools_partial = [
        {
            "name": "navigate_to",
            "arguments": {"url": "https://different.com", "userSidePrompt": "Visit different site"},
        }
    ]

    mock_similarity.return_value = 0.7  # Good semantic similarity
    reward = compute_tool_call_reward(gt_tools, pred_tools_partial, 3.0, -3.0)
    print(f"Ground Truth: {gt_tools[0]}")
    print(f"Prediction:   {pred_tools_partial[0]}")
    print(f"Reward Score: {reward:.3f}")

    print("\n3. Coordinate Matching Example")
    print("-" * 30)

    gt_coords = [{"name": "click_by_point", "arguments": {"x": 100, "y": 200, "button": "left"}}]
    pred_coords_close = [{"name": "click_by_point", "arguments": {"x": 105, "y": 195, "button": "left"}}]
    pred_coords_far = [{"name": "click_by_point", "arguments": {"x": 150, "y": 250, "button": "left"}}]

    reward_close = compute_tool_call_reward(gt_coords, pred_coords_close, 3.0, -3.0)
    reward_far = compute_tool_call_reward(gt_coords, pred_coords_far, 3.0, -3.0)

    print(f"Ground Truth: {gt_coords[0]}")
    print(f"Close Prediction (within tolerance): {pred_coords_close[0]}")
    print(f"Close Reward: {reward_close:.3f}")
    print(f"Far Prediction (outside tolerance): {pred_coords_far[0]}")
    print(f"Far Reward: {reward_far:.3f}")

    print("\n4. Multiple Tools Example")
    print("-" * 30)

    gt_multi = [
        {"name": "navigate_to", "arguments": {"url": "https://example.com", "userSidePrompt": "Go to site"}},
        {"name": "click_element", "arguments": {"index": 0, "button": "left", "userSidePrompt": "Click login button"}},
    ]

    pred_multi = [
        {"name": "navigate_to", "arguments": {"url": "https://example.com", "userSidePrompt": "Navigate to site"}},
        {
            "name": "click_element",
            "arguments": {"index": 0, "button": "left", "userSidePrompt": "Click the login button"},
        },
    ]

    mock_similarity.return_value = 0.9  # High semantic similarity
    reward = compute_tool_call_reward(gt_multi, pred_multi, 3.0, -3.0)
    print(f"Ground Truth: {len(gt_multi)} tools")
    print(f"Prediction:   {len(pred_multi)} tools")
    print(f"Combined Reward: {reward:.3f}")

    print("\n5. Wrong Tool Example")
    print("-" * 30)

    gt_wrong = [{"name": "navigate_to", "arguments": {"url": "https://example.com", "userSidePrompt": "Go to site"}}]
    pred_wrong = [
        {"name": "click_element", "arguments": {"index": 0, "button": "left", "userSidePrompt": "Click something"}}
    ]

    reward = compute_tool_call_reward(gt_wrong, pred_wrong, 3.0, -3.0)
    print(f"Ground Truth: {gt_wrong[0]['name']}")
    print(f"Prediction:   {pred_wrong[0]['name']}")
    print(f"Reward Score: {reward:.3f}")

    print("\n" + "=" * 50)
    print("Demo Complete!")
    print("\nKey Insights:")
    print("• Perfect matches receive maximum reward")
    print("• Partial matches receive proportional rewards based on parameter types")
    print("• Coordinate arguments use tolerance-based scoring")
    print("• Semantic arguments use cosine similarity")
    print("• Wrong tools receive penalty scores")
    print("• Multiple tools are scored independently and combined")


def demo_parameter_types():
    """Demonstrate different parameter type scoring."""
    print("\nParameter Type Scoring Demo")
    print("=" * 40)

    try:
        with patch("rlla.cosine_similarity_text") as mock_similarity:
            # Import functions
            from rlla import array_reward, coordinate_reward, enum_reward, exact_match_reward, semantic_reward

            print("\n1. Exact Match Scoring:")
            print(f"   Same value: {exact_match_reward('test', 'test'):.3f}")
            print(f"   Different:  {exact_match_reward('test', 'other'):.3f}")

            print("\n2. Coordinate Scoring (tolerance=20):")
            print(f"   Same point:    {coordinate_reward(100, 100, 100, 100, 20):.3f}")
            print(f"   Close (10px):  {coordinate_reward(110, 110, 100, 100, 20):.3f}")
            print(f"   Far (30px):    {coordinate_reward(130, 130, 100, 100, 20):.3f}")

            print("\n3. Semantic Similarity:")
            mock_similarity.return_value = 1.0
            print(f"   Perfect match: {semantic_reward('hello world', 'hello world', 0.8):.3f}")
            mock_similarity.return_value = 0.9
            print(f"   High similarity: {semantic_reward('hello world', 'greetings world', 0.8):.3f}")
            mock_similarity.return_value = 0.6
            print(f"   Low similarity: {semantic_reward('hello world', 'goodbye earth', 0.8):.3f}")

            print("\n4. Enum/Boolean Scoring:")
            print(f"   Valid enum match:   {enum_reward('left', 'left', ['left', 'right', 'middle']):.3f}")
            print(f"   Valid enum different: {enum_reward('right', 'left', ['left', 'right', 'middle']):.3f}")
            print(f"   Invalid enum value: {enum_reward('invalid', 'left', ['left', 'right', 'middle']):.3f}")
            print(f"   Boolean match: {enum_reward(True, True, [True, False]):.3f}")
            print(f"   Boolean different:{enum_reward(False, True, [True, False]):.3f}")

            print("\n5. Array Scoring:")
            print(f"   Same arrays:    {array_reward(['a', 'b'], ['a', 'b']):.3f}")
            print(f"   Partial match:  {array_reward(['a', 'b', 'c'], ['a', 'b']):.3f}")
            print(f"   No match:       {array_reward(['a', 'b'], ['c', 'd']):.3f}")

            return True
    except ImportError as e:
        print(f"❌ Import Error in parameter type demo: {e}")
        print("Skipping parameter type demonstrations due to dependency issues.")
        return False
    except Exception as e:
        print(f"❌ Error in parameter type demo: {e}")
        return False


if __name__ == "__main__":
    print("🔧 Checking dependencies and running demos...")

    success1 = demo_basic_usage()
    success2 = demo_parameter_types()

    if success1 and success2:
        print("\n✅ All demos completed successfully!")
    elif success1:
        print("\n⚠️  Basic demo completed, but parameter demo failed due to dependencies.")
    elif success2:
        print("\n⚠️  Parameter demo completed, but basic demo failed due to dependencies.")
    else:
        print("\n❌ Both demos failed. Please check your environment:")
        print("   1. pip install --upgrade numpy")
        print("   2. pip install --force-reinstall numpy")
        print("   3. Check for conflicting modules")
        print("   4. Consider creating a fresh virtual environment")
