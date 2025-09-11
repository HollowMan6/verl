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
Simple test runner for the argument classification system.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --scenarios        # Run scenario tests only
    python run_tests.py --unit            # Run unit tests only
    python run_tests.py --verbose         # Run with verbose output
"""

import argparse
import json
import os
import sys
from unittest.mock import patch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add the reward_score directory to the path for direct rlla import
reward_score_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if reward_score_dir not in sys.path:
    sys.path.insert(0, reward_score_dir)

try:
    from rlla import compute_tool_call_reward
    from test_argument_classification import run_test_suite
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the correct directory.")
    sys.exit(1)


def load_test_scenarios():
    """Load test scenarios from JSON file."""
    scenarios_file = os.path.join(os.path.dirname(__file__), "test_scenarios.json")
    try:
        with open(scenarios_file) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Test scenarios file not found: {scenarios_file}")
        return None


def run_scenario_tests(verbose=False):
    """Run scenario-based tests."""
    print("Running Scenario Tests")
    print("=" * 40)

    scenarios_data = load_test_scenarios()
    if not scenarios_data:
        print("No scenario data available.")
        return False

    total_tests = 0
    passed_tests = 0

    # Create a simple tokenizer for testing
    class SimpleTokenizer:
        def encode(self, text):
            return text.split()

        def decode(self, tokens):
            return " ".join(tokens) if isinstance(tokens, list) else str(tokens)

    tokenizer = SimpleTokenizer()

    with patch("rlla.load_argument_config") as mock_config:
        # Mock configuration
        mock_config.return_value = {
            "parameter_types": {
                "exact_match": {
                    "type": 1,
                    "arguments": ["index", "url", "tabId", "duration", "num_clicks", "amount", "nodeId"],
                },
                "coordinate": {"type": 2, "arguments": ["x", "y"], "tolerance": 20},
                "semantic_similarity": {
                    "type": 3,
                    "arguments": [
                        "userSidePrompt",
                        "text",
                        "option",
                        "prompt",
                        "name",
                        "value",
                        "title",
                        "progress",
                        "next_step",
                    ],
                    "threshold": 0.5,
                },
                "enum_boolean": {
                    "type": 4,
                    "arguments": [
                        "button",
                        "direction",
                        "enter",
                        "interactType",
                        "operation",
                        "extract_page_content",
                        "helpType",
                        "selectMultiple",
                    ],
                    "enum_values": {
                        "button": ["left", "right", "middle"],
                        "direction": ["up", "down"],
                        "enter": [True, False],
                        "interactType": ["confirm", "input", "select", "request_help"],
                        "operation": ["read_variable", "write_variable", "list_all_variable"],
                        "extract_page_content": [True, False],
                        "helpType": ["request_login", "request_assistance"],
                        "selectMultiple": [True, False],
                    },
                },
                "array": {"type": 5, "arguments": ["file_paths", "files", "selectOptions"]},
            },
            "tool_specifications": {
                "navigate_to": {"url": "exact_match", "userSidePrompt": "semantic_similarity"},
                "click_element": {
                    "index": "exact_match",
                    "button": "enum_boolean",
                    "num_clicks": "exact_match",
                    "userSidePrompt": "semantic_similarity",
                },
                "input_text": {
                    "text": "semantic_similarity",
                    "enter": "enum_boolean",
                    "index": "exact_match",
                    "userSidePrompt": "semantic_similarity",
                },
                "click_by_point": {
                    "x": "coordinate",
                    "y": "coordinate",
                    "button": "enum_boolean",
                    "num_clicks": "exact_match",
                    "userSidePrompt": "semantic_similarity",
                },
                "input_by_point": {
                    "x": "coordinate",
                    "y": "coordinate",
                    "text": "semantic_similarity",
                    "enter": "enum_boolean",
                    "userSidePrompt": "semantic_similarity",
                },
                "human_interact": {
                    "interactType": "enum_boolean",
                    "prompt": "semantic_similarity",
                    "helpType": "enum_boolean",
                    "selectOptions": "array",
                    "selectMultiple": "enum_boolean",
                },
                "click_upload_file": {
                    "index": "exact_match",
                    "file_paths": "array",
                    "userSidePrompt": "semantic_similarity",
                },
                "scroll_mouse_wheel": {
                    "amount": "exact_match",
                    "direction": "enum_boolean",
                    "extract_page_content": "enum_boolean",
                    "userSidePrompt": "semantic_similarity",
                },
                "variable_storage": {
                    "operation": "enum_boolean",
                    "name": "semantic_similarity",
                    "value": "semantic_similarity",
                },
                "foreach_task": {
                    "nodeId": "exact_match",
                    "progress": "semantic_similarity",
                    "next_step": "semantic_similarity",
                },
            },
        }

        for scenario in scenarios_data["test_scenarios"]:
            print(f"\nTesting scenario: {scenario['name']}")
            if verbose:
                print(f"Description: {scenario['description']}")

            gt_tools = scenario["ground_truth"]

            for prediction in scenario["predictions"]:
                total_tests += 1
                pred_tools = prediction["tools"]
                expected_range = prediction.get("expected_score", 0.5)

                try:
                    # Clear environment variables for clean test
                    with patch.dict(os.environ, {}, clear=True):
                        reward = compute_tool_call_reward(gt_tools, pred_tools, 3.0, -3.0, tokenizer)

                    # Normalize reward to 0-1 scale for comparison
                    normalized_reward = (reward + 3.0) / 6.0

                    # Check if result is reasonable (within expected range ±0.3)
                    if isinstance(expected_range, int | float):
                        if abs(normalized_reward - expected_range) <= 0.3:
                            passed_tests += 1
                            if verbose:
                                print(
                                    f"  ✅ {prediction['name']}: {normalized_reward:.3f} (expected ~{expected_range})"
                                )
                        else:
                            if verbose:
                                print(
                                    f"  ❌ {prediction['name']}: {normalized_reward:.3f} (expected ~{expected_range})"
                                )
                    else:
                        # Just check that we got a reasonable result
                        if -3.0 <= reward <= 3.0:
                            passed_tests += 1
                            if verbose:
                                print(f"  ✅ {prediction['name']}: {normalized_reward:.3f}")
                        else:
                            if verbose:
                                print(f"  ❌ {prediction['name']}: {normalized_reward:.3f} (out of range)")

                except Exception as e:
                    if verbose:
                        print(f"  ❌ {prediction['name']}: Error - {e}")

    print("\nScenario Tests Summary:")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests / total_tests * 100:.1f}%")

    return passed_tests == total_tests


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test the argument classification system")
    parser.add_argument("--scenarios", action="store_true", help="Run scenario tests only")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    success = True

    if not args.scenarios and not args.unit:
        # Run both by default
        print("Running All Tests")
        print("=" * 60)

        # Run unit tests
        print("\n1. Unit Tests")
        print("-" * 30)
        unit_result = run_test_suite()
        unit_success = unit_result.wasSuccessful()

        # Run scenario tests
        print("\n2. Scenario Tests")
        print("-" * 30)
        scenario_success = run_scenario_tests(args.verbose)

        success = unit_success and scenario_success

        print("\n" + "=" * 60)
        print("Overall Test Summary:")
        print(f"Unit Tests: {'✅ PASSED' if unit_success else '❌ FAILED'}")
        print(f"Scenario Tests: {'✅ PASSED' if scenario_success else '❌ FAILED'}")

    elif args.unit:
        print("Running Unit Tests Only")
        print("=" * 40)
        result = run_test_suite()
        success = result.wasSuccessful()

    elif args.scenarios:
        print("Running Scenario Tests Only")
        print("=" * 40)
        success = run_scenario_tests(args.verbose)

    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
