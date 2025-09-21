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
Standalone demonstration of the argument classification system concepts.

This script shows the core concepts without requiring external dependencies.
"""


def sqrt(x):
    """Simple square root implementation to avoid math module issues."""
    if x < 0:
        return 0
    if x == 0:
        return 0

    # Newton's method for square root
    guess = x / 2
    for _ in range(10):  # Sufficient iterations for our precision needs
        guess = (guess + x / guess) / 2
    return guess


def exact_match_reward(gt_value, pred_value):
    """Calculate reward for exact match arguments."""
    return 1.0 if gt_value == pred_value else 0.0


def coordinate_reward(gt_coord, pred_coord, tolerance):
    """Calculate reward for coordinate arguments with tolerance."""
    distance = abs(gt_coord - pred_coord)
    if distance <= tolerance:
        return max(0.0, 1.0 - (distance / tolerance))
    return 0.0


def semantic_reward(gt_text, pred_text, threshold):
    """Calculate reward for semantic similarity (mock implementation)."""
    # Simple mock: check for word overlap
    gt_words = set(gt_text.lower().split())
    pred_words = set(pred_text.lower().split())

    if not gt_words and not pred_words:
        return 1.0
    if not gt_words or not pred_words:
        return 0.0

    overlap = len(gt_words.intersection(pred_words))
    union = len(gt_words.union(pred_words))
    similarity = overlap / union if union > 0 else 0.0

    return similarity if similarity >= threshold else 0.0


def enum_reward(gt_value, pred_value, enum_values, param_name):
    """Calculate reward for enum/boolean arguments."""
    if param_name in enum_values:
        # Enum parameter
        valid_values = enum_values[param_name]
        if pred_value in valid_values:
            return 1.0 if gt_value == pred_value else 0.0
        return 0.0  # Invalid enum value
    else:
        # Boolean parameter
        return 1.0 if gt_value == pred_value else 0.0


def array_reward(gt_array, pred_array):
    """Calculate reward for array arguments using Jaccard similarity."""
    if not gt_array and not pred_array:
        return 1.0
    if not gt_array or not pred_array:
        return 0.0

    gt_set = set(gt_array)
    pred_set = set(pred_array)
    intersection = len(gt_set.intersection(pred_set))
    union = len(gt_set.union(pred_set))

    return intersection / union if union > 0 else 0.0


def demo_parameter_types():
    """Demonstrate parameter type scoring without external dependencies."""
    print("Standalone Parameter Type Demo")
    print("=" * 45)

    print("\n1. Exact Match Scoring:")
    print(f"   Same URL: {exact_match_reward('https://example.com', 'https://example.com'):.3f}")
    print(f"   Different URL: {exact_match_reward('https://example.com', 'https://other.com'):.3f}")
    print(f"   Same index: {exact_match_reward(5, 5):.3f}")
    print(f"   Different index: {exact_match_reward(5, 3):.3f}")

    print("\n2. Coordinate Scoring (tolerance=20px):")
    print(f"   Same point (100,100): {coordinate_reward(100, 100, 20):.3f}")
    print(f"   Close point (100,110): {coordinate_reward(100, 110, 20):.3f}")
    print(f"   Edge of tolerance (100,120): {coordinate_reward(100, 120, 20):.3f}")
    print(f"   Outside tolerance (100,130): {coordinate_reward(100, 130, 20):.3f}")

    print("\n3. Semantic Similarity (threshold=0.5):")
    print(f"   Same text: {semantic_reward('click login button', 'click login button', 0.5):.3f}")
    print(f"   Similar text: {semantic_reward('click login button', 'click the login button', 0.5):.3f}")
    print(f"   Related text: {semantic_reward('click login button', 'press login', 0.5):.3f}")
    print(f"   Different text: {semantic_reward('click login button', 'scroll down page', 0.5):.3f}")

    print("\n4. Enum/Boolean Scoring:")
    enum_values = {"button": ["left", "right", "middle"]}
    print(f"   Valid enum match: {enum_reward('left', 'left', enum_values, 'button'):.3f}")
    print(f"   Valid enum different: {enum_reward('left', 'right', enum_values, 'button'):.3f}")
    print(f"   Invalid enum value: {enum_reward('left', 'invalid', enum_values, 'button'):.3f}")
    print(f"   Boolean match: {enum_reward(True, True, {}, 'flag'):.3f}")
    print(f"   Boolean different: {enum_reward(True, False, {}, 'flag'):.3f}")

    print("\n5. Array Scoring (Jaccard similarity):")
    print(f"   Identical arrays: {array_reward(['file1.txt', 'file2.txt'], ['file1.txt', 'file2.txt']):.3f}")
    print(f"   Partial overlap: {array_reward(['file1.txt', 'file2.txt'], ['file1.txt', 'file3.txt']):.3f}")
    print(f"   No overlap: {array_reward(['file1.txt', 'file2.txt'], ['file3.txt', 'file4.txt']):.3f}")
    print(f"   Subset match: {array_reward(['file1.txt', 'file2.txt', 'file3.txt'], ['file1.txt', 'file2.txt']):.3f}")


def compute_coordinate_pair_score(gt_x, gt_y, pred_x, pred_y, tolerance):
    """Compute coordinate pair score using Euclidean distance."""
    distance = sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
    if distance <= tolerance:
        return max(0.0, 1.0 - (distance / tolerance))
    return 0.0


def demo_tool_scenarios():
    """Demonstrate realistic tool call scenarios."""
    print("\n\nRealistic Tool Call Scenarios")
    print("=" * 45)

    # Configuration
    config = {
        "parameter_types": {
            "exact_match": {"type": 1, "arguments": ["index", "url"]},
            "coordinate": {"type": 2, "arguments": ["x", "y"], "tolerance": 20},
            "semantic_similarity": {"type": 3, "arguments": ["userSidePrompt", "text"], "threshold": 0.6},
            "enum_boolean": {
                "type": 4,
                "arguments": ["button", "enter"],
                "enum_values": {"button": ["left", "right"], "enter": [True, False]},
            },
        },
        "tool_specifications": {
            "navigate_to": {"url": "exact_match", "userSidePrompt": "semantic_similarity"},
            "click_element": {
                "index": "exact_match",
                "button": "enum_boolean",
                "userSidePrompt": "semantic_similarity",
            },
            "click_by_point": {"x": "coordinate", "y": "coordinate", "button": "enum_boolean"},
            "input_text": {"text": "semantic_similarity", "enter": "enum_boolean"},
        },
    }

    scenarios = [
        {
            "name": "Perfect Navigation Match",
            "gt": {
                "name": "navigate_to",
                "arguments": {"url": "https://example.com", "userSidePrompt": "Go to example website"},
            },
            "pred": {
                "name": "navigate_to",
                "arguments": {"url": "https://example.com", "userSidePrompt": "Go to example website"},
            },
            "expected": 1.0,
        },
        {
            "name": "URL Mismatch",
            "gt": {
                "name": "navigate_to",
                "arguments": {"url": "https://example.com", "userSidePrompt": "Go to example website"},
            },
            "pred": {
                "name": "navigate_to",
                "arguments": {"url": "https://different.com", "userSidePrompt": "Go to example website"},
            },
            "expected": 0.5,  # Half credit for semantic match
        },
        {
            "name": "Close Coordinate Click",
            "gt": {"name": "click_by_point", "arguments": {"x": 100, "y": 200, "button": "left"}},
            "pred": {"name": "click_by_point", "arguments": {"x": 105, "y": 195, "button": "left"}},
            "expected": 0.75,  # Close coordinates
        },
        {
            "name": "Wrong Tool Type",
            "gt": {"name": "navigate_to", "arguments": {"url": "https://example.com", "userSidePrompt": "Go to site"}},
            "pred": {
                "name": "click_element",
                "arguments": {"index": 0, "button": "left", "userSidePrompt": "Click something"},
            },
            "expected": 0.0,  # Wrong tool
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}:")
        print(f"   Ground Truth: {scenario['gt']}")
        print(f"   Prediction:   {scenario['pred']}")

        # Simple scoring logic
        gt_tool = scenario["gt"]
        pred_tool = scenario["pred"]

        if gt_tool["name"] != pred_tool["name"]:
            score = 0.0
        else:
            tool_spec = config["tool_specifications"][gt_tool["name"]]
            param_scores = []

            for param, param_type in tool_spec.items():
                if param in gt_tool["arguments"] and param in pred_tool["arguments"]:
                    gt_val = gt_tool["arguments"][param]
                    pred_val = pred_tool["arguments"][param]

                    if param_type == "exact_match":
                        param_score = exact_match_reward(gt_val, pred_val)
                    elif param_type == "coordinate":
                        tolerance = config["parameter_types"]["coordinate"]["tolerance"]
                        param_score = coordinate_reward(gt_val, pred_val, tolerance)
                    elif param_type == "semantic_similarity":
                        threshold = config["parameter_types"]["semantic_similarity"]["threshold"]
                        param_score = semantic_reward(gt_val, pred_val, threshold)
                    elif param_type == "enum_boolean":
                        enum_values = config["parameter_types"]["enum_boolean"].get("enum_values", {})
                        param_score = enum_reward(gt_val, pred_val, enum_values, param)
                    else:
                        param_score = 0.0

                    param_scores.append(param_score)

            score = sum(param_scores) / len(param_scores) if param_scores else 0.0

        print(f"   Computed Score: {score:.3f}")
        print(f"   Expected: {scenario['expected']:.3f}")
        print(f"   Match: {'✅' if abs(score - scenario['expected']) < 0.2 else '❌'}")


def demo_coordinate_pairs():
    """Demonstrate coordinate pair scoring."""
    print("\n\nCoordinate Pair Scoring Demo")
    print("=" * 45)

    test_cases = [
        {"name": "Perfect match", "gt": (100, 200), "pred": (100, 200), "tolerance": 20},
        {"name": "Close match", "gt": (100, 200), "pred": (105, 195), "tolerance": 20},
        {"name": "Edge of tolerance", "gt": (100, 200), "pred": (114, 186), "tolerance": 20},
        {"name": "Outside tolerance", "gt": (100, 200), "pred": (130, 230), "tolerance": 20},
    ]

    for case in test_cases:
        score = compute_coordinate_pair_score(
            case["gt"][0], case["gt"][1], case["pred"][0], case["pred"][1], case["tolerance"]
        )
        distance = sqrt((case["gt"][0] - case["pred"][0]) ** 2 + (case["gt"][1] - case["pred"][1]) ** 2)
        print(f"   {case['name']}: GT{case['gt']} -> Pred{case['pred']} = {score:.3f} (distance: {distance:.1f}px)")


if __name__ == "__main__":
    print("🎯 Standalone Argument Classification Demo")
    print("   (No external dependencies required)")
    print("=" * 60)

    try:
        demo_parameter_types()
        demo_tool_scenarios()
        demo_coordinate_pairs()

        print("\n" + "=" * 60)
        print("✅ Standalone demo completed successfully!")
        print("\nKey Concepts Demonstrated:")
        print("• 5 parameter types with different scoring methods")
        print("• Realistic browser automation tool scenarios")
        print("• Coordinate tolerance and distance calculations")
        print("• Semantic similarity using word overlap")
        print("• Enum validation and boolean matching")
        print("• Array comparison with Jaccard similarity")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
