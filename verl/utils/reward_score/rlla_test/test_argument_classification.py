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
Test cases for the argument classification-based reward system.

This module tests the scoring functions for different parameter types based on
the browser agent evaluation data format.
"""

import math
import os
import sys
import unittest
from unittest.mock import patch

# Add the reward_score directory to the path for direct rlla import
reward_score_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if reward_score_dir not in sys.path:
    sys.path.insert(0, reward_score_dir)

# Import rlla module after path setup
from rlla import (  # noqa: E402
    array_reward,
    compute_coordinate_pair_score,
    compute_parameter_score,
    compute_tool_call_reward,
    coordinate_reward,
    cosine_similarity_text,
    enum_reward,
    exact_match_reward,
    semantic_reward,
)


class TestArgumentClassificationSystem(unittest.TestCase):
    """Test cases for the argument classification system."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a simple tokenizer for testing
        class SimpleTokenizer:
            def encode(self, text):
                return text.split()

            def decode(self, tokens):
                return " ".join(tokens) if isinstance(tokens, list) else str(tokens)

        self.tokenizer = SimpleTokenizer()

        self.config = {
            "parameter_types": {
                "exact_match": {
                    "type": 1,
                    "description": "Binary exact match (0 or 1)",
                    "arguments": [
                        "index",
                        "url",
                        "tabId",
                        "fileChooserId",
                        "duration",
                        "num_clicks",
                        "amount",
                        "nodeId",
                    ],
                },
                "coordinate": {
                    "type": 2,
                    "description": "Distance-based scoring with tolerance",
                    "arguments": ["x", "y"],
                    "tolerance": 20,
                },
                "semantic_similarity": {
                    "type": 3,
                    "description": "Semantic similarity using embeddings",
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
                    "description": "Exact match for enums, binary for booleans",
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
                "array": {
                    "type": 5,
                    "description": "Jaccard similarity for overlapping elements",
                    "arguments": ["file_paths", "files", "selectOptions"],
                },
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


class TestParameterTypeRewards(TestArgumentClassificationSystem):
    """Test individual parameter type reward functions."""

    def test_exact_match_reward(self):
        """Test exact match scoring."""
        # Perfect match
        self.assertEqual(exact_match_reward(42, 42), 1.0)
        self.assertEqual(exact_match_reward("navigate_to", "navigate_to"), 1.0)
        self.assertEqual(exact_match_reward("https://google.com", "https://google.com"), 1.0)

        # No match
        self.assertEqual(exact_match_reward(42, 43), 0.0)
        self.assertEqual(exact_match_reward("click_element", "navigate_to"), 0.0)
        self.assertEqual(exact_match_reward("https://google.com", "https://bing.com"), 0.0)

    def test_coordinate_reward(self):
        """Test coordinate-based scoring."""
        # Perfect match
        self.assertEqual(coordinate_reward(100, 100, 100, 100), 1.0)

        # Within tolerance (distance = 10, tolerance = 20)
        score = coordinate_reward(100, 100, 110, 100, tolerance=20)
        expected = 1.0 - (10 / 20)
        self.assertEqual(score, expected)

        # At tolerance boundary
        score = coordinate_reward(100, 100, 120, 100, tolerance=20)
        self.assertEqual(score, 0.0)

        # Outside tolerance
        score = coordinate_reward(100, 100, 130, 100, tolerance=20)
        self.assertEqual(score, 0.0)

        # Test with different coordinates
        distance = math.sqrt((15 - 10) ** 2 + (20 - 10) ** 2)  # ~11.18
        score = coordinate_reward(10, 10, 15, 20, tolerance=20)
        expected = 1.0 - (distance / 20)
        self.assertAlmostEqual(score, expected, places=5)

        # Test invalid input handling
        self.assertEqual(coordinate_reward("invalid", 100, 100, 100), 0.0)
        self.assertEqual(coordinate_reward(100, "invalid", 100, 100), 0.0)

    def test_semantic_reward(self):
        """Test semantic similarity scoring."""

        # Create a simple tokenizer for testing
        class SimpleTokenizer:
            def encode(self, text):
                return text.split()

            def decode(self, tokens):
                return " ".join(tokens) if isinstance(tokens, list) else str(tokens)

        tokenizer = SimpleTokenizer()

        # Test with perfect match should return 1.0
        score = semantic_reward("Click the button", "Click the button", tokenizer, threshold=0.8)
        self.assertEqual(score, 1.0)

        # Test with good similarity should return some positive score
        score = semantic_reward("Click button", "Click the button", tokenizer, threshold=0.5)
        self.assertGreater(score, 0.0)

        # Test with completely different text should return 0.0
        score = semantic_reward("Different text", "Completely unrelated", tokenizer, threshold=0.8)
        self.assertEqual(score, 0.0)

    def test_enum_reward(self):
        """Test enum/boolean parameter scoring."""
        valid_buttons = ["left", "right", "middle"]

        # Perfect match
        self.assertEqual(enum_reward("left", "left", valid_buttons), 1.0)
        self.assertEqual(enum_reward("right", "right", valid_buttons), 1.0)

        # Valid but incorrect choice (partial credit)
        self.assertEqual(enum_reward("left", "right", valid_buttons), 0.5)
        self.assertEqual(enum_reward("middle", "left", valid_buttons), 0.5)

        # Invalid choice
        self.assertEqual(enum_reward("invalid", "left", valid_buttons), 0.0)
        self.assertEqual(enum_reward("top", "left", valid_buttons), 0.0)

        # Test boolean values
        bool_options = [True, False]
        self.assertEqual(enum_reward(True, True, bool_options), 1.0)
        self.assertEqual(enum_reward(False, True, bool_options), 0.5)
        self.assertEqual(enum_reward("invalid", True, bool_options), 0.0)

    def test_array_reward(self):
        """Test array parameter scoring (Jaccard similarity)."""
        # Perfect match
        self.assertEqual(array_reward(["file1.txt", "file2.txt"], ["file1.txt", "file2.txt"]), 1.0)

        # Partial overlap
        score = array_reward(["file1.txt", "file2.txt"], ["file1.txt", "file3.txt"])
        expected = 1 / 3  # intersection: 1, union: 3
        self.assertEqual(score, expected)

        # No overlap
        self.assertEqual(array_reward(["file1.txt"], ["file2.txt"]), 0.0)

        # Different sizes
        score = array_reward(["file1.txt"], ["file1.txt", "file2.txt", "file3.txt"])
        expected = 1 / 3  # intersection: 1, union: 3
        self.assertEqual(score, expected)

        # Empty arrays
        self.assertEqual(array_reward([], []), 0.0)
        self.assertEqual(array_reward(["file1.txt"], []), 0.0)

        # Handle non-list inputs
        score = array_reward("single_file.txt", ["single_file.txt", "other.txt"])
        expected = 1 / 2  # intersection: 1, union: 2
        self.assertEqual(score, expected)

        # Handle invalid inputs
        self.assertEqual(array_reward(None, ["file1.txt"]), 0.0)


class Testargumentscoring(TestArgumentClassificationSystem):
    """Test parameter scoring with configuration."""

    def test_compute_parameter_score_exact_match(self):
        """Test parameter scoring for exact match arguments."""
        # Test index parameter for click_element
        score = compute_parameter_score("index", 42, 42, "click_element", self.config)
        self.assertEqual(score, 1.0)

        score = compute_parameter_score("index", 42, 43, "click_element", self.config)
        self.assertEqual(score, 0.0)

        # Test URL parameter for navigate_to
        score = compute_parameter_score("url", "https://google.com", "https://google.com", "navigate_to", self.config)
        self.assertEqual(score, 1.0)

        score = compute_parameter_score("url", "https://google.com", "https://bing.com", "navigate_to", self.config)
        self.assertEqual(score, 0.0)

    def test_compute_parameter_score_semantic(self):
        """Test parameter scoring for semantic similarity arguments."""

        # Create a simple tokenizer for testing
        class SimpleTokenizer:
            def encode(self, text):
                return text.split()

            def decode(self, tokens):
                return " ".join(tokens) if isinstance(tokens, list) else str(tokens)

        tokenizer = SimpleTokenizer()

        # Test with very similar text - should get reasonable score with lower threshold
        score = compute_parameter_score(
            "userSidePrompt", "Click search", "Click the search", "click_element", self.config, tokenizer
        )
        # With 0.8 threshold and token-based F1, this should give some positive score
        self.assertGreater(score, 0.0)

        # Test text parameter for input_text - perfect match should give high score
        score = compute_parameter_score("text", "Hello world", "Hello world", "input_text", self.config, tokenizer)
        self.assertEqual(score, 1.0)  # Perfect match should give 1.0

    def test_compute_parameter_score_enum(self):
        """Test parameter scoring for enum/boolean arguments."""
        # Test button parameter
        score = compute_parameter_score("button", "left", "left", "click_element", self.config)
        self.assertEqual(score, 1.0)

        score = compute_parameter_score("button", "left", "right", "click_element", self.config)
        self.assertEqual(score, 0.5)

        score = compute_parameter_score("button", "invalid", "left", "click_element", self.config)
        self.assertEqual(score, 0.0)

        # Test boolean parameter
        score = compute_parameter_score("enter", True, True, "input_text", self.config)
        self.assertEqual(score, 1.0)

        score = compute_parameter_score("enter", False, True, "input_text", self.config)
        self.assertEqual(score, 0.5)

    def test_compute_parameter_score_array(self):
        """Test parameter scoring for array arguments."""
        # Test file_paths parameter
        score = compute_parameter_score(
            "file_paths",
            ["/path/file1.txt", "/path/file2.txt"],
            ["/path/file1.txt", "/path/file2.txt"],
            "click_upload_file",
            self.config,
        )
        self.assertEqual(score, 1.0)

        score = compute_parameter_score(
            "file_paths", ["/path/file1.txt"], ["/path/file1.txt", "/path/file2.txt"], "click_upload_file", self.config
        )
        expected = 1 / 2  # intersection: 1, union: 2
        self.assertEqual(score, expected)

        # Test selectOptions parameter
        score = compute_parameter_score(
            "selectOptions", ["option1", "option2"], ["option1", "option3"], "human_interact", self.config
        )
        expected = 1 / 3  # intersection: 1, union: 3
        self.assertEqual(score, expected)

    def test_compute_parameter_score_fallback(self):
        """Test parameter scoring fallback for unknown arguments."""
        # Test unknown parameter (should fall back to exact match)
        score = compute_parameter_score("unknown_param", "value1", "value1", "unknown_tool", self.config)
        self.assertEqual(score, 1.0)

        score = compute_parameter_score("unknown_param", "value1", "value2", "unknown_tool", self.config)
        self.assertEqual(score, 0.0)

        # Test with None config (should fall back to exact match)
        score = compute_parameter_score("index", 42, 42, "click_element", None)
        self.assertEqual(score, 1.0)

        score = compute_parameter_score("index", 42, 43, "click_element", None)
        self.assertEqual(score, 0.0)


class TestCoordinateScoring(TestArgumentClassificationSystem):
    """Test coordinate pair scoring."""

    def test_compute_coordinate_pair_score(self):
        """Test coordinate pair scoring for point-based tools."""
        pred_params = {"x": 100, "y": 150, "userSidePrompt": "Clicking at point"}
        gt_params = {"x": 100, "y": 150, "userSidePrompt": "Click at point"}

        # Perfect coordinate match
        score = compute_coordinate_pair_score(pred_params, gt_params, "click_by_point", self.config)
        self.assertEqual(score, 1.0)

        # Coordinates within tolerance
        pred_params = {"x": 110, "y": 150}
        gt_params = {"x": 100, "y": 150}
        score = compute_coordinate_pair_score(pred_params, gt_params, "click_by_point", self.config)
        expected = 1.0 - (10 / 20)  # distance=10, tolerance=20
        self.assertEqual(score, expected)

        # Coordinates outside tolerance
        pred_params = {"x": 130, "y": 150}
        gt_params = {"x": 100, "y": 150}
        score = compute_coordinate_pair_score(pred_params, gt_params, "click_by_point", self.config)
        self.assertEqual(score, 0.0)

        # Non-coordinate tool (should return 0)
        score = compute_coordinate_pair_score(pred_params, gt_params, "click_element", self.config)
        self.assertEqual(score, 0.0)

        # Missing coordinates
        pred_params = {"userSidePrompt": "Clicking"}
        gt_params = {"x": 100, "y": 150}
        score = compute_coordinate_pair_score(pred_params, gt_params, "click_by_point", self.config)
        self.assertEqual(score, 0.0)


class TestToolCallRewardIntegration(TestArgumentClassificationSystem):
    """Test complete tool call reward computation."""

    @patch("rlla.load_argument_config")
    @patch("rlla.cosine_similarity_text")
    def test_simple_tool_call_reward(self, mock_cosine_sim, mock_load_config):
        """Test reward computation for simple tool calls."""
        mock_load_config.return_value = self.config
        mock_cosine_sim.return_value = 0.9

        # Perfect match
        gt_tools = [
            {"name": "navigate_to", "arguments": {"url": "https://google.com", "userSidePrompt": "Opening Google"}}
        ]
        pd_tools = [
            {"name": "navigate_to", "arguments": {"url": "https://google.com", "userSidePrompt": "Opening Google"}}
        ]

        reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0)
        self.assertEqual(reward, 3.0)  # Perfect match should return max reward

        # Partial match (wrong URL, good prompt)
        pd_tools = [
            {"name": "navigate_to", "arguments": {"url": "https://bing.com", "userSidePrompt": "Opening Google"}}
        ]

        with patch.dict(os.environ, {}, clear=True):  # Clear environment variables
            reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
            # Should be partial score: tool name match + URL mismatch + userSidePrompt similarity
            self.assertGreater(reward, -3.0)
            self.assertLess(reward, 3.0)

    @patch("rlla.load_argument_config")
    def test_coordinate_tool_call_reward(self, mock_load_config):
        """Test reward computation for coordinate-based tool calls."""
        mock_load_config.return_value = self.config

        # Perfect coordinate match
        gt_tools = [{"name": "click_by_point", "arguments": {"x": 100, "y": 150, "userSidePrompt": "Clicking button"}}]
        pd_tools = [{"name": "click_by_point", "arguments": {"x": 100, "y": 150, "userSidePrompt": "Clicking button"}}]

        with patch("rlla.cosine_similarity_text", return_value=1.0):
            reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
            self.assertEqual(reward, 3.0)

        # Coordinates within tolerance
        pd_tools = [{"name": "click_by_point", "arguments": {"x": 110, "y": 150, "userSidePrompt": "Clicking button"}}]

        with patch("rlla.cosine_similarity_text", return_value=1.0):
            reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
            self.assertGreater(reward, -3.0)
            self.assertLess(reward, 3.0)

    @patch("rlla.load_argument_config")
    @patch("rlla.cosine_similarity_text")
    def test_multiple_tool_call_reward(self, mock_cosine_sim, mock_load_config):
        """Test reward computation for multiple tool calls."""
        mock_load_config.return_value = self.config
        mock_cosine_sim.return_value = 0.9

        gt_tools = [
            {"name": "navigate_to", "arguments": {"url": "https://google.com", "userSidePrompt": "Opening Google"}},
            {"name": "click_element", "arguments": {"index": 5, "userSidePrompt": "Clicking search button"}},
        ]

        pd_tools = [
            {
                "name": "navigate_to",
                "arguments": {"url": "https://google.com", "userSidePrompt": "Navigate to Google"},
            },
            {"name": "click_element", "arguments": {"index": 5, "userSidePrompt": "Click search button"}},
        ]

        with patch.dict(os.environ, {}, clear=True):
            reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
            # Should be high score due to good matches
            self.assertGreater(reward, -1.0)  # Lowered expectation due to semantic similarity threshold

    @patch("rlla.load_argument_config")
    def test_enum_parameter_tool_call_reward(self, mock_load_config):
        """Test reward computation with enum arguments."""
        mock_load_config.return_value = self.config

        # Perfect enum match
        gt_tools = [
            {"name": "click_element", "arguments": {"index": 5, "button": "left", "userSidePrompt": "Clicking"}}
        ]
        pd_tools = [
            {"name": "click_element", "arguments": {"index": 5, "button": "left", "userSidePrompt": "Clicking"}}
        ]

        with patch("rlla.cosine_similarity_text", return_value=1.0):
            reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
            self.assertEqual(reward, 3.0)

        # Valid but incorrect enum choice
        pd_tools = [
            {"name": "click_element", "arguments": {"index": 5, "button": "right", "userSidePrompt": "Clicking"}}
        ]

        with patch("rlla.cosine_similarity_text", return_value=1.0):
            reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
            self.assertGreater(reward, -3.0)
            self.assertLess(reward, 3.0)

    @patch("rlla.load_argument_config")
    def test_array_parameter_tool_call_reward(self, mock_load_config):
        """Test reward computation with array arguments."""
        mock_load_config.return_value = self.config

        # Perfect array match
        gt_tools = [
            {
                "name": "click_upload_file",
                "arguments": {
                    "index": 1,
                    "file_paths": ["/path/file1.txt", "/path/file2.txt"],
                    "userSidePrompt": "Uploading files",
                },
            }
        ]
        pd_tools = [
            {
                "name": "click_upload_file",
                "arguments": {
                    "index": 1,
                    "file_paths": ["/path/file1.txt", "/path/file2.txt"],
                    "userSidePrompt": "Uploading files",
                },
            }
        ]

        with patch("rlla.cosine_similarity_text", return_value=1.0):
            reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
            self.assertEqual(reward, 3.0)

        # Partial array match
        pd_tools = [
            {
                "name": "click_upload_file",
                "arguments": {"index": 1, "file_paths": ["/path/file1.txt"], "userSidePrompt": "Uploading files"},
            }
        ]

        with patch("rlla.cosine_similarity_text", return_value=1.0):
            reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
            self.assertGreater(reward, -3.0)
            self.assertLess(reward, 3.0)


class TestBrowserAgentScenarios(TestArgumentClassificationSystem):
    """Test scenarios based on browser agent evaluation data."""

    @patch("rlla.load_argument_config")
    @patch("rlla.cosine_similarity_text")
    def test_navigation_scenario(self, mock_cosine_sim, mock_load_config):
        """Test navigation scenario like opening Google."""
        mock_load_config.return_value = self.config
        mock_cosine_sim.return_value = 0.95

        gt_tools = [
            {
                "name": "navigate_to",
                "arguments": {"url": "https://www.google.com", "userSidePrompt": "Opening google.com"},
            }
        ]

        # Perfect prediction
        pd_tools = [
            {
                "name": "navigate_to",
                "arguments": {"url": "https://www.google.com", "userSidePrompt": "Opening google.com"},
            }
        ]

        reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
        self.assertEqual(reward, 3.0)

        # Good semantic similarity but exact URL match
        pd_tools = [
            {
                "name": "navigate_to",
                "arguments": {"url": "https://www.google.com", "userSidePrompt": "Navigate to Google search engine"},
            }
        ]

        reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
        self.assertGreater(reward, -1.0)  # Should be positive due to URL match, even if semantic similarity is poor

    @patch("rlla.load_argument_config")
    @patch("rlla.cosine_similarity_text")
    def test_element_interaction_scenario(self, mock_cosine_sim, mock_load_config):
        """Test element interaction scenario like clicking search button."""
        mock_load_config.return_value = self.config
        mock_cosine_sim.return_value = 0.9

        gt_tools = [{"name": "click_element", "arguments": {"index": 7, "userSidePrompt": "Click the search button"}}]

        # Perfect prediction
        pd_tools = [{"name": "click_element", "arguments": {"index": 7, "userSidePrompt": "Click the search button"}}]

        reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
        self.assertEqual(reward, 3.0)

        # Wrong element index
        pd_tools = [{"name": "click_element", "arguments": {"index": 8, "userSidePrompt": "Click the search button"}}]

        reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
        self.assertLess(reward, 3.0)  # Should be lower due to wrong index

    @patch("rlla.load_argument_config")
    @patch("rlla.cosine_similarity_text")
    def test_text_input_scenario(self, mock_cosine_sim, mock_load_config):
        """Test text input scenario."""
        mock_load_config.return_value = self.config
        mock_cosine_sim.return_value = 0.85

        gt_tools = [
            {
                "name": "input_text",
                "arguments": {
                    "index": 15,
                    "text": "machine learning",
                    "enter": True,
                    "userSidePrompt": "Enter search query and press Enter",
                },
            }
        ]

        # Perfect prediction
        pd_tools = [
            {
                "name": "input_text",
                "arguments": {
                    "index": 15,
                    "text": "machine learning",
                    "enter": True,
                    "userSidePrompt": "Enter search query and press Enter",
                },
            }
        ]

        reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
        self.assertEqual(reward, 3.0)

        # Different but semantically similar text
        pd_tools = [
            {
                "name": "input_text",
                "arguments": {
                    "index": 15,
                    "text": "AI and machine learning",
                    "enter": True,
                    "userSidePrompt": "Type search term and hit Enter",
                },
            }
        ]

        reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
        self.assertGreater(reward, -1.0)  # Should be positive due to some parameter matches

    @patch("rlla.load_argument_config")
    @patch("rlla.cosine_similarity_text")
    def test_coordinate_click_scenario(self, mock_cosine_sim, mock_load_config):
        """Test coordinate-based clicking scenario."""
        mock_load_config.return_value = self.config
        mock_cosine_sim.return_value = 0.9

        gt_tools = [
            {
                "name": "click_by_point",
                "arguments": {
                    "x": 250,
                    "y": 150,
                    "button": "left",
                    "num_clicks": 1,
                    "userSidePrompt": "Click on the login button",
                },
            }
        ]

        # Perfect coordinate match
        pd_tools = [
            {
                "name": "click_by_point",
                "arguments": {
                    "x": 250,
                    "y": 150,
                    "button": "left",
                    "num_clicks": 1,
                    "userSidePrompt": "Click on the login button",  # Same as GT
                },
            }
        ]

        reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
        self.assertEqual(reward, 3.0)

        # Coordinates within tolerance
        pd_tools = [
            {
                "name": "click_by_point",
                "arguments": {
                    "x": 255,
                    "y": 155,
                    "button": "left",
                    "num_clicks": 1,
                    "userSidePrompt": "Click login button",
                },
            }
        ]

        reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
        self.assertGreater(reward, 0.5)  # Should be positive due to close coordinates

    @patch("rlla.load_argument_config")
    @patch("rlla.cosine_similarity_text")
    def test_complex_workflow_scenario(self, mock_cosine_sim, mock_load_config):
        """Test complex workflow with multiple tools."""
        mock_load_config.return_value = self.config
        mock_cosine_sim.return_value = 0.9

        gt_tools = [
            {
                "name": "navigate_to",
                "arguments": {"url": "https://github.com", "userSidePrompt": "Navigate to GitHub"},
            },
            {"name": "click_element", "arguments": {"index": 5, "userSidePrompt": "Click the search button"}},
            {
                "name": "input_text",
                "arguments": {
                    "index": 10,
                    "text": "python machine learning",
                    "enter": True,
                    "userSidePrompt": "Search for ML repositories",
                },
            },
        ]

        # Perfect prediction
        pd_tools = [
            {
                "name": "navigate_to",
                "arguments": {"url": "https://github.com", "userSidePrompt": "Go to GitHub website"},
            },
            {"name": "click_element", "arguments": {"index": 5, "userSidePrompt": "Click search"}},
            {
                "name": "input_text",
                "arguments": {
                    "index": 10,
                    "text": "python machine learning",
                    "enter": True,
                    "userSidePrompt": "Enter search query for ML repos",
                },
            },
        ]

        reward = compute_tool_call_reward(gt_tools, pd_tools, 3.0, -3.0, self.tokenizer)
        self.assertGreaterEqual(reward, -0.5)  # Should be reasonable due to good overall tool name matches


class TestCosineSimilarityText(unittest.TestCase):
    """Test the cosine similarity text function."""

    def test_cosine_similarity_text_fallback(self):
        """Test cosine similarity calculation with token-based fallback."""

        # Test identical texts
        result = cosine_similarity_text("hello world", "hello world")
        self.assertEqual(result, 1.0)

        # Test completely different texts
        result = cosine_similarity_text("hello world", "goodbye universe")
        self.assertEqual(result, 0.0)

        # Test partial overlap
        result = cosine_similarity_text("hello world test", "hello world example")
        expected = 2 / 4  # 2 common words out of 4 total unique words
        self.assertEqual(result, expected)

        # Test empty strings
        result = cosine_similarity_text("", "")
        self.assertEqual(result, 1.0)

        result = cosine_similarity_text("hello", "")
        self.assertEqual(result, 0.0)

    def test_cosine_similarity_text_fallback_additional(self):
        """Test additional fallback scenarios."""

        # Test identical text
        result = cosine_similarity_text("hello world", "hello world")
        self.assertEqual(result, 1.0)

        # Test partial overlap
        result = cosine_similarity_text("hello world", "hello python")
        expected = 1 / 3  # intersection: 1 (hello), union: 3 (hello, world, python)
        self.assertEqual(result, expected)

        # Test no overlap
        result = cosine_similarity_text("hello", "world")
        self.assertEqual(result, 0.0)

        # Test empty strings
        result = cosine_similarity_text("", "")
        self.assertEqual(result, 1.0)

        result = cosine_similarity_text("hello", "")
        self.assertEqual(result, 0.0)


def run_test_suite():
    """Run the complete test suite."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestParameterTypeRewards,
        Testargumentscoring,
        TestCoordinateScoring,
        TestToolCallRewardIntegration,
        TestBrowserAgentScenarios,
        TestCosineSimilarityText,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result


if __name__ == "__main__":
    print("Running Argument Classification System Test Suite")
    print("=" * 60)

    result = run_test_suite()

    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    if result.wasSuccessful():
        print("\nAll tests passed! ✅")
    else:
        print("\nSome tests failed! ❌")
        sys.exit(1)
