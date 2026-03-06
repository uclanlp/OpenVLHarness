import unittest

from rl.rewards.compute_score import compute_score


class ComputeScoreTest(unittest.TestCase):
    def test_mcq_accuracy_with_tool_bonus(self):
        result = compute_score(
            data_source="MMStar",
            solution_str=(
                "<thinking>...</thinking>\n"
                '<tool_call>{"name":"OCR_Tool","arguments":{"image_path":"image_1"}}</tool_call>\n'
                "Final Answer: B"
            ),
            ground_truth="B",
            extra_info={"num_turns": 4},
        )
        self.assertAlmostEqual(result["acc"], 1.0)
        self.assertAlmostEqual(result["wellformed_tool_json_rate"], 1.0)
        self.assertGreater(result["score"], 1.0)

    def test_malformed_tool_penalty(self):
        result = compute_score(
            data_source="MMStar",
            solution_str='<tool_call>{"name":"OCR_Tool","arguments":</tool_call> Final Answer: A',
            ground_truth="B",
            extra_info=None,
        )
        self.assertEqual(result["acc"], 0.0)
        self.assertEqual(result["wellformed_tool_json_rate"], 0.0)
        self.assertLess(result["score"], 0.1)


if __name__ == "__main__":
    unittest.main()
