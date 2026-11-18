import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ra_kvc.scoring import paired_bootstrap
from run_humaneval import extract_program


class EvaluationTests(unittest.TestCase):
    def test_unclosed_code_fence_keeps_generated_code(self):
        problem = {"entry_point": "f", "prompt": "def f(x):\n    \"\"\"Return x.\"\"\"\n"}
        code = "def f(x):\n    return x"
        self.assertEqual(extract_program("```python\n" + code, problem), code)
        self.assertEqual(extract_program("```python\n" + code + "\n```\nExplanation", problem).rstrip(), code)

    def test_wrong_function_name_is_not_repaired(self):
        problem = {"entry_point": "f", "prompt": "def f(x):\n    \"\"\"Return x.\"\"\"\n"}
        extracted = extract_program("def g(x):\n    return 123", problem)
        self.assertIn("def g(x):", extracted)
        self.assertNotIn("def f(x):\n    return 123", extracted)

    def test_paired_discordance_and_exact_test(self):
        result = paired_bootstrap([1]*6, [0]*6)
        self.assertEqual(result["difference"], 1)
        self.assertEqual(result["wins"], 6)
        self.assertEqual(result["losses"], 0)
        self.assertEqual(result["mcnemar_exact_p_two_sided"], 0.03125)
        equal = paired_bootstrap([0, 1], [0, 1])
        self.assertEqual(equal["discordant_pairs"], 0)
        self.assertIn("does not establish equivalence", equal["note"])


if __name__ == "__main__":
    unittest.main()
