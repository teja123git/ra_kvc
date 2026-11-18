import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import torch
from ra_kvc.quantization import fake_quantize, quantize_block
from ra_kvc.controller import BudgetController
from ra_kvc.signals import SignalTracker, uncertainty
from ra_kvc.memory import cache_storage
from ra_kvc.cache import CacheQuantizer
from ra_kvc.scoring import extract_answer, wilson

CAL = {"entropy": {"mean": 0.5, "std": 0.1}, "margin": {"mean": 1, "std": 1}}


class QuantizerTests(unittest.TestCase):
    def test_constant_groups(self):
        x = torch.full((1, 2, 32, 64), -3.25)
        self.assertTrue(torch.equal(x, fake_quantize(x, 2, -2)))

    def test_four_levels_and_endpoints(self):
        x = torch.linspace(-2, 3, 100).reshape(1, 1, 100, 1)
        q = fake_quantize(x, 2, -2)
        self.assertEqual(q.unique().numel(), 4)
        self.assertEqual(q.min().item(), -2)
        self.assertEqual(q.max().item(), 3)

    def test_promoting_all_channels_matches_int4(self):
        torch.manual_seed(7)
        k = torch.randn(1, 2, 32, 64)
        v = torch.randn_like(k)
        got, values = quantize_block(k, v, 64)
        self.assertTrue(torch.equal(got, fake_quantize(k, 4, -2)))
        self.assertTrue(torch.equal(values, fake_quantize(v, 2, -1)))
        self.assertLess(((got-k)**2).mean(), ((fake_quantize(k, 2, -2)-k)**2).mean())


class ControllerTests(unittest.TestCase):
    def test_never_spends_future_budget(self):
        for signal in (0.0, 0.5, 1.0):
            controller = BudgetController()
            for _ in range(100):
                n = controller.allocate(64, signal, CAL)
                self.assertGreaterEqual(n, 0)
                self.assertLessEqual(n, 16)
                self.assertLessEqual(controller.spent, controller.earned+1e-9)
                self.assertLessEqual(controller.credit, 9)

    def test_saved_credit_can_be_used(self):
        c = BudgetController()
        self.assertEqual(c.allocate(64, 0, CAL), 0)
        self.assertEqual(c.allocate(64, 1, CAL), 16)
        self.assertEqual(c.credit, 0)


class CacheTests(unittest.TestCase):
    def test_sink_tail_and_finalized_history(self):
        torch.manual_seed(8)
        k, v = torch.randn(1, 2, 88, 64), torch.randn(1, 2, 88, 64)
        original_k = k.clone()
        cache = CacheQuantizer("fixed", 20, SignalTracker(), CAL, group_size=16, sink_tokens=8, recent_tokens=16)
        out = cache.apply(((k, v),))
        self.assertTrue(torch.equal(out[0][0][..., :8, :], original_k[..., :8, :]))
        self.assertTrue(torch.equal(out[0][0][..., 72:, :], original_k[..., 72:, :]))
        self.assertFalse(torch.equal(out[0][0][..., 8:72, :], original_k[..., 8:72, :]))
        frozen = out[0][0].clone()
        cache.apply(out)
        self.assertTrue(torch.equal(frozen, out[0][0]))
        self.assertEqual(len(cache.records), 4)

    def test_fp_reference_unchanged(self):
        k, v = torch.randn(1, 2, 100, 64), torch.randn(1, 2, 100, 64)
        cache = CacheQuantizer("fp16", 10, SignalTracker(), CAL)
        got = cache.apply(((k, v),))
        self.assertTrue(torch.equal(got[0][0], k))
        self.assertEqual(cache.records, [])


class AccountingTests(unittest.TestCase):
    def test_reference_exact_size(self):
        got = cache_storage(4, 2, 64, 100, [])
        self.assertEqual(got["estimated_total_bytes"], 4*2*64*100*2*2)

    def test_boost_cost_and_metadata(self):
        base = {"start": 0, "end": 32, "mode": "fixed", "boosted": 0}
        a = cache_storage(1, 1, 64, 32, [base])
        b = cache_storage(1, 1, 64, 32, [{**base, "boosted": 8}])
        self.assertEqual(b["estimated_total_bytes"]-a["estimated_total_bytes"], 32*8*2/8)
        self.assertEqual(a["metadata_bytes"], 8*64+8*32+8+8)


class ScoringTests(unittest.TestCase):
    def test_explicit_answer_beats_intermediate(self):
        self.assertEqual(extract_answer("We have 30. #### 1,200.00\nDone.")[0], "1200")
        self.assertEqual(extract_answer("\\boxed{-2.5}")[0], "-2.5")
        self.assertEqual(extract_answer("No answer", False)[0], None)

    def test_entropy_and_margin(self):
        uniform = uncertainty(torch.zeros(32))
        certain = uncertainty(torch.tensor([20.]+[0.]*31))
        self.assertAlmostEqual(uniform["entropy"], 1.0, places=5)
        self.assertLess(certain["entropy"], 0.001)
        self.assertGreater(certain["margin"], uniform["margin"])


if __name__ == "__main__":
    unittest.main()
