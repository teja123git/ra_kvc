"""Small randomly initialized model checks. No model download required."""

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from ra_kvc.cache import CacheQuantizer, as_legacy
from ra_kvc.signals import SignalTracker


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        config = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, max_position_embeddings=256)
        self.model = Qwen2ForCausalLM(config).eval()
        self.ids = torch.randint(0, 128, (1, 48))
        self.cal = {"entropy": {"mean": 0.5, "std": 0.2}, "margin": {"mean": 1, "std": 1}}

    @torch.inference_mode()
    def test_cache_quantization_changes_next_logits(self):
        baseline = self.model(self.ids, use_cache=True)
        cache = as_legacy(baseline.past_key_values)
        copy = tuple((k.clone(), v.clone()) for k, v in cache)
        q = CacheQuantizer("uniform2", 48, SignalTracker(), self.cal,
                           group_size=8, sink_tokens=4, recent_tokens=8)
        altered = q.apply(copy)
        token = torch.tensor([[5]])
        a = self.model(token, past_key_values=cache, use_cache=True).logits
        b = self.model(token, past_key_values=altered, use_cache=True).logits
        self.assertGreater((a-b).abs().max().item(), 1e-7)

    @torch.inference_mode()
    def test_cache_path_matches_full_prefix(self):
        out = self.model(self.ids, use_cache=True)
        token = torch.tensor([[5]])
        cached = self.model(token, past_key_values=as_legacy(out.past_key_values)).logits[:, -1]
        full = self.model(torch.cat([self.ids, token], dim=1), use_cache=False).logits[:, -1]
        torch.testing.assert_close(cached, full, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
