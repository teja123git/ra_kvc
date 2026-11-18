# Generation-length check

The 256-token pilot showed frequent unfinished answers. This diagnostic reuses the same 20 predetermined test IDs and the same model, prompt and greedy decoding, but increases the cap to 512 tokens for the unquantized reference. It was added after observing pilot truncation and is not an independent confirmation study. Calibration is reused unchanged; no test answer selects an uncertainty threshold.

This check measures sensitivity to the response cap. It does not compare fixed versus adaptive policies at the longer cap.
