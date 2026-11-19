"""Build the report directly from saved, paired experimental outputs."""

import argparse
import csv
import html
import json
import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
TEAM = [
    ("Kondapi Venkata Dhruv", "230122024", "Quantizer, channel selection and fixed baselines"),
    ("Mathangi Venkatraman Srinivasan", "230121037", "Uncertainty signals and sensitivity diagnostic"),
    ("Venkat Sai Teja Kalluri", "230121032", "Budget controller and cache integration"),
    ("Sanjay G", "230121049", "Evaluation, scoring, statistics and plots"),
    ("Vipul Sharma", "230103117", "Storage accounting and hardware analysis"),
]


def pct(value):
    return f"{100*value:.1f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/local_pilot")
    p.add_argument("--humaneval")
    p.add_argument("--length-check", help="Optional separate unquantized longer-generation diagnostic")
    p.add_argument("--output-dir", default="report")
    args = p.parse_args()
    run, output = Path(args.run).resolve(), Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run/"summary.json").read_text())
    manifest, methods = summary["manifest"], summary["methods"]
    cfg, dims = manifest["config"], manifest["model_dimensions"]
    n = summary["completed_paired_questions"]
    rows = {r["mode"]: r for r in methods}
    diagnostic = json.loads((run/"signal_diagnostic.json").read_text()) if (run/"signal_diagnostic.json").exists() else None
    profile = json.loads((run/"overhead_profile.json").read_text()) if (run/"overhead_profile.json").exists() else None
    human = None
    if args.humaneval and (Path(args.humaneval)/"summary.json").exists():
        human = json.loads((Path(args.humaneval)/"summary.json").read_text())
    length_check = None
    if args.length_check:
        length_check = json.loads((Path(args.length_check)/"summary.json").read_text())
        if (length_check["manifest"]["config"]["model"] != cfg["model"] or
                length_check["manifest"]["dataset"]["sample_ids"] != manifest["dataset"]["sample_ids"]):
            raise ValueError("Length diagnostic must use the same model and selected test IDs")
    fixed = rows.get("fixed")
    adaptive = rows.get("entropy")
    headline = ", ".join(f"{r['label']}: {r['correct']}/{n}" for r in methods)
    if fixed and adaptive:
        delta = 100*(adaptive["accuracy"]-fixed["accuracy"])
        outcome = f"Entropy-adaptive allocation differed from fixed allocation by {delta:+.1f} percentage points on this cohort."
        pair = summary["paired_comparisons"].get("entropy_minus_fixed")
        if pair:
            outcome += f" The paired bootstrap interval for this difference was [{100*pair['ci95'][0]:+.1f}, {100*pair['ci95'][1]:+.1f}] percentage points."
            if pair["ci95"][0] <= 0 <= pair["ci95"][1]:
                outcome += " This interval includes zero and does not establish an improvement."
            if pair.get("discordant_pairs") == 0:
                outcome += " Both policies have identical correctness on every sampled question. The resulting degenerate bootstrap interval does not establish equivalence."
    else:
        outcome = "The supplied run does not include a complete fixed-versus-entropy comparison."

    sections = []
    def section(title, paragraphs=(), table=None, figures=(), widths=None):
        sections.append({"title": title, "paragraphs": list(paragraphs), "table": table,
                         "figures": list(figures), "widths": widths})

    section("Abstract", [
        "The KV cache stores the key and value representations needed for autoregressive language-model generation. Low-bit quantization reduces its potential storage cost but can change the model's outputs. This project studies whether output uncertainty can guide a time-varying allocation of extra precision while constraining the total protection budget.",
        f"We implement RA-KVC as a PyTorch fake-quantization prototype and compare policies on {cfg['model']} using {n} paired GSM8K test questions. The shared quantizer uses 2-bit keys and values with selected key channels at 4 bits. Entropy and logit-margin variants change the protection fraction for generated token blocks. A credit controller limits cumulative allocation, and an explicit storage model includes metadata and floating buffers.",
        outcome,
        "The findings are exploratory. Accuracy is measured from actual generated answers. Packed-cache memory is an analytical projection, while the implementation retains floating tensors. The study does not demonstrate production speedup, real INT2 memory allocation, or a general ability to detect reasoning."
    ])
    if length_check:
        ref_long = next(r for r in length_check["methods"] if r["mode"] == "fp16")
        ref_short = rows["fp16"]
        sections[-1]["paragraphs"].insert(3, f"A post-pilot length diagnostic changes the unquantized cap from {cfg['max_new_tokens']} to {length_check['manifest']['config']['max_new_tokens']} tokens. Correct answers change from {ref_short['correct']}/{ref_short['n']} to {ref_long['correct']}/{ref_long['n']}, while truncated responses change from {ref_short['truncated']} to {ref_long['truncated']}. This highlights generation length as a confounder; the longer-cap diagnostic does not compare adaptive policies.")
    section("1. Introduction and research question", [
        "An autoregressive transformer repeatedly attends to earlier token representations. Caching keys and values avoids recomputing those representations, but the cache grows with sequence length. For grouped-query attention, storage depends on the number of KV heads, which can be smaller than the number of query heads. A hardware-aware evaluation must therefore account for model dimensions, context length, precision and representation overhead.",
        "Kitty [1] motivates protecting selected key channels at higher precision while keeping most cache values at 2 bits. Its channel choices already adapt to the contents of each group. Our research question concerns a different decision: can the fraction of channels protected in each generated block vary usefully with output uncertainty? We use a simplified Kitty-inspired baseline and do not claim a reproduction of its optimized kernels or complete configuration.",
        "The initial proposal described entropy as a reasoning-demand signal. Here it is treated as a hypothesis. A model can be uncertain for reasons unrelated to reasoning, and a confidently generated token may still need accurate retention. We therefore measure both answer accuracy and a local cache-sensitivity diagnostic.",
        "The central comparison keeps the model, prompts, quantizer, channel ranker and buffer layout the same. Fixed allocation protects a constant fraction of key channels in every eligible block. Adaptive allocation changes that fraction only for blocks made entirely of generated tokens. HumanEval, when included, is a secondary coding task rather than an easy non-reasoning control."
    ])
    section("2. Quantization and uncertainty signals", [
        "For a group x with minimum l and maximum u, an affine b-bit quantizer uses scale s=(u-l)/(2^b-1) and reconstructs x_hat=l+s*clip(round((x-l)/s), 0, 2^b-1). Constant groups are preserved exactly. K groups reduce over token positions within a block, while V groups reduce over channels within a token. Computation uses FP32 before returning to the cache dtype.",
        "The mixed-precision method ranks key channels by their mean absolute magnitude within the block, independently for each layer and KV head. Its highest-ranked channels use 4 bits, while the remaining key channels and the values use 2 bits. Fixed and adaptive policies share this rule. Only the number of protected channels changes.",
        "Normalized entropy is H=-sum(p log p)/log(V), where V is vocabulary size. The margin is the largest minus second-largest raw logit. Larger entropy and smaller margin both indicate greater uncertainty. The implementation records the uncertainty that predicted a token at that token's absolute position and smooths the signal with an exponential moving average.",
        f"The smoothing coefficient is {cfg['smoothing']}. Calibration uses {cfg['calibration_samples']} predetermined GSM8K training questions, disjoint from the test cohort. Mean and standard deviation of the smoothed signals define a standardized score, clipped to [-1,1]. The margin's sign is reversed. A block uses the average signal of its tokens.",
        "The model already produces logits, but the entropy reduction, top-two selection, smoothing and controller still cost time. These operations are included in the measured simulation runtime. No zero-overhead claim is made."
    ])
    section("3. Causal cache operation and budget control", [
        f"The cache retains the first {cfg['sink_tokens']} tokens and at least the most recent {cfg['recent_tokens']} tokens in floating precision. Eligible tokens form groups of {cfg['group_size']}. A trailing incomplete group remains floating. Each new block is finalized after a forward pass and before the next forward pass, so quantization affects subsequent attention. Already finalized blocks are never requantized or retroactively promoted.",
        "Prompt blocks and blocks crossing the prompt/response boundary use fixed protection in every mixed-precision method. Adaptation starts only for wholly generated blocks. This prevents prompt processing from introducing an unrelated difference between fixed and adaptive variants.",
        f"The nominal protection fraction is r={cfg['target_fraction']:.3f}. For head dimension D, each newly finalized generated block earns rD credits. The controller requests round(rD*(1+z)) protected channels, where z is clipped uncertainty. It cannot spend future credits. A lower allocation bound also prevents the unspent balance from growing beyond approximately one nominal block budget.",
        "This design guarantees that cumulative generated-block protection does not exceed its target, apart from numerical tolerance. It does not guarantee exact equality when generation ends: unused credit can remain at EOS or the token limit. Integer channel counts, incomplete blocks and realized storage are explicitly reported. A nominally shared target alone is insufficient to establish an equal-memory improvement.",
        "The random control replaces the uncertainty score with a seeded random score and keeps the same controller and channel ranker. The rapid study uses one random seed. It is a diagnostic control, not a robust average over many random schedules."
    ], table=[
        ["Policy", "Eligible keys", "Eligible values"],
        ["Unquantized", "Original floating dtype", "Original floating dtype"],
        ["Uniform 2-bit", "All channels at 2 bits", "2 bits"],
        ["Fixed boost", "Fixed fraction at 4 bits", "2 bits"],
        ["Entropy / margin", "Credit-constrained variable fraction", "2 bits"],
        ["Random allocation", "Random score under the same controller", "2 bits"]
    ], widths=[110,270,125])
    section("4. Experimental protocol", [
        "The test cohort is a deterministic shuffled subset of GSM8K [3]. All methods see the same questions and the same chat prompt. The prompt requests a concise step-by-step solution ending with an explicit numeric answer marker. Greedy decoding avoids sampling variation. The runner saves text, token IDs, answer extraction, cache allocations, timing and storage accounting for every method/question pair.",
        "Numeric exact match is the primary metric. The scorer prioritizes the explicit answer marker and boxed/final-answer forms. For completed generations only, it can fall back to the last number. A truncated response without an explicit final answer is counted as incorrect instead of treating an intermediate number as its answer. Truncation and missing-answer counts are reported.",
        "Accuracy intervals use the Wilson method. Comparisons with fixed allocation use a paired bootstrap over question identities, with discordant-pair counts and an exact McNemar test in the saved summary. Small-sample intervals are descriptive and cannot support broad generalization. Analysis includes only questions completed by every configured method; the status below records whether the requested cohort completed.",
        "The small model and short generation cap are practical scope reductions from the original proposal. Larger GPU configurations are included for replication, but they are not presented as completed experiments. CPU latency measures this Python simulation, not the performance of a packed low-bit inference engine."
    ], table=[
        ["Setting", "Executed value"], ["Model", cfg["model"]],
        ["Model revision", str(manifest.get("model_revision"))],
        ["Device / weight dtype", manifest["device"]+" / "+manifest["dtype"]],
        ["Test questions", f"{n} paired of {cfg['samples']} planned ({summary['status']})"],
        ["Calibration / test split", f"{cfg['calibration_samples']} train / {n} test questions"],
        ["Maximum generated tokens", str(cfg["max_new_tokens"])],
        ["Layers / KV heads / head dimension", f"{dims['layers']} / {dims['kv_heads']} / {dims['head_dim']}"],
        ["PyTorch / Transformers", manifest["torch"]+" / "+manifest["transformers"]],
        ["Seed", str(cfg["seed"])],
    ], widths=[180,325])
    result_table = [["Method", "Correct", "Accuracy", "95% interval", "Truncated"]]
    for r in methods:
        result_table.append([r["label"], f"{r['correct']}/{r['n']}", pct(r["accuracy"]),
                             f"{pct(r['ci95_low'])} - {pct(r['ci95_high'])}", str(r["truncated"])])
    section("5. GSM8K results", [
        "The following values are calculated from the saved generated answers. They are measurements of this configuration and selected cohort, not expected values copied from prior work.",
        outcome,
        "Quantization can occasionally change a wrong answer into a correct one, so the unquantized score is a reference rather than a guaranteed upper bound. Conversely, an apparent gain can reflect response-length changes or chance on a small cohort. The raw answers and truncation counts allow these cases to be inspected."
    ], table=result_table, widths=[145,60,70,155,75],
        figures=[(run/"figures/accuracy.png", "Figure 1. Actual paired GSM8K accuracy with 95% Wilson intervals.")])
    if length_check:
        long = next(r for r in length_check["methods"] if r["mode"] == "fp16")
        short = rows["fp16"]
        section("5.1. Generation-length diagnostic", [
            "The short-cap pilot produced many unfinished responses. We therefore added an unquantized reference run with a larger generation cap on the same predetermined questions. This is a post-pilot diagnostic, not an independently selected confirmation study. Model weights, prompt, greedy decoding and the calibration split remain unchanged.",
            "The table isolates the generation-limit change within the unquantized policy. It helps determine whether low short-cap scores reflect missing final answers. Fixed and adaptive policies were not evaluated at the larger cap in this diagnostic, so the longer-reference result must not be compared with their short-cap scores as an equal-protocol compression result.",
            "A stronger follow-up should rerun every compared policy with a cap that permits complete answers. The supplied rapid GPU configuration allows 512 new tokens for this reason."
        ], table=[["Reference run", "Questions", "Correct", "Accuracy", "Truncated"],
                  [f"{cfg['max_new_tokens']}-token cap", str(short["n"]), str(short["correct"]), pct(short["accuracy"]), str(short["truncated"])],
                  [f"{length_check['manifest']['config']['max_new_tokens']}-token cap", str(long["n"]), str(long["correct"]), pct(long["accuracy"]), str(long["truncated"])]],
                widths=[155,80,75,85,110])

    storage_table = [["Method", "Effective bits", "Mean cache KiB", "Generated boost"]]
    for r in methods:
        storage_table.append([r["label"], f"{r['effective_bits_including_overhead']:.3f}",
                              f"{r['mean_estimated_cache_KiB']:.1f}",
                              pct(r["boosted_fraction_generated"]) if r["generated_blocks"] else "Not applicable"])
    section("6. Storage accounting and allocation behavior", [
        "With fraction r of keys protected, payload precision averages 2+2r bits for K and 2 bits for V, hence 2+r bits across both. At r=0.125 this is 2.125 bits before representation overhead. A 32-token, 64-channel group for one layer/head needs 1088 payload bytes plus 784 metadata bytes: 3.65625 bits per K/V scalar before floating buffers, versus 8192 bytes for that group at 16 bits.",
        "Our specified packed format adds two FP32 parameters per K channel and V token, a channel bitmap, an eight-byte per-head block descriptor, and 16-bit floating buffers. The table includes these costs. Effective bits aggregate estimated bytes over observed cache elements, and total storage also depends on response length.",
        "Generated boost counts only finalized generated blocks. The real CPU cache remains FP32, while the storage projection assumes a 16-bit reference. Actual tensor bytes are logged separately. No physical packing or memory reduction is implemented."
    ], table=storage_table, widths=[165,100,115,125],
        figures=[(run/"figures/memory_breakdown.png", "Figure 2. Mean projected cache size at observed response lengths, decomposed into integer payload, representation metadata and floating storage.")])
    section("6.1. Allocation trace and observed trade-off", [
        "The trace uses the longest entropy-policy cache trajectory, selected by block count rather than correctness. Prompt/boundary blocks keep fixed protection; generated blocks may use saved credits. Nearby or overlapping trade-off points indicate similar measured outcomes, not proven equivalence."
    ], figures=[(run/"figures/allocation_trace.png", "Figure 3. Smoothed entropy and the logged key-channel protection schedule."),
                (run/"figures/accuracy_storage.png", "Figure 4. Measured accuracy against projected effective storage. Methods may generate different lengths and finish with different unused credits.")])
    section("7. Hardware implications", [
        "The context-length projection holds batch size and the model's layer/head dimensions fixed, then applies the same group size, buffer policy and metadata format. It separates potential packed-cache capacity savings from model weights, temporary attention workspaces and allocator overhead. It therefore describes cache capacity, not total GPU memory savings.",
        "Longer contexts amortize fixed floating buffers, while metadata continues to scale with the number of blocks. Real hardware gains would additionally require integer packing, efficient dequantization and attention kernels. The prototype performs floating-point attention and Python-level control, which can add runtime overhead.",
        "A meaningful production experiment would measure end-to-end latency, throughput at equal device-memory budgets, packing/dequantization cost and maximum batch size. None of these production benefits is inferred from the fake-quantization pilot."
    ], figures=[(run/"figures/memory_projection.png", "Figure 5. Analytical packed-cache storage including metadata and floating buffers. These curves are projections, not measured GPU allocations.")])
    if profile:
        labels = {"entropy_and_margin_per_token": "Entropy + margin, per token",
                  "controller_per_block": "Credit controller, per block",
                  "uniform2_one_block_all_layers": "2-bit quantizer, one block / all layers",
                  "fixed_one_block_all_layers": "Mixed quantizer, one block / all layers"}
        sections[-1]["paragraphs"].append("An additional microbenchmark measures individual operations on real calibration-prefix cache tensors and logits. The values below are medians across repeated calls on the recorded device. Per-token and per-block costs have different frequencies. Shared CPU load affects timing, so these are descriptive operation costs, not end-to-end speedup estimates.")
        sections[-1]["table"] = [["Operation", "Median milliseconds"]] + [[labels[name], f"{entry['median_ms']:.4f}"] for name,entry in profile["operations"].items()]
        sections[-1]["widths"] = [355,150]

    diagnostic_text = ["The diagnostic isolates a generated cache block on a fixed calibration trajectory. It quantizes that block while leaving the other blocks unchanged, teacher-forces the same following token, and compares the next output distribution with the unquantized reference. The difference between uniform 2-bit distortion and a 25% boost is a local estimate of the value of extra precision."]
    if diagnostic:
        diagnostic_text.append(f"The executed diagnostic contains {diagnostic['n_blocks']} blocks from {diagnostic['n_questions']} calibration questions. Pearson correlation between entropy and KL reduction is {round(diagnostic['pearson_entropy_vs_kl_reduction'], 3) if diagnostic['pearson_entropy_vs_kl_reduction'] is not None else 'undefined'}; the corresponding raw-margin correlation is {round(diagnostic['pearson_margin_vs_kl_reduction'], 3) if diagnostic['pearson_margin_vs_kl_reduction'] is not None else 'undefined'}.")
    else:
        diagnostic_text.append("The sensitivity diagnostic has not been executed for this report's run. Its implementation is included, and no diagnostic result is asserted here.")
    diagnostic_text.append("Blocks from the same question are correlated. A small descriptive correlation does not demonstrate general predictive validity, and one-step KL reduction is not a measure of final-answer accuracy. The diagnostic is never used to tune policies on test answers.")
    section("8. Signal diagnostic", diagnostic_text,
            figures=[(run/"figures/signal_sensitivity.png", "Figure 6. Executed calibration-block sensitivity measurements.")] if diagnostic else [])
    if human:
        htable = [["Method", "Passed", "Pass@1", "95% interval"]]
        for r in human["methods"]:
            htable.append([r["mode"], f"{r['passed']}/{r['n']}", pct(r["pass_at_1"]), f"{pct(r['ci95'][0])} - {pct(r['ci95'][1])}"])
        section("9. Secondary HumanEval pilot", [
            f"A separate coding pilot evaluated {human['n_tasks']} predetermined HumanEval tasks [4] on {human['manifest']['config']['model']}. The prompt asks for a complete Python function and necessary imports. One greedy program per task is scored with the original functional tests, giving a subset pass@1 result. The maximum generation length is {human['manifest']['config']['max_new_tokens']} tokens.",
            "The coding pilot reuses the GSM8K training-set signal calibration without tuning on HumanEval. The report records this prompt and calibration choice because they differ from some standardized benchmark protocols. Programs, tests, scores and execution hashes are included for audit.",
            "This is a small coding comparison, not a non-reasoning control and not a claim about full HumanEval performance. The pilot's evaluator uses either Docker isolation or exact manually reviewed program/test hashes with resource limits, as identified in its score records. Infrastructure errors are not counted as model failures."
        ], table=htable, widths=[160,70,90,185])
        if all(r["passed"] == 0 for r in human["methods"]):
            sections[-1]["paragraphs"].append("Every tested policy passed zero tasks in this small coding slice. The observed failures include incorrect function behavior, wrong entry-point names, recursion errors and incomplete responses. This floor effect makes the coding pilot uninformative about relative retention quality; it does not establish that quantization preserves coding capability.")
    else:
        section("9. Secondary coding evaluation", [
            "HumanEval generation and isolated scoring scripts are supplied, but a scored HumanEval run was not supplied to this report generator. No coding accuracy is reported. Use the optional evaluation and then rebuild the report with --humaneval pointing to its result directory."
        ])
    section("10. Limitations and future work", [
        "The small model, selected question subset and generation cap limit statistical power and generality. Failed or truncated base-model solutions can obscure cache effects. The rapid experiment uses one allocation budget and one random-control seed. Stronger claims require a larger paired cohort, multiple budgets, sensitivity to grouping/buffers, and additional models.",
        "Uncertainty may be unrelated to future cache importance. Recently generated high-entropy text need not be the historical information that later queries require. A future policy could combine uncertainty with attention-based reuse or measured sensitivity, while accounting for the extra computation.",
        "The controller guarantees an upper allocation bound but leaves bounded terminal credit. Exact equal-budget claims require matching realized budgets or comparing carefully controlled storage/accuracy curves. Different response lengths and the fraction of tokens left floating must remain visible.",
        "The packed-memory format is a model, and its FP32 scale/offset choice affects the projection. A deployment would need a real allocator, packing rules and optimized attention. The draft/target disagreement signal inspired by SEQUOIA [2] requires another model and a speculative pipeline and remains future work.",
        "The runnable GPU configurations extend the study, but their presence does not count as an executed experiment. The report must be regenerated from actual outputs whenever the model, settings or cohort changes."
    ])
    section("11. Conclusion", [
        "RA-KVC implements a causal, credit-constrained mechanism for varying key-cache precision across generated blocks. The project separates channel selection from temporal allocation, shares the quantizer across policies, and exposes the actual precision schedule and storage assumptions.",
        outcome,
        "The measured pilot and reproducible implementation provide a concrete basis for further evaluation. They do not establish that entropy detects reasoning or that the simulation achieves real low-bit inference speedups. The next useful experiment is a larger paired GPU run with longer completion limits and tightly matched realized storage."
    ])
    section("12. Individual contributions", [
        "Our team divided the work across five technical areas."
    ], table=[["Member and roll number", "Technical responsibility"]]+[
        [name+"\n"+roll, responsibility] for name,roll,responsibility in TEAM], widths=[210,295])
    references = [
        "[1] Xia et al. Kitty: Accurate and Efficient 2-bit KV Cache Quantization with Dynamic Channel-wise Precision Boost. arXiv:2511.18643, 2025. https://arxiv.org/abs/2511.18643",
        "[2] Chen et al. Sequoia: Scalable and Robust Speculative Decoding. NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract-Conference.html",
        "[3] Cobbe et al. Training Verifiers to Solve Math Word Problems. GSM8K dataset and official repository. https://github.com/openai/grade-school-math",
        "[4] Chen et al. Evaluating Large Language Models Trained on Code. HumanEval. https://github.com/openai/human-eval",
        "[5] Qwen Team. Qwen2.5 model cards and pretrained models. https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
        "[6] Hugging Face Transformers. Version 4.46.3 source and cache interfaces. https://github.com/huggingface/transformers/tree/v4.46.3",
    ]
    section("References", references)

    title = "Reasoning-Adaptive Key-Value Cache Compression (RA-KVC)"
    subtitle = "Uncertainty-guided precision allocation under a storage budget"
    md = ["# "+title, "", subtitle, "", "DA4022 Hardware-Aware Deep Learning, IIT Guwahati", "",
          "**Evidence status:** "+summary["status"]+f"; {n} paired GSM8K questions; model "+cfg["model"]+".", "",
          "**Team:** "+"; ".join(name+" ("+roll+")" for name,roll,_ in TEAM), ""]
    for sec in sections:
        md.extend(["## "+sec["title"], ""])
        for para in sec["paragraphs"]:
            md.extend([para, ""])
        if sec["table"]:
            md.append("| "+" | ".join(sec["table"][0])+" |")
            md.append("| "+" | ".join(["---"]*len(sec["table"][0]))+" |")
            for row in sec["table"][1:]:
                md.append("| "+" | ".join(str(v).replace("\n", " ") for v in row)+" |")
            md.append("")
        for image, caption in sec["figures"]:
            if image.exists():
                import os
                relative = os.path.relpath(image, output).replace("\\", "/")
                md.extend([f"![{caption}]({relative})", ""])
    (output/"RA_KVC_Report.md").write_text("\n".join(md))

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    bold_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf")
    if font_path.exists() and bold_path.exists():
        pdfmetrics.registerFont(TTFont("ReportBody", str(font_path)))
        pdfmetrics.registerFont(TTFont("ReportBold", str(bold_path)))
        pdfmetrics.registerFontFamily("ReportBody", normal="ReportBody", bold="ReportBold")
        font, bold = "ReportBody", "ReportBold"
    else:
        font, bold = "Times-Roman", "Times-Bold"
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyRA", fontName=font, fontSize=10, leading=14, spaceAfter=9, textColor=colors.HexColor("#222222")))
    styles.add(ParagraphStyle(name="HeadingRA", fontName=bold, fontSize=15, leading=19, spaceBefore=6, spaceAfter=13, textColor=colors.HexColor("#253448")))
    styles.add(ParagraphStyle(name="TitleRA", fontName=bold, fontSize=25, leading=31, spaceAfter=16, textColor=colors.HexColor("#253448")))
    styles.add(ParagraphStyle(name="SmallRA", fontName=font, fontSize=8, leading=11, spaceAfter=8, textColor=colors.HexColor("#535861")))
    styles.add(ParagraphStyle(name="CellRA", fontName=font, fontSize=8, leading=11, textColor=colors.HexColor("#222222")))
    styles.add(ParagraphStyle(name="HeaderRA", fontName=bold, fontSize=8, leading=11, textColor=colors.white))
    escape = lambda text: html.escape(str(text)).replace("\n", "<br/>")
    story = [Spacer(1, 45), Paragraph(title, styles["TitleRA"]), Paragraph(subtitle, styles["HeadingRA"]),
             Spacer(1, 8), Paragraph("DA4022 &nbsp;|&nbsp; Hardware-Aware Deep Learning<br/>Indian Institute of Technology Guwahati", styles["BodyRA"]), Spacer(1, 15)]
    for name, roll, _ in TEAM:
        story.append(Paragraph(escape(name+"   "+roll), styles["BodyRA"]))
    story.append(PageBreak())
    for index, sec in enumerate(sections):
        if index and sec["title"] not in ("References", "11. Conclusion", "1. Introduction and research question"):
            story.append(PageBreak())
        story.append(Paragraph(escape(sec["title"]), styles["HeadingRA"]))
        for para in sec["paragraphs"]:
            story.append(Paragraph(escape(para), styles["BodyRA"]))
        if sec["table"]:
            cells = [[Paragraph(escape(v), styles["HeaderRA"] if r==0 else styles["CellRA"]) for v in row] for r,row in enumerate(sec["table"])]
            widths = [v*0.98 for v in sec["widths"]] if sec["widths"] else None
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#253448")),
                                       ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F2F3F4"), colors.white]),
                                       ("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0),(-1,-1),7),
                                       ("BOTTOMPADDING",(0,0),(-1,-1),7), ("LEFTPADDING",(0,0),(-1,-1),7),
                                       ("LINEBELOW",(0,0),(-1,0),0.5,colors.HexColor("#253448"))]))
            story.extend([table, Spacer(1, 13)])
        for image, caption in sec["figures"]:
            if image.exists():
                from PIL import Image as PILImage
                with PILImage.open(image) as img:
                    iw, ih = img.size
                width = 420 if sec["title"].startswith("6.1.") else 415 if sec["title"] == "7. Hardware implications" else 460
                height = width*ih/iw
                story.append(KeepTogether([Image(str(image), width=width, height=height),
                                           Paragraph(escape(caption), styles["SmallRA"])]))
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#AEB2B8")); canvas.line(48, 37, A4[0]-48, 37)
        canvas.setFont(font, 8); canvas.setFillColor(colors.HexColor("#535861"))
        canvas.drawString(48, 24, "RA-KVC   |   DA4022 Hardware-Aware Deep Learning")
        canvas.drawRightString(A4[0]-48, 24, str(doc.page)); canvas.restoreState()
    doc = SimpleDocTemplate(str(output/"RA_KVC_Report.pdf"), pagesize=A4,
                            rightMargin=48, leftMargin=48, topMargin=44, bottomMargin=50,
                            title=title, author="DA4022 project team")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    import os
    (output/"report_data.json").write_text(json.dumps({"source_run": os.path.relpath(run,ROOT).replace("\\", "/"), "summary": summary, "diagnostic": diagnostic, "overhead_profile": profile, "humaneval": human, "length_check": length_check, "sections": [
        {**sec, "figures": [[str(path),caption] for path,caption in sec["figures"]]} for sec in sections]}, indent=2))
    print(f"Wrote {output/'RA_KVC_Report.pdf'} and editable Markdown")


if __name__ == "__main__":
    main()
