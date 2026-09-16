# Uniform W3 P32 YAQA/GSQ validation protocol

- Source model: `/monster/data/model/Llama-3.2-1B-Instruct`.
- QVQ source: detached `origin/main` at `f23ded027e476fd8a19a07fbc00f3f36609b2135`.
- ZML-ultra source: detached `origin/master` at `459a7827b07d094cf57d5e26ae349a5c8138636d` (`origin/main` does not exist).
- Evalution source: `origin/main` at `8cbed6ec0fdd09e567f6a35a5093b40258d9c10d` (package version 0.0.17).
- Both quantization arms: uniform W3, `qvq_v2b2_p32`, two banks, YAQA, seed 0,
  regularization 0.02, and the same frozen 10,178-row YAQA/NM Fisher stream.
- Lifecycle calibration: the same frozen NM rows `[0,128)` in both arms.
- Treatment: GSQ enabled for every supported projection, seed 7, 33 legal whole-tile
  candidates, at most 100 Lion steps with one paper-style Gumbel draw and a
  10-step no-hard-improvement stop, fixed scales, and a 4-GiB decoded-candidate cap.
- Evaluation: all 1,209 `madrylab/gsm8k-platinum` test rows through Evalution and
  the ZML native P32 runner, with deterministic greedy generation.
- Contamination policy: quantization/calibration rows are audited using normalized
  user-turn hashes. GSM8K-Platinum is evaluation-only and is never exposed to QVQ,
  YAQA, GSQ candidate selection, or configuration selection.
