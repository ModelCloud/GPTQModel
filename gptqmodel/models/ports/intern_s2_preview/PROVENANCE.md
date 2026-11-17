# Intern-S2 Preview source provenance

The package owns this implementation of `intern_s2_preview`. Transformers
5.23.0 has neither this config nor an exact `AutoModelForImageTextToText`
mapping. The model is not substituted with another architecture.

Upstream: [internlm/Intern-S2-Preview](https://huggingface.co/internlm/Intern-S2-Preview/tree/4f57cab513689b089019fce4ad24e26520df183c),
pinned revision `4f57cab513689b089019fce4ad24e26520df183c`.
All original copyright notices are retained; the unmodified Apache-2.0
license is in `LICENSE`. The model/config/processor identify the Qwen Team
and Hugging Face; the tokenizer identifies the Intern team and Shanghai AI Lab.

SHA256 of the original downloaded files, before adaptation:

| File | SHA256 |
| --- | --- |
| configuration_interns2_preview.py | `08fd7412817b06a0973f820e125863af9c8348a4a86f1659dca8ab243e54a83c` |
| modeling_interns2_preview.py | `57e8d975b5d5f3c03dde3cc403e54b622104708371834f76c931d23a765edf76` |
| processing_interns2_preview.py | `5ff8733f200719da26090320fa4889d321f824af6a20ee9982ffb1160b0c2a13` |
| tokenization_interns1.py | `ad0ecddabf936ad205382ebc7dc87755eb711c2833ebd590179188a7fb2716ac` |
| config.json | `14943b1574b2467db8e1e2474d9f84a0e09e67ba296f9e7a14de877de8f84767` |
| model.safetensors.index.json | `dca0c926deca75095f377bc54fb1c88e8775252fd3767d6c8dda25d7c4194ac1` |
| LICENSE | `68b1e6dfad75985fd85f282872caab48bb1b2bcc90fd4226ba665a3a9ef09a09` |

The three generated-source banners were replaced with adaptation notices;
the absent upstream modular generator is not part of this package.
Imports between these files are package relative. Adaptations:

- Use the Transformers 5 `create_causal_mask` and cache query-offset/length
  interfaces. The hybrid cache owns its token count, supports both all-linear
  and all-full layouts, and carries convolution/recurrent state through
  multi-token continuation. Disable GenerationMixin's generic DynamicCache
  allocation for this custom state representation.
- Select the upstream portable Torch convolution, RMSNorm-gating and delta
  kernels explicitly, with no import-time CUDA allocation or FLA dependency.
  CUDA/FLA acceleration is not enabled or qualified by this port.
- Preserve checkpoint parameter names, packed MoE experts and computation;
  normal Defuser conversion expands the packed experts for quantization.
  The released index has 2,098 entries, including 785 MTP entries intentionally
  excluded by the existing definition. The 1,313 remaining keys match the
  package-owned meta model exactly. `checkpoint_ownership.json` pins the
  independent index audit, and `upstream_config.json` pins the released config.
- Preserve the time-series encoder's model/input dtype rather than forcing
  BF16. Round positional sums into the activation dtype even when a persistent
  checkpoint buffer remains FP32. Load optional soundfile/pandas only when
  reading their input formats.
- The package-owned processor selects native Qwen image/video components
  directly; JSON cannot choose component Python. Tokenizer normalization is
  delegated to Tokenicer through the common GPT-QModel loader.
- Keep the **S2** `InternS1Tokenizer`, its byte BPE, domain offsets, transient
  SMILES/PROT/XNA auto-detection markers, protection markers, NFC handling and
  domain decode semantics. Import the Transformers 5 Python tokenizer
  directly, import optional RDKit only when validating SMILES, avoid changing
  RDKit's process-global logger, and return all five saved vocabulary assets.
  Without RDKit, upstream's rule-based SMILES validation remains available.

Despite the same Python class name, the S1-mini tokenizer is not equivalent:
it uses IUPAC/FASTA models whereas this tokenizer uses PROT/XNA. Dispatch must
use this definition's complete package path, never the checkpoint class name.
The S2 tokenizer needs these data assets from the pinned checkpoint:

| Asset | SHA256 |
| --- | --- |
| vocab.json | `657191b77e1d51b28521773221d4121bfe4844900147f7818499e0347d8c66f6` |
| merges.txt | `3bd640ba6d8da8f5844f3548b7e2184fc664dd670e1447e425a48dbaaad1ef43` |
| tokenizer_SMILES.model | `fba1c97da0353ccbffd368ae78e311ccbc762aa5ba74f9aff8bf2ab363c4d37d` |
| tokenizer_PROT.model | `1144f52f86f3ca5a29940d69b037e508c05a89e6eedbe42bea641e226b20dbe0` |
| tokenizer_XNA.model | `58fc8bfb2af3dfe936a13dad8a9cb28dab7850b70b358db19605d867c133fb35` |

Those assets and the trained model weights are not vendored. Offline tests
build real byte BPE and SentencePiece fixtures with distinct domain offsets.
The released tokenizer assets were separately loaded and checked directly:
`Hello world -> [9419, 1814]`, `<SMILES>CCO</SMILES> ->
[248094, 248268, 248095]`, and tagged/automatic protein and DNA sequences
decoded exactly. See `README.md` for execution coverage and support limits.
The real pinned image/video processor metadata was also loaded without
checkpoint Python; image/video tensors and five-asset processor save/reload
preserved the above IDs.
