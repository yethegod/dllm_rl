"""
Apply site-packages patches required to run SDAR evaluation with lmdeploy==0.12.3
on PyTorch 2.6 (xgrammar==0.2.0 ABI mismatch workaround + SDARConfig fix).

Run once after setting up the conda environment:
    conda activate opencompass
    python patch_env.py
"""

import importlib.util
import site
import sys
from pathlib import Path


def find_file(relative: str) -> Path:
    for sp in site.getsitepackages():
        p = Path(sp) / relative
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {relative} in site-packages")


# ---------------------------------------------------------------------------
# Patch 1 & 2 & 3: lmdeploy/pytorch/engine/guided_process.py
#   - make xgrammar a soft import (ABI mismatch with torch 2.6)
#   - add `from __future__ import annotations` to defer type annotation eval
#   - raise ValueError in __init__ when xgrammar unavailable (agent.py catches it)
# ---------------------------------------------------------------------------
GUIDED = find_file("lmdeploy/pytorch/engine/guided_process.py")
original = GUIDED.read_text()

patched = original

# Patch 1: from __future__ + soft xgrammar import
OLD_IMPORT = "import torch\nimport xgrammar as xgr\nfrom transformers import PreTrainedTokenizerBase"
NEW_IMPORT = (
    "from __future__ import annotations\n"
    "import torch\n"
    "from transformers import PreTrainedTokenizerBase\n"
    "\n"
    "try:\n"
    "    import xgrammar as xgr\n"
    "    _XGRAMMAR_AVAILABLE = True\n"
    "except Exception:\n"
    "    xgr = None\n"
    "    _XGRAMMAR_AVAILABLE = False"
)
if OLD_IMPORT in patched:
    patched = patched.replace(OLD_IMPORT, NEW_IMPORT)
    print("  [OK] guided_process.py: soft xgrammar import + __future__ annotations")
elif "_XGRAMMAR_AVAILABLE" in patched:
    print("  [SKIP] guided_process.py: xgrammar patch already applied")
else:
    print("  [WARN] guided_process.py: unexpected content, skipping xgrammar patch")

# Patch 3: no-op __init__ when xgrammar unavailable
OLD_INIT = (
    "    def __init__(self, tokenizer: PreTrainedTokenizerBase, vocab_size: int | None):\n"
    "        if vocab_size is None:\n"
    "            vocab_size = tokenizer.vocab_size\n"
    "\n"
    "        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)\n"
    "        self.compiler = xgr.GrammarCompiler(tokenizer_info)\n"
    "        self.vocab_size = vocab_size"
)
NEW_INIT = (
    "    def __init__(self, tokenizer: PreTrainedTokenizerBase, vocab_size: int | None):\n"
    "        if not _XGRAMMAR_AVAILABLE:\n"
    "            raise ValueError('xgrammar is not available; guided decoding disabled')\n"
    "        if vocab_size is None:\n"
    "            vocab_size = tokenizer.vocab_size\n"
    "\n"
    "        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)\n"
    "        self.compiler = xgr.GrammarCompiler(tokenizer_info)\n"
    "        self.vocab_size = vocab_size"
)
if OLD_INIT in patched:
    patched = patched.replace(OLD_INIT, NEW_INIT)
    print("  [OK] guided_process.py: GuidedDecodingManager.__init__ guard")
elif "_XGRAMMAR_AVAILABLE" in patched and "raise ValueError" in patched:
    print("  [SKIP] guided_process.py: __init__ guard already applied")
else:
    print("  [WARN] guided_process.py: unexpected __init__ content, skipping guard patch")

if patched != original:
    GUIDED.write_text(patched)

# ---------------------------------------------------------------------------
# Patch 4: lmdeploy/pytorch/models/sdar.py
#   SDARConfig has no pad_token_id — use getattr with None fallback
# ---------------------------------------------------------------------------
SDAR = find_file("lmdeploy/pytorch/models/sdar.py")
sdar_src = SDAR.read_text()

OLD_PAD = "        self.padding_idx = config.pad_token_id"
NEW_PAD = "        self.padding_idx = getattr(config, 'pad_token_id', None)"

if OLD_PAD in sdar_src:
    SDAR.write_text(sdar_src.replace(OLD_PAD, NEW_PAD))
    print("  [OK] sdar.py: pad_token_id getattr fallback")
elif NEW_PAD in sdar_src:
    print("  [SKIP] sdar.py: pad_token_id patch already applied")
else:
    print("  [WARN] sdar.py: unexpected content, skipping pad_token_id patch")

# ---------------------------------------------------------------------------
# Verify: lmdeploy imports cleanly
# ---------------------------------------------------------------------------
print("\nVerifying lmdeploy import...")
try:
    from lmdeploy import pipeline, PytorchEngineConfig, GenerationConfig  # noqa: F401
    print("  [OK] lmdeploy imports successfully")
except Exception as e:
    print(f"  [FAIL] lmdeploy import error: {e}")
    sys.exit(1)

print("\nAll patches applied. Ready to run evaluation.")
