"""Depth-routed Qwen3 variants: AR / DAR / GDAR + the comparison baselines.

Each variant keeps the Qwen3 backbone (token embedding, RoPE, attention, MLP,
RMSNorm placement, LM head) and replaces only the *connection module* -- how
information travels along the layer axis.  That is what makes the comparisons in
the redo plan apples-to-apples: same tokenizer, same data, same hyper-parameters,
one swapped module.

===========  ==========================================  =====================
Variant      routing sources                             stream update
===========  ==========================================  =====================
AR           cumulative block / sublayer states          ``partial + out``
DAR          per-sublayer deltas (or block deltas)       ``partial + out``
GDAR         per-sublayer deltas (or block deltas)       gated delta rule
Gated-AR     cumulative block states                     gated delta rule
===========  ==========================================  =====================

Baselines (reviewer-mandated, see tmp/EXTRACT_NOTES.md §5): ``hc`` / ``mhc``
(Hyper-Connections, manifold-constrained HC -- ported from megatron-core 0.18.2
``hyper_connection.py``), ``mudd`` (MUDDFormer) and ``denseformer`` (DenseFormer),
each reproduced from its official implementation (see the file headers for what was
and was not cross-checked).

AR x {gates off, gates on} vs DAR x {gates off, gates on} is the 2x2 factorial
that the redo plan calls the scientific core: it separates the contribution of
the *source* (cumulative vs delta) from the contribution of the *gates*.

Usage
-----
>>> from models import Qwen3DARConfig, Qwen3DARForCausalLM
>>> cfg = Qwen3DARConfig(
...     vocab_size=32000,
...     hidden_size=1024,
...     num_hidden_layers=28,
...     num_attention_heads=16,
...     num_key_value_heads=8,
...     attnres_mode="delta_block",
...     attnres_num_blocks=8,
... )
>>> model = Qwen3DARForCausalLM(cfg)

Every variant also registers its ``model_type`` with the HF auto classes, so
``AutoModelForCausalLM.from_config(cfg)`` and ``from_pretrained`` work once the
module has been imported.
"""

from .modeling_qwen3_ar import (
    Qwen3ARConfig,
    Qwen3ARDecoderLayer,
    Qwen3ARForCausalLM,
    Qwen3ARModel,
)
from .modeling_qwen3_dar import (
    Qwen3DARConfig,
    Qwen3DARDecoderLayer,
    Qwen3DARForCausalLM,
    Qwen3DARModel,
)
from .modeling_qwen3_gdar import (
    Qwen3GDARConfig,
    Qwen3GDARDecoderLayer,
    Qwen3GDARForCausalLM,
    Qwen3GDARModel,
)

from .modeling_qwen3_hc import Qwen3HCConfig, Qwen3HCForCausalLM
from .modeling_qwen3_mhc import Qwen3MHCConfig, Qwen3MHCForCausalLM
from .modeling_qwen3_mudd import Qwen3MUDDConfig, Qwen3MUDDForCausalLM
from .modeling_qwen3_denseformer import Qwen3DenseFormerConfig, Qwen3DenseFormerForCausalLM

__all__ = [
    "Qwen3ARDecoderLayer",
    "Qwen3DARDecoderLayer",
    "Qwen3GDARDecoderLayer",
    "Qwen3ARConfig",
    "Qwen3ARModel",
    "Qwen3ARForCausalLM",
    "Qwen3DARConfig",
    "Qwen3DARModel",
    "Qwen3DARForCausalLM",
    "Qwen3GDARConfig",
    "Qwen3GDARModel",
    "Qwen3GDARForCausalLM",
    "Qwen3HCConfig",
    "Qwen3HCForCausalLM",
    "Qwen3MHCConfig",
    "Qwen3MHCForCausalLM",
    "Qwen3MUDDConfig",
    "Qwen3MUDDForCausalLM",
    "Qwen3DenseFormerConfig",
    "Qwen3DenseFormerForCausalLM",
]
