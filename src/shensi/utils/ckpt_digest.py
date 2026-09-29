# Copyright (c) 2026 FlagOS Contributors
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# limitations under the License.


import hashlib
from collections.abc import Iterable

import torch

EXTRA_STATE_SUFFIX = "_extra_state"


def tensor_bytes(t: torch.Tensor) -> bytes:
    if t.device.type != "cpu":
        t = t.detach().to("cpu")
    t = t.detach().contiguous()
    return t.view(torch.uint8).reshape(-1).numpy().tobytes()


def weight_keys(sd: dict[str, torch.Tensor]) -> list[str]:
    return sorted(k for k in sd if not k.endswith(EXTRA_STATE_SUFFIX))


def digest_state_dict(
    sd: dict[str, torch.Tensor], keys: Iterable[str] | None = None
) -> tuple[str, int]:
    ks = weight_keys(sd) if keys is None else list(keys)
    h = hashlib.sha256()
    for k in ks:
        v = sd[k]
        h.update(k.encode("utf-8"))
        h.update(str(tuple(v.shape)).encode("utf-8"))
        h.update(str(v.dtype).encode("utf-8"))
        h.update(tensor_bytes(v))
    return h.hexdigest(), len(ks)


def tensor_digest(sd: dict[str, torch.Tensor], key: str) -> str:
    v = sd[key]
    h = hashlib.sha256()
    h.update(str(tuple(v.shape)).encode("utf-8"))
    h.update(str(v.dtype).encode("utf-8"))
    h.update(tensor_bytes(v))
    return h.hexdigest()


def sample_keys(sd: dict[str, torch.Tensor], n: int = 6) -> list[str]:
    ks = weight_keys(sd)
    want = [
        "embedding.word_embeddings.weight",
        "decoder.layers.0.self_attention.linear_q_down_proj.weight",
        "lm_head.weight" if "lm_head.weight" in sd else "output_layer.weight",
        "output_norm.weight",
    ]
    picked = [k for k in want if k in sd]
    for k in ks:
        if len(picked) >= n:
            break
        if "router.weight" in k or "fc1_latent_proj" in k or ".experts." in k or "mlp." in k:
            if k not in picked:
                picked.append(k)
    return picked[:n]


def summarize(sd: dict[str, torch.Tensor]) -> dict[str, object]:
    import collections
    digest, n = digest_state_dict(sd)
    return {
        "sha256": digest,
        "num_weight_tensors": n,
        "num_extra_state_tensors": len(sd) - n,
        "non_tensor_keys": sorted(k for k, v in sd.items() if not torch.is_tensor(v)),
        "dtypes": dict(
            collections.Counter(str(v.dtype) for v in sd.values() if torch.is_tensor(v))
        ),
    }


def load_mcore_torch_ckpt(path: str) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)
