"""The Triton gate kernel against routing.gates(), on an NVIDIA GPU or on the
CPU under Triton's interpreter (TRITON_INTERPRET=1); skipped without
either."""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if torch.cuda.is_available():
    DEVICE = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEVICE = "cpu"
else:
    pytest.skip("needs CUDA or TRITON_INTERPRET=1", allow_module_level=True)

from weightpack_vllm import routing_kernel  # noqa: E402
from weightpack_vllm.routing import Routing, attach, gate_op, gates  # noqa: E402

SLOTS = 4


def state(g, n_slices, n_exp, in_f, rank, ties=False):
    """A routing state with a mix in slots 0, 2 and 3 (top_k 1, 2 and more
    than the experts that touch), slot 1 holding a plain adapter; some
    experts touch no slice, and one slice is touched by none in slot 3."""
    r = Routing(n_slices, SLOTS, rank, in_f, n_exp, DEVICE)
    r.protos.copy_(torch.randn(r.protos.shape, generator=g, device=DEVICE))
    if ties:  # same prototypes: every token ties, top_k go to the lowest
        r.protos.copy_(r.protos[:, :, :1].expand_as(r.protos))
    r.bias.copy_(torch.randn(r.bias.shape, generator=g, device=DEVICE) * 0.1)
    r.touch.copy_(torch.rand(r.touch.shape, generator=g, device=DEVICE) > 0.3)
    r.touch[:, :, 0] = True
    r.touch[-1, 3] = False
    owner = torch.randint(0, n_exp, (SLOTS, rank), generator=g, device=DEVICE)
    r.rank_expert.copy_(owner)
    r.top_k.copy_(torch.tensor([1, 1, 2, n_exp + 1], device=DEVICE))
    r.is_mix.copy_(torch.tensor([True, False, True, True], device=DEVICE))
    return r


DTYPES = [torch.float32, torch.float16] + ([torch.bfloat16] if DEVICE == "cuda" else [])
SIZES = [(1, 2, 16, 8), (3, 8, 100, 24), (2, 5, 1500, 70)]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("n_slices,n_exp,in_f,rank", SIZES)
def test_kernel_scales_as_gates(dtype, ties, n_slices, n_exp, in_f, rank):
    g = torch.Generator(device=DEVICE).manual_seed(0)
    r = state(g, n_slices, n_exp, in_f, rank, ties)
    t = 37
    x = torch.randn(t, in_f, generator=g, device=DEVICE).to(dtype)
    slots = torch.randint(-1, SLOTS, (t,), generator=g, device=DEVICE)
    buf = torch.randn(n_slices, t, rank, generator=g, device=DEVICE)
    want = buf * gates(x, slots, *r.tensors())
    got = buf.clone()
    routing_kernel.gate_(got, x, slots, *r.tensors())
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
    # Tokens of no mix are left exactly as they were.
    plain = (slots < 0) | ~r.is_mix[slots.clamp(min=0)]
    assert torch.equal(got[:, plain], buf[:, plain])


def test_op_gates_only_a_batch_with_adapters_and_routing():
    g = torch.Generator(device=DEVICE).manual_seed(1)
    r = state(g, 2, 4, 32, 8)
    x = torch.randn(5, 32, generator=g, device=DEVICE)
    slots = torch.zeros(5, dtype=torch.long, device=DEVICE)
    buf = torch.randn(2, 5, 8, generator=g, device=DEVICE)
    lora_a = torch.zeros(SLOTS, 1, 8, 32, device=DEVICE)
    got = buf.clone()
    gate_op()(got, x, slots, lora_a, torch.tensor([False]))
    assert torch.equal(got, buf)  # no routing attached to lora_a
    attach(lora_a, r)
    gate_op()(got, x, slots, lora_a, torch.tensor([True]))
    assert torch.equal(got, buf)
    gate_op()(got, x, slots, lora_a, torch.tensor([False]))
    torch.testing.assert_close(got, buf * gates(x, slots, *r.tensors()))
