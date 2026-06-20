import time
import torch

from wan_5b.modules.t5 import umt5_xxl

print("=== device='meta' + load_state_dict(assign=True) ===")

t0 = time.time()
model = umt5_xxl(
    encoder_only=True,
    return_tokenizer=False,
    dtype=torch.bfloat16,
    device=torch.device('meta'),
).eval().requires_grad_(False)
print(f"umt5_xxl() construction on meta: {time.time()-t0:.1f}s")

t0 = time.time()
state_dict = torch.load(
    "wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    map_location='cpu', weights_only=False, mmap=True,
)
print(f"torch.load(mmap=True): {time.time()-t0:.1f}s")

t0 = time.time()
missing, unexpected = model.load_state_dict(state_dict, assign=True, strict=True)
print(f"load_state_dict(assign=True): {time.time()-t0:.1f}s missing={missing} unexpected={unexpected}")

# Confirm no meta tensors survive the assign (would silently break forward()).
meta_params = [n for n, p in model.named_parameters() if p.is_meta]
print(f"meta params remaining after assign: {len(meta_params)} {meta_params[:5]}")

t0 = time.time()
model = model.cuda()
torch.cuda.synchronize()
print(f".cuda(): {time.time()-t0:.1f}s")
print(f"sample param dtype/device after .cuda(): {next(model.parameters()).dtype} {next(model.parameters()).device}")

print("\nDone.")
