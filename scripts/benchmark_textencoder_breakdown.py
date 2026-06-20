import time
import torch

from wan_5b.modules.t5 import umt5_xxl

print("=== Isolating WanTextEncoder's internal sub-steps ===")

t0 = time.time()
model = umt5_xxl(
    encoder_only=True,
    return_tokenizer=False,
    dtype=torch.bfloat16,
    device=torch.device('cpu'),
).eval().requires_grad_(False)
print(f"umt5_xxl() construction (random init, CPU, bf16): {time.time()-t0:.1f}s")

t0 = time.time()
state_dict = torch.load(
    "wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
    map_location='cpu', weights_only=False, mmap=True,
)
print(f"torch.load(mmap=True): {time.time()-t0:.1f}s")

t0 = time.time()
model.load_state_dict(state_dict)
print(f"load_state_dict (materializes mmap'd tensors): {time.time()-t0:.1f}s")

t0 = time.time()
model = model.cuda()
torch.cuda.synchronize()
print(f".cuda(): {time.time()-t0:.1f}s")

print("\nDone.")
