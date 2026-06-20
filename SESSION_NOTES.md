# Notas de sesión — LongLive-2.0 en RunPod (RTX A6000)

Este documento resume el trabajo hecho sobre este fork de LongLive en una sesión larga de Claude Code, para que cualquier sesión nueva (en esta VPS o en otra) tenga contexto completo sin tener que re-investigar desde cero. Está organizado de lo más importante (qué cambió y por qué) a lo más detallado (números exactos, archivos).

## Entorno

- GPU: 1x RTX A6000 (Ampere, sm_86, ~49GB).
- Venv: `/opt/longlive_venv` (en disco local `/opt`, no en `/workspace` que es un mount de red con cuota oculta).
- Pesos del modelo: `wan_models/`, `checkpoints/` son symlinks a `/opt/longlive_models/` (mismo motivo: evitar el mount de red).
- `torch==2.8.0+cu128`. **`torchao` está instalado pero sus extensiones C++ no cargan** (requieren `torch>=2.11.0`) — relevante si en el futuro se quiere retomar cuantización.
- `origin` de este repo apunta a `https://github.com/NVlabs/LongLive.git` (el repo oficial) — **nunca se hace push ahí**. Los commits de esta sesión son locales; el respaldo real se sincroniza a mano (copiar archivos + commit nuevo, no merge) hacia `jesustorres-code/vps-python-env` en GitHub (repo público, mismo dueño que esta VPS).

## 1. Optimizaciones de tiempo de carga (109.1s → 12.3s, ~89%)

Tres fixes con el mismo patrón: construir el módulo en `device='meta'` (sin memoria real ni init aleatorio) y luego `load_state_dict(..., assign=True)` para que las tensores del checkpoint **reemplacen** los parámetros en vez de copiarse encima de un init que se iba a descartar de todos modos.

| Fix | Archivo | Antes | Después |
|---|---|---|---|
| `WanTextEncoder` (UMT5-XXL) construido en meta + `mmap=True` en el `torch.load` | `utils/wan_5b_wrapper.py` | 74.3s (pipeline build) | 5.6s |
| `load_generator_checkpoint` con `mmap=True` + `assign=True` | `utils/inference_utils.py` | 12.5s perdidos en `copy_` sobre pesos de `from_pretrained` que se iban a tirar | ~0.1s |

`CausalWanModel.from_pretrained()` (diffusers `ModelMixin`) ya hacía este truco internamente — no era el cuello de botella (se verificó: `from_pretrained`=0.6s vs `from_config`=43.0s).

## 2. Optimizaciones investigadas y DESCARTADAS (con números)

No las reabras sin una razón nueva — ya se midieron y no valen la pena en este hardware/setup:

- **`torch.compile`**: descartado en la parte temprana de la sesión (recompilaciones por shapes dinámicos entre bloques causales, costo no se recupera para generación de un solo request a la vez).
- **`cudnn.benchmark=True` + TF32`** (`benchmark_cudnn_flags.py`): primer call paga +9.8s de autotune (38.62s vs 28.82s baseline) por solo ~1.1s/call de ganancia en steady-state — necesitarías ~9 llamadas seguidas para amortizar, no aplica a un Gradio de un usuario a la vez.
- **`channels_last_3d` en el VAE estándar** (`benchmark_channels_last.py`): 58.4s (contiguo) vs 60.0s (channels_last) — **más lento**, no más rápido. Probablemente el decoder mezcla `Conv3d` con capas (GroupNorm/atención) que no soportan ese formato y fuerzan reconversiones. Además el max abs diff (0.14 en escala [0,1]) no era solo ruido de redondeo.
- **Cuantización INT8 de las capas Linear**: `torch._int_mm` (el kernel INT8 nativo de PyTorch en este torch/CUDA) resultó **~2x más lento** que el matmul BF16 ya optimizado de cuBLAS, medido directamente con las shapes reales del transformer (`dim=3072, ffn_dim=14336`, M=7040 tokens/bloque). `torchao` tiene kernels mejores pero sus extensiones no cargan con torch 2.8. Subir de versión de torch es un cambio de dependencias pesado para una ganancia incierta — no se intentó.
- **mg_lightvae v1 vs v2** (`benchmark_lightvae_v1_vs_v2.py`, resultados en `videos/lightvae_compare/v1_vs_v2_summary.json`): v1 (`MG-LightVAE.pth`, pruning_rate=0.5) tiene calidad **prácticamente idéntica** a v2 (`MG-LightVAE_v2.pth`, pruning_rate=0.75) — PSNR_Y 15.61 vs 15.62 dB, SSIM_Y 0.7111 vs 0.7115 — pero v1 es 2.6x más lento decodificando (23.5s vs 9.1s para 64 frames latentes). **v1 se descartó, v2 se mantiene como está.** (Nota: estos números de calidad, medidos comparando tensores en memoria con `skimage`, son más bajos que una medición manual anterior con ffmpeg de ~18.7dB/SSIM_Y~0.787 — esa medición vieja tenía un bug de desalineamiento de frames a mitad de archivo y estaba inflada.)

Scripts de estas investigaciones quedaron en el repo (`benchmark_*.py`, `diagnose_vae_encode_temporal.py`) por si hace falta repetir alguna medición.

## 3. UI de Gradio (`app.py`) — 3 pestañas

### Pestaña "Generación simple" (original, sin cambios de comportamiento)
Un prompt, slider de frames (32-128, múltiplos de 8), sampling steps, guidance scale, seed, switch Calidad/Rápido/Turbo. `decode_mode` controla qué VAE decodifica: estándar Wan2.2 (Calidad, ~60s/64 frames latentes), `mg_lightvae_v2` (Rápido/Turbo, ~9s).

**Dato importante descubierto al verificar con `ffprobe`**: el slider de "frames" es en **frames latentes**, no en frames de píxel del video final. El VAE tiene compresión/expansión temporal: con decode estándar, 96 frames latentes dieron **366 frames de píxel** (15.25s a 24fps), no 96 frames/4s como uno asumiría ingenuamente. La función `save_video(..., fps=24)` no cambia esto — el conteo de frames de salida lo decide el decode del VAE, no el slider.

### Pestaña "Multi-escena (beta)"
Permite definir varias escenas con prompts distintos en un solo video, formato `segundos :: prompt` (una escena por línea). **Esto usa un mecanismo nativo del modelo que NVIDIA ya entrena/soporta**, no algo inventado en esta sesión:

- El pipeline (`pipeline/causal_diffusion_inference.py`) acepta un prompt **por bloque de 8 frames**, no uno solo para todo el video.
- Si el prompt de un bloque empieza con el prefijo `"The scene transitions. "` (`scene_cut_prefix`, default en `utils/dataset.py::DEFAULT_SCENE_CUT_PREFIX`), el pipeline lo detecta como corte de escena (`_is_shot_boundary`).
- `configs/inference.yaml` ya tiene `multi_shot_sink: true` — al detectar el corte, el pipeline migra el "sink" de atención (el ancla permanente de la ventana deslizante) del primer frame de la escena vieja al primer frame de la nueva. Confirmado en logs reales: `[inference] Scene cut at chunk 6, pinning chunk as shot-sink`.
- `app.py::_parse_scenes`/`_build_block_prompts` parsean el texto y aplican el prefijo automáticamente — el usuario nunca escribe el prefijo a mano.
- `utils/inference_utils.py::prepare_multi_scene_inputs` (nueva) construye el ruido + lista de prompts ya expandida por bloque.

**Limitación real, no un bug**: el texto NO es acumulativo entre bloques (cada bloque solo ve su propio prompt vía cross-attention) y el corte de escena resetea la memoria visual a propósito (es para escenas genuinamente distintas, no para mantener un personaje idéntico). Si quieres que un objeto/personaje se vea igual entre escenas, hay que repetir sus detalles de identidad (color, forma) en CADA prompt de escena — el modelo no "recuerda" texto de bloques anteriores.

### Pestaña "Imagen + formato (beta)"
Dos features:

**a) Formato 16:9 / 9:16.** El modelo (`Wan2.2-TI2V-5B`) está oficialmente entrenado en ambas orientaciones (`wan_5b/configs/__init__.py:38`: `SUPPORTED_SIZES = {'ti2v-5B': ('704*1280', '1280*704')}`). RoPE, `frame_seq_length`, `local_attn_size` y `sink_size` dependen solo del PRODUCTO H×W, nunca del orden — así que basta intercambiar H/W del shape (`PORTRAIT_SHAPE = FULL_SHAPE[:3] + [FULL_SHAPE[4], FULL_SHAPE[3]]`) sin tocar el pipeline. Verificado con `ffprobe`: sale exactamente 704×1280 o 1280×704 según se pida.

**b) Imagen de referencia (image-to-video / i2v).** `pipe.vae.encode_to_latent()` convierte una imagen en píxeles a latente, y se pasa como `initial_latent` a `pipe.inference()`. El pipeline lo usa como contexto "congelado" (timestep=0) para el primer bloque, antes de empezar a denoisear el resto. **Tres bugs reales encontrados y corregidos durante las pruebas** (no documentados por NVIDIA, hubo que descubrirlos a prueba y error):

1. `assert num_input_frames % num_frame_per_block == 0` en `pipeline/causal_diffusion_inference.py:408` — con `independent_first_frame=False` (default de este config), **no acepta 1 solo frame de referencia**, necesita un bloque completo (múltiplo de 8 frames latentes). Fix: repetir la imagen en píxeles antes de codificarla.
2. El VAE comprime temporalmente al codificar: confirmado empíricamente (`diagnose_vae_encode_temporal.py`) que `latente_T = (píxel_T - 1) // 4 + 1` (primer frame latente es un caso especial causal). Para conseguir exactamente 8 frames latentes (1 bloque) hace falta repetir la imagen **32 veces** en píxeles, no 8 — `REFERENCE_IMAGE_PIXEL_FRAMES = 32` en `app.py`.
3. `encode_to_latent()` devuelve siempre `float32` (cast interno), pero el generador espera `bf16` — hay que castear el resultado.
4. **(Bug en código de NVIDIA, no de esta sesión — ya corregido)** Al inyectar el bloque de imagen de referencia (`pipeline/causal_diffusion_inference.py`, líneas ~387 y ~417), el código usaba el `conditional_dict` **completo sin recortar** (con batch = número total de bloques del video, ej. 7) en vez de la versión correctamente recortada por bloque (`conditional_dict_list[block_index]`, batch=1). Esto rompía la atención cruzada cuando el número de bloques no dividía exactamente el largo de secuencia por bloque (7040 tokens a resolución completa) — ej. con 7 bloques (56 frames) fallaba con `AssertionError: L1 (7040) must be divisible by num_chunks (7)`. Con 4 o 12 bloques "funcionaba" por coincidencia numérica. **Fix aplicado**: `conditional_dict=conditional_dict` → `conditional_dict=conditional_dict_list[block_index]` en la línea 417, y `conditional_dict=conditional_dict_list[0]` en la línea 387 (rama `independent_first_frame=True`). Verificado reproduciendo el caso exacto que fallaba (7 bloques/56 frames, misma imagen, mismo prompt): denoise 51.1s, decode 61.0s, video 1280×704 de 10.2s sin errores.

Probado con éxito de punta a punta, incluyendo el caso que antes disparaba el bug #4: imagen de referencia real (frame extraído de un video ya generado) → primer frame del resultado idéntico a la referencia, frames posteriores muestran movimiento coherente con el prompt.

## Archivos nuevos/modificados esta sesión

- `app.py` — 3 pestañas, funciones `generate_multiscene`, `generate_image_conditioned`, `_parse_scenes`, `_build_block_prompts`, `_shape_for`, `_encode_reference_image`.
- `utils/inference_utils.py` — `prepare_multi_scene_inputs` (nueva), fixes de carga (`_torch_load`, `load_generator_checkpoint` con mmap+assign).
- `utils/wan_5b_wrapper.py` — `WanTextEncoder` construido en `device='meta'`.
- `benchmark_*.py`, `diagnose_vae_encode_temporal.py` — scripts de diagnóstico de esta sesión, no productivos pero útiles para no repetir investigación.

## Pendiente / próximos pasos sugeridos

1. Probar la combinación 9:16 + imagen de referencia + modo Rápido/Turbo (LightVAE) — no probada todavía.
2. Si se quiere revisitar cuantización INT8 en el futuro, primero evaluar si vale la pena subir `torch` a >=2.11 para que `torchao` cargue sus kernels compilados (no intentado en esta sesión, riesgo de romper el setup actual).
