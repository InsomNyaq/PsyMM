# Environment and paths

Audio generation and DICE-Talk video generation can use separate Python environments. Code alone cannot generate media. Install compatible dependencies under each component's own terms and pin versions for a reproducible run. A machine-local `pip freeze` is not presented as a portable requirements file.

| Setting | Purpose |
| --- | --- |
| `PSYMM_ROOT` | Checkout root; defaults to the package's parent |
| `PSYMM_DATA` | Authorized source files and `AvaMERG/train.json`, `AvaMERG/emo/`; defaults to `<repo>/data` |
| `PSYMM_WEIGHTS` | OpenVoice, DICE-Talk, Ola and auxiliary checkpoints; defaults to `<repo>/weights` |
| `PSYMM_AUDIO_PYTHON` | Audio-stage interpreter; defaults to `python` on `PATH` |
| `PSYMM_VIDEO_PYTHON` | Video-stage interpreter; defaults to `python` on `PATH` |
| `HF_HOME`, `TORCH_HOME`, `NLTK_DATA` | Component caches; default to `<repo>/.cache/` subdirectories |

Use `run_multimodal.sh` with the two interpreter variables, or activate a compatible environment and call `generate_multimodal.py` directly. Set `PSYMM_DATA` and `PSYMM_WEIGHTS` to the actual local asset locations when they are outside the checkout. The `<repo>/data`, `<repo>/weights` and `<repo>/.cache` defaults are local conventions, not bundled assets.

The wrapper sets `PSYMM_ROOT` to its own checkout. The Ola video fine-tuning script additionally requires `E0_MANIFEST` to point to a verified experiment manifest with its `legacy_ola_denylist`; the original machine-specific manifest is not distributed. Paths and hashes in a manifest must be reviewed for the target data before training.

Required model inputs include Chinese and English MeloTTS, OpenVoice converter and speaker embeddings, DICE-Talk checkpoints and eight emotion-prior arrays, and the models loaded by DICE-Talk. These are excluded from this code release. `python generate_multimodal.py preflight` checks availability but also hashes weight files, so it can take time.

## Generation asset layout

The generation code currently expects the following local layout. Obtain the assets under their original terms; none of the paths below contains a bundled checkpoint in this repository. Symlinks to an existing installation can avoid duplicate copies.

| Local location | Required contents |
| --- | --- |
| `<repo>/MeloTTS-main/myshell-ai/MeloTTS-Chinese/` | `config.json`, `checkpoint.pth` |
| `<repo>/MeloTTS-main/myshell-ai/MeloTTS-English-v3/` | `config.json`, `checkpoint.pth` |
| `$PSYMM_WEIGHTS/openvoice_weights/converter/` | `config.json`, `checkpoint.pth` |
| `$PSYMM_WEIGHTS/openvoice_weights/base_speakers/ses/` | `en-newest.pth`, `zh.pth` matching the base TTS speakers |
| `$PSYMM_WEIGHTS/bert-base-uncased/`, `$PSYMM_WEIGHTS/bert-base-multilingual-uncased/` | Corresponding tokenizer and model assets |
| `$PSYMM_WEIGHTS/dicetalk_weights/DICE-Talk/` | `unet.pth`, `audio_linear.pth`, `pose_guider.pth`, `emo_model.pth` |
| `$PSYMM_WEIGHTS/dicetalk_weights/` | `whisper-tiny/`, `stable-video-diffusion-img2vid-xt/`, `RIFE/`, `yoloface_v5m.pt` |
| `<repo>/DICE-Talk-main/examples/emo/` | Emotion-prior arrays used by video generation |

MeloTTS checkpoints and DICE-Talk emotion priors use the checkout-relative locations above; setting `PSYMM_WEIGHTS` alone does not relocate those two asset groups. The source catalog, emotion-pool references and dialogue input are also required. See `python generate_multimodal.py prepare --help` for their explicit arguments.

Vendored components retain their own `LICENSE` files and source notices. These are adapted source snapshots; additional upstream runtime assets may be needed. The source copies include local integration changes. Upstream projects: [OpenVoice](https://github.com/myshell-ai/OpenVoice), [MeloTTS](https://github.com/myshell-ai/MeloTTS), [DICE-Talk](https://github.com/toto222/DICE-Talk), [Ola](https://github.com/Ola-Omni/Ola). Exact upstream revisions and model redistribution terms still need verification; DICE-Talk's noncommercial share-alike license is particularly relevant.
