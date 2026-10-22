# Character Replace AI

A Streamlit app that accepts an original scene and a character reference, creates a downloadable JSON edit specification, and edits the image with a local Qwen Diffusers pipeline or a compatible image server.

## What changed

- Removed Hugging Face Inference Providers and the access-token field.
- Added direct local inference through the Diffusers Qwen pipeline.
- Default model: `Qwen/Qwen-Image-Edit-2509`, which supports multiple input images. The app sends the original scene and character reference as separate images.
- Kept the JSON download, editing instructions, background/pose/lighting/clothing preservation settings, optional compatible image-server backend, output format, result preview and image download.

## Important: local model weights versus inference APIs

Local mode does **not** call Hugging Face Inference Providers and does not require an access token. The model weights are public and are downloaded to the machine running the app on first use; the default model source is the public model repository `Qwen/Qwen-Image-Edit-2509`. That download can be tens of gigabytes and requires internet access and free disk space. After the weights are cached, inference can run locally without a hosted inference API.

## Hardware requirements

Qwen-Image-Edit is a large model. Practical memory needs depend on model version, image size, dtype and offloading:

- **Recommended:** a Linux/Windows machine with a recent NVIDIA GPU, ideally 24–80 GB VRAM, and 64 GB or more system RAM. More VRAM makes inference easier.
- **Lower-VRAM setups:** CPU offloading may reduce GPU memory needs but uses system RAM and is slower; success is not guaranteed on every GPU.
- **Apple Silicon:** MPS may work depending on PyTorch/Diffusers support and available unified memory.
- **CPU-only:** not enabled in the app because this model is extremely large and generation may be impractically slow or run out of memory.
- **Streamlit Community Cloud / standard GitHub Codespaces:** normally do not provide the suitable GPU and disk resources required for this model. The app can display its interface there, but local generation needs to run on a suitable GPU host. Do not expect local generation to work on a standard free hosted instance.

No token or paid inference API is needed for local inference, but electricity/hardware and network/download costs may apply. The code does not promise that GPU execution is free.

## Deploy on Streamlit Community Cloud

Select repository `Servejit/ImageGen`, branch `main`, and entrypoint `app.py`. The app interface may deploy, but local Qwen generation requires a suitable GPU host. For actual local generation, run on your own GPU machine or a GPU server you control.

## Run locally

Use a Python environment with a compatible PyTorch build for your hardware. Install the PyTorch build appropriate for your CUDA version from the official PyTorch installation selector, then install the project dependencies:

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0
```

The first generation downloads the model weights and can take a long time. The app caches the loaded pipeline during the running session.

## Compatible image-server option

Choose **Compatible image server** and enter a reachable server URL. The adapter expects:

- `GET /health` returning JSON with `status: ready` or `status: loading`
- `POST /api/jobs` accepting JSON with `prompt`, `specification` (the generated JSON object), `image_roles` (`original_scene`, `character_reference`), and `images` (two base64-encoded images in that order), returning an `id`
- `GET /api/jobs/{id}` returning `status` and, on completion, `result.images[0].url`

This is an adapter contract, not a universal API. A server with different endpoints needs a matching adapter.

Background preservation is a model instruction, not a guarantee of pixel-identical background. Inspect generated results before use.
