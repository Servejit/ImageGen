# Character Replace AI

A Streamlit app that accepts an original scene and a character reference, creates a downloadable JSON edit specification, and can submit the edit to an image-editing model.

## Deploy on Streamlit Community Cloud

Select repository `Servejit/ImageGen`, branch `main`, and entrypoint `app.py`. The root `requirements.txt` includes the required packages.

## Built-in Hugging Face model option

1. Create a Hugging Face account at https://huggingface.co/.
2. Create a user access token at https://huggingface.co/settings/tokens with **Inference Providers** permission.
3. In Streamlit Community Cloud, open **App settings → Secrets** and add:

   ```toml
   HF_TOKEN = "hf_your_token_here"
   ```

4. Reboot the Streamlit app. In the sidebar, choose **Hugging Face model**, choose a provider, then upload both images and click **Replace character**.

The default model is `Qwen/Qwen-Image-Edit` and the default provider is `fal-ai`. Inference Providers can require paid credits; check the current account pricing and available balance before generating. Do not paste a secret token into a public GitHub file.

The app packages the original scene and reference into one side-by-side visual guide because the selected image-to-image API accepts one input image. This is a best-effort approach: a model may not faithfully preserve the exact background, pose, or identity. Inspect the result before use.

## Compatible image-server option

Choose **Compatible image server** and enter a reachable server URL. The adapter expects:
- `GET /health` returning JSON with `status: ready` or `status: loading`
- `POST /api/jobs` accepting JSON with `prompt`, `specification` (the generated JSON object), `image_roles` (`original_scene`, `character_reference`), and `images` (two base64-encoded images in that order), returning an `id`
- `GET /api/jobs/{id}` returning `status` and, on completion, `result.images[0].url`

This is an adapter contract, not a universal API. A server with different endpoints needs a matching adapter.

## Run locally / in Codespaces

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0
```

Background preservation is a model instruction, not a guarantee of pixel-identical background.
