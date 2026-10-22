# Character Replace AI

A Streamlit task for preparing a JSON specification to replace the main character in an original image using a second reference image.

## Run in GitHub Codespaces

```bash
pip install -r tasks/character_replace/requirements.txt
streamlit run tasks/character_replace/app.py --server.address 0.0.0.0
```

## Important: image generation backend

The app can create and download the JSON edit specification without a model server. Actual AI image replacement requires a compatible image-editing server. Set `IMAGEGEN_SERVER_URL` or enter its URL in the sidebar.

The optional adapter currently expects:
- `GET /health` returning JSON with `status: ready` or `status: loading`
- `POST /api/jobs` accepting JSON with `prompt` and `images` (two base64-encoded images), returning an `id`
- `GET /api/jobs/{id}` returning `status` and, on completion, `result.images[0].url`

These are an adapter contract, not a universal API. A server with different endpoints needs a matching adapter.

Background preservation is a model instruction, not a guarantee of pixel-identical background.
