import base64
import io
import json
import os
import time

import requests
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Character Replace AI", page_icon="🎭", layout="wide")

st.markdown("""
<style>
.block-container { max-width: 1200px; padding-top: 2rem; }
div[data-testid="stMetric"] { background: rgba(128,128,128,.08); padding: 14px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🎭 Character Replace AI")
st.write(
    "Upload an original scene and a character reference. Create a JSON specification "
    "and generate an edited image using a configured image-editing model."
)

def get_secret(name, default=""):
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return os.environ.get(name, default)

with st.sidebar:
    st.header("Generation settings")
    backend = st.radio(
        "Image generation method",
        ["Hugging Face model", "Compatible image server"],
        help="Hugging Face uses an image-editing model through Inference Providers. Provider use may cost money; check your account credits and billing first.",
    )
    hf_token = ""
    server_url = ""
    if backend == "Hugging Face model":
        hf_token = st.text_input(
            "Hugging Face access token",
            value=get_secret("HF_TOKEN"),
            type="password",
            help="Create a token with Inference Providers permission at huggingface.co/settings/tokens. Prefer adding HF_TOKEN under Streamlit App Settings > Secrets.",
        ).strip()
        model_id = st.text_input("Image-editing model", value="Qwen/Qwen-Image-Edit").strip()
        provider = st.selectbox("Inference provider", ["fal-ai", "auto"], index=0)
    else:
        server_url = st.text_input(
            "Compatible image-editing server URL",
            value=get_secret("IMAGEGEN_SERVER_URL"),
            help="Expected API: GET /health, POST /api/jobs, GET /api/jobs/{id}.",
        ).strip().rstrip("/")

    preserve_background = st.checkbox("Preserve original background", True)
    preserve_pose = st.checkbox("Preserve original pose and position", True)
    preserve_lighting = st.checkbox("Match original lighting and shadows", True)
    preserve_clothing = st.checkbox("Keep original clothing where possible", False)
    output_format = st.selectbox("Output format", ["PNG", "JPEG"])

left, right = st.columns(2, gap="large")
with left:
    st.subheader("1. Original scene")
    original_file = st.file_uploader("Upload original image", type=["png", "jpg", "jpeg", "webp"], key="original")
    if original_file:
        original_image = Image.open(original_file).convert("RGB")
        st.image(original_image, use_container_width=True)
        st.caption(f"{original_image.width} × {original_image.height} px")

with right:
    st.subheader("2. Replacement character")
    reference_file = st.file_uploader("Upload character reference", type=["png", "jpg", "jpeg", "webp"], key="reference")
    if reference_file:
        reference_image = Image.open(reference_file).convert("RGB")
        st.image(reference_image, use_container_width=True)
        st.caption(f"{reference_image.width} × {reference_image.height} px")

st.subheader("3. Editing instructions")
instructions = st.text_area(
    "Describe the desired edit",
    value=(
        "Replace the main character in the original scene with the character shown in the "
        "reference image. Keep the original background, camera angle, composition, and "
        "unrelated objects unchanged. Blend the replacement naturally into the scene."
    ),
    height=120,
)

def make_spec():
    return {
        "task": "character_replacement",
        "version": "1.1",
        "inputs": {
            "original_image": original_file.name,
            "reference_character_image": reference_file.name,
        },
        "operation": {
            "action": "replace_main_character",
            "scene_source": "original_image",
            "character_identity_source": "reference_character_image",
        },
        "preservation": {
            "background": preserve_background,
            "pose_and_position": preserve_pose,
            "lighting_and_shadows": preserve_lighting,
            "original_clothing_where_possible": preserve_clothing,
            "composition": True,
            "camera_angle": True,
            "unrelated_objects": True,
        },
        "instructions": instructions,
        "output": {"format": output_format.lower(), "keep_original_dimensions": True},
        "generation_backend": backend,
    }

def make_prompt():
    parts = [
        "The input image is a two-panel visual guide. The LEFT panel is the original scene to edit. "
        "The RIGHT panel is the character identity reference. Return only the edited LEFT scene, "
        "not a collage or split-panel image. Replace only the main character in the left scene "
        "with the person/character shown on the right. Preserve the scene's composition and all "
        "unrelated details as faithfully as possible.",
        instructions,
    ]
    if preserve_background:
        parts.append("Keep the original left-panel background unchanged as closely as possible.")
    if preserve_pose:
        parts.append("Keep the original character's pose, position, framing and body orientation.")
    if preserve_lighting:
        parts.append("Match the original lighting, shadows, perspective and color balance.")
    if preserve_clothing:
        parts.append("Keep the original character's clothing where compatible with the reference.")
    return " ".join(parts)

def build_reference_canvas(scene, reference):
    # A single image-to-image request accepts one image. Place the scene and reference
    # side-by-side so the model receives both visually; this is a best-effort workaround.
    target_h = min(1024, max(512, scene.height))
    target_w = max(1, int(scene.width * target_h / scene.height))
    target_w = min(target_w, 1024)
    scene_panel = ImageOps.fit(scene, (target_w, target_h), method=Image.Resampling.LANCZOS)
    ref_panel = ImageOps.contain(reference, (max(256, target_w // 2), target_h), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w + ref_panel.width, target_h), "white")
    canvas.paste(scene_panel, (0, 0))
    canvas.paste(ref_panel, (target_w, (target_h - ref_panel.height) // 2))
    return canvas, target_w, target_h

def extract_scene_panel(generated, scene_width, scene_height):
    # Many image-editing models return a single square image regardless of input size.
    # If the output retains the two-panel wide layout, keep its left scene panel.
    if generated.width / max(1, generated.height) > 1.35:
        crop_width = int(generated.height * (scene_width / scene_height))
        crop_width = max(1, min(crop_width, generated.width))
        generated = generated.crop((0, 0, crop_width, generated.height))
    return generated.resize((scene_width, scene_height), Image.Resampling.LANCZOS)

if original_file and reference_file:
    spec = make_spec()
    spec_text = json.dumps(spec, indent=2, ensure_ascii=False)
    st.subheader("4. JSON edit specification")
    st.code(spec_text, language="json")
    st.download_button("⬇️ Download JSON", data=spec_text, file_name="character_replacement.json", mime="application/json", use_container_width=True)

    st.divider()
    st.subheader("5. Generate replacement image")
    if backend == "Hugging Face model":
        if not hf_token:
            st.info("Add an HF_TOKEN secret or enter a Hugging Face token in the sidebar to enable generation.")
        st.caption(
            "This mode sends a combined scene/reference image to an image-editing model. "
            "Two-image character replacement is best-effort: the model may alter background details "
            "or fail to preserve identity exactly. Hugging Face Inference Providers may require paid credits."
        )
    elif not server_url:
        st.info("Enter a reachable compatible image-server URL in the sidebar to enable this backend.")

    can_generate = bool(hf_token) if backend == "Hugging Face model" else bool(server_url)
    if can_generate and st.button("✨ Replace character", type="primary", use_container_width=True):
        try:
            prompt = make_prompt()
            if backend == "Hugging Face model":
                from huggingface_hub import InferenceClient
                with st.spinner("Submitting images to the model. This may take several minutes…"):
                    canvas, scene_panel_width, scene_panel_height = build_reference_canvas(
                        original_image, reference_image
                    )
                    client = InferenceClient(provider=provider, api_key=hf_token, timeout=600)
                    result = client.image_to_image(
                        canvas,
                        prompt=prompt,
                        model=model_id or "Qwen/Qwen-Image-Edit",
                    )
                    result_image = extract_scene_panel(
                        result.convert("RGB"), original_image.width, original_image.height
                    )
            else:
                health = requests.get(f"{server_url}/health", timeout=15)
                health.raise_for_status()
                health_data = health.json()
                if health_data.get("status") not in ("ready", "loading"):
                    raise RuntimeError(f"Model server is not ready: {health_data}")

                def encode_upload(upload):
                    return base64.b64encode(upload.getvalue()).decode("utf-8")

                payload = {
                    "prompt": prompt,
                    "images": [encode_upload(original_file), encode_upload(reference_file)],
                }
                response = requests.post(f"{server_url}/api/jobs", json=payload, timeout=120)
                response.raise_for_status()
                job = response.json()
                job_id = job.get("id")
                if not job_id:
                    raise RuntimeError(f"Server response did not include job id: {job}")

                progress = st.progress(0)
                status_box = st.empty()
                deadline = time.monotonic() + 1800
                while time.monotonic() < deadline:
                    poll = requests.get(f"{server_url}/api/jobs/{job_id}", timeout=60)
                    poll.raise_for_status()
                    job = poll.json()
                    status = str(job.get("status", "")).lower()
                    status_box.write(f"Server status: **{status or 'working'}**")
                    try:
                        progress.progress(max(0, min(100, int((job.get("progress") or {}).get("percent", 0)))))
                    except (TypeError, ValueError):
                        progress.progress(0)
                    if status in ("failed", "error", "cancelled", "canceled"):
                        raise RuntimeError(job.get("error") or str(job))
                    if status in ("succeeded", "completed", "success"):
                        break
                    time.sleep(3)
                else:
                    raise TimeoutError("Generation exceeded 30 minutes.")

                images = (job.get("result") or {}).get("images") or []
                image_url = images[0].get("url") if images else None
                if not image_url:
                    raise RuntimeError("Job completed but no result image URL was returned.")
                if image_url.startswith("/"):
                    image_url = server_url + image_url
                result_response = requests.get(image_url, timeout=180)
                result_response.raise_for_status()
                result_image = Image.open(io.BytesIO(result_response.content)).convert("RGB")

            output = io.BytesIO()
            if output_format == "JPEG":
                result_image.save(output, format="JPEG", quality=95)
                file_name, mime = "character_replaced.jpg", "image/jpeg"
            else:
                result_image.save(output, format="PNG")
                file_name, mime = "character_replaced.png", "image/png"
            st.session_state["result_bytes"] = output.getvalue()
            st.session_state["result_name"] = file_name
            st.session_state["result_mime"] = mime
            st.success("Image model returned a result. Review it carefully before use.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the configured server. Check the URL and confirm the server is running.")
        except requests.exceptions.Timeout:
            st.error("The image service timed out. Try again later or use a smaller image.")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

if st.session_state.get("result_bytes"):
    st.subheader("6. Edited result")
    st.image(st.session_state["result_bytes"], use_container_width=True)
    st.download_button(
        "⬇️ Download edited image",
        data=st.session_state["result_bytes"],
        file_name=st.session_state["result_name"],
        mime=st.session_state["result_mime"],
        type="primary",
        use_container_width=True,
    )
    st.caption("The output is AI-generated. Background details and character identity may not be exact.")
elif not (original_file and reference_file):
    st.info("Upload both images to create the JSON specification.")
