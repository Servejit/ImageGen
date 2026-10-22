import base64
import io
import json
import os
import time

import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Character Replace AI", page_icon="🎭", layout="wide")

st.markdown("""
<style>
.block-container { max-width: 1200px; padding-top: 2rem; }
div[data-testid="stMetric"] { background: rgba(128,128,128,.08); padding: 14px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🎭 Character Replace AI")
st.write(
    "Upload an original scene and a character reference. Create a downloadable JSON "
    "specification, then edit with a local Qwen model or a compatible image server."
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
        ["Local Qwen model", "Compatible image server"],
        help="Local Qwen runs the model in this Python process. It does not use Hugging Face Inference Providers or an access token. Public model weights are downloaded on first run.",
    )
    server_url = ""
    if backend == "Local Qwen model":
        model_id = st.selectbox(
            "Local Qwen model",
            ["Qwen/Qwen-Image-Edit-2509", "Qwen/Qwen-Image-Edit"],
            index=0,
            help="The 2509 model supports multiple input images. The older model is single-image input.",
        )
        steps = st.slider("Inference steps", min_value=10, max_value=50, value=30, step=5)
        st.caption("First run downloads several tens of GB of model files and may take a long time.")
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
    original_image = None
    if original_file:
        original_image = Image.open(original_file).convert("RGB")
        st.image(original_image, use_container_width=True)
        st.caption(f"{original_image.width} × {original_image.height} px")

with right:
    st.subheader("2. Replacement character")
    reference_file = st.file_uploader("Upload character reference", type=["png", "jpg", "jpeg", "webp"], key="reference")
    reference_image = None
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
        "unrelated objects unchanged. Match the original character's pose and position where "
        "possible. Blend the replacement naturally into the scene."
    ),
    height=120,
)

def make_spec():
    return {
        "task": "character_replacement",
        "version": "1.2",
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
        "model": model_id if backend == "Local Qwen model" else "compatible-image-server",
    }

def make_prompt(spec):
    preservation = spec.get("preservation", {})
    parts = [
        "Use the first input image as the original scene and the second input image as the "
        "character identity/reference image. Edit the first image: replace only its main "
        "character with the character from the second image. Do not make a collage or split "
        "panel. Preserve the original scene, framing, camera angle, pose and unrelated details "
        "as faithfully as possible.",
        str(spec.get("instructions", "")),
    ]
    if preservation.get("background"):
        parts.append("Keep the original scene background unchanged as closely as possible.")
    if preservation.get("pose_and_position"):
        parts.append("Keep the original character's pose, position, framing and body orientation.")
    if preservation.get("lighting_and_shadows"):
        parts.append("Match the original lighting, shadows, perspective and color balance.")
    if preservation.get("original_clothing_where_possible"):
        parts.append("Keep the original character's clothing where compatible with the reference.")
    parts.append(
        "Follow this JSON edit specification as the source of truth: "
        + json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
    )
    return " ".join(part for part in parts if part)

@st.cache_resource(show_spinner=False)
def load_local_qwen(model_name, device_name):
    import torch
    from diffusers import QwenImageEditPlusPipeline, QwenImageEditPipeline

    if device_name == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device_name == "mps":
        dtype = torch.float16
    else:
        raise RuntimeError("Local Qwen needs a supported GPU. CUDA or Apple Silicon MPS was not detected.")

    if model_name == "Qwen/Qwen-Image-Edit-2509":
        pipe = QwenImageEditPlusPipeline.from_pretrained(model_name, torch_dtype=dtype)
    else:
        pipe = QwenImageEditPipeline.from_pretrained(model_name, torch_dtype=dtype)

    if device_name == "cuda":
        # Offload model components to system RAM when they are not actively running.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device_name)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def detect_local_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

if original_file and reference_file:
    spec = make_spec()
    spec_text = json.dumps(spec, indent=2, ensure_ascii=False)
    st.subheader("4. JSON edit specification")
    st.code(spec_text, language="json")
    st.download_button(
        "⬇️ Download JSON",
        data=spec_text,
        file_name="character_replacement.json",
        mime="application/json",
        use_container_width=True,
    )

    st.divider()
    st.subheader("5. Generate replacement image")
    if backend == "Local Qwen model":
        st.info(
            "Local mode uses the Qwen Diffusers pipeline directly. It does not call Hugging Face "
            "Inference Providers and does not need an HF token. It downloads public model weights "
            "on first use, so internet access and substantial disk/RAM/GPU resources are required."
        )
    elif not server_url:
        st.info("Enter a reachable compatible image-server URL in the sidebar to enable this backend.")

    can_generate = True if backend == "Local Qwen model" else bool(server_url)
    if can_generate and st.button("✨ Replace character", type="primary", use_container_width=True):
        try:
            prompt = make_prompt(spec)
            if backend == "Local Qwen model":
                import torch

                device_name = detect_local_device()
                if device_name == "cpu":
                    raise RuntimeError(
                        "No supported GPU was detected. Qwen-Image-Edit is a very large model and "
                        "CPU-only generation is not enabled because it may need enormous RAM and run "
                        "extremely slowly. Run this app on a GPU machine (NVIDIA CUDA recommended, "
                        "or Apple Silicon MPS where supported). Streamlit Community Cloud and a "
                        "standard Codespace usually do not provide a suitable GPU."
                    )
                with st.spinner(
                    "Loading local Qwen model if needed, then generating. First run downloads the model and may take a long time…"
                ):
                    pipe = load_local_qwen(model_id, device_name)
                    if model_id == "Qwen/Qwen-Image-Edit-2509":
                        model_inputs = [original_image, reference_image]
                    else:
                        st.warning(
                            "The selected older Qwen model accepts one image. A side-by-side guide "
                            "is used for this model; the 2509 model is recommended for two-image input."
                        )
                        from PIL import ImageOps
                        target_size = (1024, 1024)
                        scene_panel = ImageOps.contain(original_image, (700, 1024), method=Image.Resampling.LANCZOS)
                        ref_panel = ImageOps.contain(reference_image, (320, 1024), method=Image.Resampling.LANCZOS)
                        canvas = Image.new("RGB", (scene_panel.width + ref_panel.width, max(scene_panel.height, ref_panel.height)), "white")
                        canvas.paste(scene_panel, (0, (canvas.height - scene_panel.height) // 2))
                        canvas.paste(ref_panel, (scene_panel.width, (canvas.height - ref_panel.height) // 2))
                        model_inputs = canvas

                    call_args = {
                        "image": model_inputs,
                        "prompt": prompt,
                        "num_inference_steps": steps,
                    }
                    if model_id == "Qwen/Qwen-Image-Edit-2509":
                        call_args["true_cfg_scale"] = 4.0
                        call_args["negative_prompt"] = " "
                    with torch.inference_mode():
                        result_image = pipe(**call_args).images[0].convert("RGB")
                    result_image = result_image.resize(
                        (original_image.width, original_image.height),
                        Image.Resampling.LANCZOS,
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
                    "specification": spec,
                    "images": [encode_upload(original_file), encode_upload(reference_file)],
                    "image_roles": ["original_scene", "character_reference"],
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
