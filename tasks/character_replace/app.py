import base64
import io
import json
import os
import time

import requests
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Character Replace AI",
    page_icon="🎭",
    layout="wide",
)

st.markdown("""
<style>
.block-container { max-width: 1200px; padding-top: 2rem; }
div[data-testid="stMetric"] {
    background: rgba(128,128,128,.08);
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎭 Character Replace AI")
st.write(
    "Upload an original scene and a character reference. "
    "Create a portable JSON edit specification and, when a compatible "
    "image-editing server is configured, submit the edit for generation."
)

with st.sidebar:
    st.header("Settings")
    server_url = st.text_input(
        "Compatible image-editing server URL",
        value=os.environ.get("IMAGEGEN_SERVER_URL", ""),
        help=(
            "Optional. This app's generation adapter expects POST /api/jobs, "
            "GET /api/jobs/{id}, GET /health and a result image URL. "
            "A generic image server may use a different API."
        ),
    ).strip().rstrip("/")
    preserve_background = st.checkbox("Preserve original background", True)
    preserve_pose = st.checkbox("Preserve original pose and position", True)
    preserve_lighting = st.checkbox("Match original lighting and shadows", True)
    preserve_clothing = st.checkbox("Keep original clothing where possible", False)
    output_format = st.selectbox("Output format", ["PNG", "JPEG"])

left, right = st.columns(2, gap="large")
with left:
    st.subheader("1. Original scene")
    original_file = st.file_uploader(
        "Upload original image", type=["png", "jpg", "jpeg", "webp"],
        key="original",
    )
    if original_file:
        original_image = Image.open(original_file).convert("RGB")
        st.image(original_image, use_container_width=True)
        st.caption(f"{original_image.width} × {original_image.height} px")

with right:
    st.subheader("2. Replacement character")
    reference_file = st.file_uploader(
        "Upload character reference", type=["png", "jpg", "jpeg", "webp"],
        key="reference",
    )
    if reference_file:
        reference_image = Image.open(reference_file).convert("RGB")
        st.image(reference_image, use_container_width=True)
        st.caption(f"{reference_image.width} × {reference_image.height} px")

st.subheader("3. Editing instructions")
instructions = st.text_area(
    "Describe the desired edit",
    value=(
        "Replace the main character in the original image with the character "
        "shown in the reference image. Keep the original background, scene, "
        "camera angle, composition, and unrelated objects unchanged. Blend "
        "the replacement naturally into the original scene."
    ),
    height=120,
)

def make_spec():
    return {
        "task": "character_replacement",
        "version": "1.0",
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
        "output": {
            "format": output_format.lower(),
            "keep_original_dimensions": True,
        },
    }

if original_file and reference_file:
    spec = make_spec()
    st.subheader("4. JSON edit specification")
    spec_text = json.dumps(spec, indent=2, ensure_ascii=False)
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
    if not server_url:
        st.warning(
            "JSON creation is ready. To generate the edited image, configure "
            "a compatible image-editing server URL in the sidebar. JSON itself "
            "does not perform image editing, and this repository does not "
            "include a model server."
        )
    else:
        st.caption(
            "Server adapter contract: POST /api/jobs with JSON fields "
            "'prompt' and 'images' (base64 strings), then poll "
            "GET /api/jobs/{id}. Confirm your server supports this contract."
        )
        if st.button("✨ Replace character", type="primary", use_container_width=True):
            try:
                health = requests.get(f"{server_url}/health", timeout=15)
                health.raise_for_status()
                health_data = health.json()
                if health_data.get("status") not in ("ready", "loading"):
                    st.error(f"Model server is not ready: {health_data}")
                    st.stop()

                def encode_upload(upload):
                    return base64.b64encode(upload.getvalue()).decode("utf-8")

                prompt_parts = [
                    "Use image 1 as the original scene and image 2 as the character reference.",
                    "Replace the main character in image 1 with the character from image 2.",
                    "Do not create a new scene. Preserve unrelated objects and composition.",
                    instructions,
                ]
                if preserve_background:
                    prompt_parts.append("Preserve the original background as accurately as possible.")
                if preserve_pose:
                    prompt_parts.append("Preserve the original character's pose and position where compatible.")
                if preserve_lighting:
                    prompt_parts.append("Match original lighting, shadows, perspective and color balance.")
                if preserve_clothing:
                    prompt_parts.append("Keep original clothing where possible.")

                payload = {
                    "prompt": " ".join(prompt_parts),
                    "images": [encode_upload(original_file), encode_upload(reference_file)],
                }
                response = requests.post(
                    f"{server_url}/api/jobs", json=payload, timeout=120
                )
                response.raise_for_status()
                job = response.json()
                job_id = job.get("id")
                if not job_id:
                    raise RuntimeError(f"Server response did not include job id: {job}")

                progress = st.progress(0)
                status_box = st.empty()
                deadline = time.monotonic() + 1800
                while time.monotonic() < deadline:
                    poll = requests.get(
                        f"{server_url}/api/jobs/{job_id}", timeout=60
                    )
                    poll.raise_for_status()
                    job = poll.json()
                    status = str(job.get("status", "")).lower()
                    status_box.write(f"Server status: **{status or 'working'}**")
                    progress_value = (job.get("progress") or {}).get("percent", 0)
                    try:
                        progress.progress(max(0, min(100, int(progress_value))))
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
                    raise RuntimeError(
                        "Job completed but no result image URL was returned. "
                        "Adjust download_result for your server's response format."
                    )
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

                result_bytes = output.getvalue()
                st.session_state["result_bytes"] = result_bytes
                st.session_state["result_name"] = file_name
                st.session_state["result_mime"] = mime
                st.success("Image generation completed.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the server. Check the URL and ensure the model server is running and accessible.")
            except requests.exceptions.Timeout:
                st.error("The image server timed out. Check the server logs and try again.")
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
    st.caption(
        "AI editing may still change background details. Review the result "
        "before using it."
    )
elif not (original_file and reference_file):
    st.info("Upload both images to create the JSON specification.")
