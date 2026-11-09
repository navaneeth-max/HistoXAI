import io
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from captum.attr import LayerGradCam, LayerAttribution, IntegratedGradients

st.set_page_config(
    page_title="HistoxAI | Colorectal Histopathology",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 HistoxAI")
st.subheader("Colorectal Histopathology Classification & Explainable AI")
st.caption(
    "Research/educational demonstration only. This application is not a "
    "medical diagnostic device and must not be used for clinical decisions."
)

DEFAULT_CHECKPOINT = Path("colorectal_output/colorectal_inceptionv3_final.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Keep CPU resource use predictable on hosted instances.
if DEVICE.type == "cpu":
    torch.set_num_threads(2)

@st.cache_resource
def load_model(checkpoint_bytes):
    # Only load a checkpoint produced by your own trusted training pipeline.
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes),
        map_location=DEVICE,
        weights_only=False
    )
    classes = checkpoint["classes"]
    input_size = int(checkpoint.get("input_size", 299))
    mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("std", [0.229, 0.224, 0.225])

    model = models.inception_v3(weights=None, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    if model.AuxLogits is not None:
        model.AuxLogits.fc = nn.Linear(
            model.AuxLogits.fc.in_features, len(classes)
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return model, classes, preprocess, input_size, mean, std

def get_model_scores(model, batch):
    output = model(batch)
    # InceptionV3 returns a tensor in eval mode; this is a defensive fallback.
    if hasattr(output, "logits"):
        output = output.logits
    return output

def prepare_image(pil_image, preprocess):
    rgb = pil_image.convert("RGB")
    tensor = preprocess(rgb).unsqueeze(0).to(DEVICE)
    return rgb, tensor

def make_gradcam(model, input_tensor, target_class, image_size):
    model.eval()
    target_layer = model.Mixed_7c
    gradcam = LayerGradCam(model, target_layer)
    x = input_tensor.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    attribution = gradcam.attribute(x, target=target_class)
    attribution = LayerAttribution.interpolate(
        attribution, image_size, interpolate_mode="bilinear"
    )
    cam = attribution.squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam

def make_integrated_gradients(model, input_tensor, target_class):
    model.eval()
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor)
    attribution = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class,
        n_steps=16,
        internal_batch_size=1
    )
    attr = attribution.squeeze(0).detach().cpu().numpy()
    heat = np.abs(attr).sum(axis=0)
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    else:
        heat = np.zeros_like(heat)
    return heat

def overlay_heatmap(rgb_image, heatmap, alpha=0.42):
    h, w = rgb_image.shape[:2]
    heatmap = cv2.resize(heatmap.astype(np.float32), (w, h))
    colored = cv2.applyColorMap(
        np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip(
        (1 - alpha) * rgb_image + alpha * colored, 0, 255
    ))

def region_boxes(cam, image_shape, threshold=0.65):
    """Return connected high-activation areas; these are attribution regions,
    not segmented tumor boundaries or pathologist-verified structures."""
    h, w = image_shape[:2]
    resized = cv2.resize(cam.astype(np.float32), (w, h))
    mask = np.uint8(resized >= threshold) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    regions = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area >= max(30, int(h * w * 0.001)):
            regions.append((int(x), int(y), int(bw), int(bh), int(area)))
    regions.sort(key=lambda item: item[4], reverse=True)
    return resized, regions[:5]

def draw_regions(rgb_image, regions):
    canvas = rgb_image.copy()
    for idx, (x, y, w, h, area) in enumerate(regions, start=1):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(canvas, (x + 14, y + 14), 12, (0, 255, 0), -1)
        cv2.putText(
            canvas, str(idx), (x + 10, y + 19),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
        )
    return canvas

def lime_explanation(model, image_rgb, preprocess, target_class):
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries

        def predict_fn(images):
            # Process perturbations in small batches to limit RAM usage.
            outputs = []
            chunk_size = 8
            for start in range(0, len(images), chunk_size):
                chunk = images[start:start + chunk_size]
                batch = []
                for arr in chunk:
                    pil = Image.fromarray(
                        np.uint8(np.clip(arr, 0, 1) * 255)
                    )
                    batch.append(preprocess(pil).unsqueeze(0))
                batch_tensor = torch.cat(batch).to(DEVICE)
                with torch.inference_mode():
                    logits = get_model_scores(model, batch_tensor)
                    probs = torch.softmax(logits, dim=1)
                outputs.append(probs.cpu().numpy())
                del batch_tensor, logits, probs
            return np.concatenate(outputs, axis=0)

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            np.asarray(image_rgb).astype(np.float32) / 255.0,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=300
        )
        label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=False,
            num_features=8,
            hide_rest=False
        )
        lime_visual = mark_boundaries(temp, mask)
        return lime_visual, None
    except Exception as exc:
        return None, str(exc)

# ------------------------- Sidebar / checkpoint -------------------------
st.sidebar.header("Model")
uploaded_checkpoint = st.sidebar.file_uploader(
    "Upload trained .pth checkpoint", type=["pth"]
)
if uploaded_checkpoint is not None:
    checkpoint_bytes = uploaded_checkpoint.getvalue()
elif DEFAULT_CHECKPOINT.exists():
    checkpoint_bytes = DEFAULT_CHECKPOINT.read_bytes()
else:
    checkpoint_bytes = None

if checkpoint_bytes is None:
    st.warning(
        "The trained model checkpoint is not available yet. First run "
        "`colorectal_histology_pipeline.py` until it completes. It should "
        "create `colorectal_output/colorectal_inceptionv3_final.pth`. "
        "You can then place the `colorectal_output` folder beside this app "
        "or upload the .pth file using the sidebar."
    )
    st.info(
        "You can prepare the app now, but predictions and XAI visualizations "
        "will become available only after model training finishes."
    )
    st.stop()

try:
    model, classes, preprocess, input_size, mean, std = load_model(checkpoint_bytes)
except Exception as exc:
    st.error(f"Could not load this checkpoint: {exc}")
    st.stop()

st.sidebar.success(f"Model loaded on {DEVICE}")
st.sidebar.caption(f"Architecture: InceptionV3 · Input: {input_size} × {input_size}")
st.sidebar.caption(f"Classes: {len(classes)}")

# ------------------------- Upload and prediction -------------------------
st.markdown("### 1. Upload a histopathology image")
uploaded_image = st.file_uploader(
    "Choose an image tile", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
)

if uploaded_image is None:
    st.info("Upload a previously unseen image tile to run prediction and explanations.")
    st.stop()

try:
    pil_image = Image.open(uploaded_image).convert("RGB")
except Exception as exc:
    st.error(f"Unable to read the uploaded image: {exc}")
    st.stop()

rgb_image, input_tensor = prepare_image(pil_image, preprocess)

with torch.no_grad():
    logits = get_model_scores(model, input_tensor)
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

predicted_idx = int(np.argmax(probabilities))
predicted_class = classes[predicted_idx]
confidence = float(probabilities[predicted_idx])

st.markdown("### 2. Prediction")
left, right = st.columns([1, 1])
with left:
    st.image(rgb_image, caption="Uploaded image", use_container_width=True)
with right:
    st.metric("Predicted tissue class", predicted_class)
    st.metric("Model confidence", f"{confidence * 100:.2f}%")
    if confidence < 0.60:
        st.warning(
            "The model confidence is relatively low. Treat this prediction "
            "as uncertain and review the probability distribution."
        )
    st.caption(
        "Confidence is the model's softmax score, not a calibrated probability "
        "of clinical correctness."
    )

st.markdown("#### Class probability distribution")
sorted_indices = np.argsort(probabilities)[::-1]
st.bar_chart({
    "Probability (%)": {
        classes[i]: float(probabilities[i] * 100) for i in sorted_indices
    }
})

# ------------------------- Explainability -------------------------
st.markdown("### 3. Explainable AI demonstrations")
st.write(
    "Generate each explanation only when needed. These visualizations describe "
    "which image regions influenced this model's output; they do not establish "
    "a diagnosis or prove that a highlighted region contains a particular "
    "tissue structure."
)

original_array = np.asarray(rgb_image)

if st.button("Generate Grad-CAM", key="generate_gradcam"):
    with st.spinner("Generating Grad-CAM..."):
        try:
            gradcam = make_gradcam(
                model, input_tensor, predicted_idx, (input_size, input_size)
            )
            grad_overlay = overlay_heatmap(original_array, gradcam)
            _, regions = region_boxes(gradcam, original_array.shape)
            region_image = draw_regions(original_array, regions)

            tab1, tab2 = st.tabs(["Grad-CAM", "High-activation regions"])
            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(
                        original_array, caption="Original image",
                        use_container_width=True
                    )
                with c2:
                    st.image(
                        grad_overlay, caption="Grad-CAM overlay",
                        use_container_width=True
                    )
                st.caption(
                    "Warmer/high-activation areas indicate stronger relative "
                    "attribution for the selected predicted class."
                )
            with tab2:
                st.image(
                    region_image,
                    caption="Top high-activation regions (numbered)",
                    use_container_width=True
                )
                if regions:
                    for idx, (x, y, w, h, area) in enumerate(regions, start=1):
                        st.write(
                            f"**Region {idx}:** location ({x}, {y}), "
                            f"size {w} × {h} pixels. This area has relatively "
                            f"high Grad-CAM activation for '{predicted_class}'."
                        )
                else:
                    st.info(
                        "No distinct connected high-activation regions were "
                        "found at the current threshold. The heatmap is still "
                        "available."
                    )
                st.caption(
                    "These are attribution areas, not tumor segmentation "
                    "masks or pathologist-verified regions."
                )
        except Exception as exc:
            st.error(f"Grad-CAM generation failed: {exc}")
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

if st.button(
    "Generate Integrated Gradients",
    key="generate_integrated_gradients"
):
    with st.spinner("Generating Integrated Gradients..."):
        try:
            ig_map = make_integrated_gradients(
                model, input_tensor, predicted_idx
            )
            ig_overlay = overlay_heatmap(original_array, ig_map)
            c1, c2 = st.columns(2)
            with c1:
                st.image(
                    ig_map, caption="Integrated Gradients attribution",
                    use_container_width=True
                )
            with c2:
                st.image(
                    ig_overlay, caption="Integrated Gradients overlay",
                    use_container_width=True
                )
            st.caption(
                "Attribution indicates relative influence on the model output; "
                "it is not a verified tissue or tumor boundary."
            )
        except Exception as exc:
            st.error(f"Integrated Gradients generation failed: {exc}")
        finally:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

st.markdown("#### LIME superpixel explanation")
st.write(
    "LIME perturbs image superpixels and estimates which regions support "
    "or oppose the selected prediction. This may take some time."
)
if st.button("Generate LIME explanation", key="generate_lime"):
    with st.spinner("Generating LIME explanation (300 perturbation samples)..."):
        lime_visual, lime_error = lime_explanation(
            model, rgb_image, preprocess, predicted_idx
        )
    if lime_visual is not None:
        st.image(
            lime_visual,
            caption="LIME superpixels: green supports, red opposes the selected class",
            use_container_width=True
        )
    else:
        st.warning(f"LIME could not be generated: {lime_error}")

st.markdown("### 4. Download prediction summary")
summary = {
    "predicted_class": predicted_class,
    "confidence_percent": round(confidence * 100, 2),
    "class_probabilities_percent": {
        classes[i]: round(float(probabilities[i] * 100), 2)
        for i in range(len(classes))
    },
    "explanation_note": (
        "Attribution maps show model influence, not a clinical diagnosis."
    )
}
st.download_button(
    "Download prediction summary (JSON)",
    data=json.dumps(summary, indent=2),
    file_name="histoxai_prediction.json",
    mime="application/json"
)

st.divider()
st.caption(
    "HistoxAI is an academic research prototype for colorectal histopathology "
    "image-tile classification. It is not validated for patient diagnosis, "
    "screening, treatment selection, or other clinical use."
)
