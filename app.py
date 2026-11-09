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
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------- Visual system -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --red:#e32636; --red2:#ff4b55; --ink:#0b0b0d; --muted:#9a9aa2; --line:#29292e; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.stApp { background:
 radial-gradient(ellipse at 76% 15%, rgba(227,38,54,.12), transparent 31%),
 radial-gradient(ellipse at 12% 72%, rgba(227,38,54,.07), transparent 28%),
 #09090b; color:#f5f5f6; }
.block-container { max-width: 1220px; padding-top: 1.3rem; padding-bottom: 4rem; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
div[data-testid="stToolbar"] { visibility:hidden; height:0; }
.nav-label { font-family:'DM Mono',monospace; color:#777780; letter-spacing:.18em; font-size:.68rem; text-transform:uppercase; }
.brand { font-weight:800; letter-spacing:-.06em; font-size:1.35rem; color:#fff; }
.brand span { color:var(--red); }
.topline { border-bottom:1px solid var(--line); padding: .2rem 0 1.1rem 0; margin-bottom:2rem; }
.hero-kicker { font-family:'DM Mono',monospace; color:#ff6872; letter-spacing:.22em; font-size:.72rem; text-transform:uppercase; }
.hero-title { font-size:clamp(3.2rem,7vw,6.4rem); line-height:.94; letter-spacing:-.075em; font-weight:800; color:#fff; margin:.9rem 0 1.2rem; }
.hero-title em { color:var(--red); font-style:normal; }
.hero-copy { color:#b0b0b8; font-size:1.06rem; line-height:1.8; max-width:650px; }
.hero-panel { min-height:430px; border:1px solid #29292e; border-radius:26px; position:relative; overflow:hidden;
 background:radial-gradient(circle at 50% 48%,rgba(227,38,54,.23),transparent 22%),
 radial-gradient(circle at 55% 55%,#1a1115 0%,#101013 44%,#0b0b0d 78%); }
.orbit { position:absolute; border:1px solid rgba(255,255,255,.14); border-radius:50%; transform:rotate(-22deg); }
.orbit.one { width:330px;height:150px;left:12%;top:34%; }
.orbit.two { width:440px;height:205px;left:0%;top:27%; }
.orbit.three { width:250px;height:250px;left:31%;top:22%; border-color:rgba(227,38,54,.35); }
.core { position:absolute; width:112px;height:112px; border-radius:50%; left:43%;top:39%;
 background:radial-gradient(circle at 32% 28%,#ff9aa1,#e32636 43%,#690b17 100%);
 box-shadow:0 0 55px rgba(227,38,54,.45), inset -12px -14px 25px rgba(0,0,0,.35); }
.node { position:absolute; width:17px;height:17px;border-radius:50%; background:#f4f4f5; box-shadow:0 0 18px rgba(255,255,255,.3); }
.node.red { background:var(--red); box-shadow:0 0 22px rgba(227,38,54,.7); }
.node.n1 {left:18%;top:35%}.node.n2 {left:72%;top:28%;width:11px;height:11px}.node.n3 {left:78%;top:68%;width:23px;height:23px}.node.n4 {left:29%;top:76%;width:9px;height:9px}.node.n5 {left:61%;top:79%;width:12px;height:12px}
.panel-tag { position:absolute; left:25px; bottom:23px; color:#b9b9c0; font: .7rem 'DM Mono',monospace; letter-spacing:.13em; text-transform:uppercase; }
.section-kicker { font: .7rem 'DM Mono',monospace; color:#ff6872; letter-spacing:.2em; text-transform:uppercase; }
.section-title { font-size:clamp(2rem,4vw,3.25rem); letter-spacing:-.06em; font-weight:800; color:#fff; margin:.4rem 0 1rem; }
.body-muted { color:#aaaab2; line-height:1.8; }
.info-card { background:linear-gradient(145deg,#151518,#101013); border:1px solid #2a2a30; border-radius:18px; padding:24px; min-height:170px; }
.info-card .num { color:var(--red); font: .75rem 'DM Mono',monospace; }
.info-card h3 { color:#fff; font-size:1.15rem; margin:.75rem 0 .5rem; }
.info-card p { color:#a7a7b0; line-height:1.65; font-size:.92rem; }
.stat { border-top:1px solid #34343a; padding-top:15px; }
.stat strong { display:block; color:#fff; font-size:2rem; letter-spacing:-.05em; }
.stat span { color:#9a9aa2; font: .67rem 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase; }
.step { border-left:1px solid #4b252b; padding:4px 0 22px 22px; margin-left:8px; position:relative; }
.step:before { content:''; position:absolute; width:10px;height:10px;border-radius:50%;background:var(--red);left:-5.5px;top:7px;box-shadow:0 0 15px rgba(227,38,54,.65); }
div[data-testid="stRadio"] > label { display:none; }
div[data-testid="stRadio"] div[role="radiogroup"] { gap:8px; justify-content:flex-end; }
div[data-testid="stRadio"] div[role="radiogroup"] label { background:#111114; border:1px solid #29292e; border-radius:30px; padding:7px 17px; }
div[data-testid="stRadio"] div[role="radiogroup"] label p { color:#bcbcc3; font-size:.8rem; }
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) { background:#351116; border-color:#a51c2b; }
div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p { color:#fff; }
.stButton > button, .stDownloadButton > button { background:var(--red); color:white; border:1px solid var(--red); border-radius:30px; padding:.65rem 1.25rem; font-weight:700; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#bf1728; border-color:#bf1728; color:white; }
div[data-testid="stFileUploader"] { background:#111114; border:1px dashed #55515a; border-radius:16px; padding:12px; }
div[data-testid="stMetric"] { background:#141417; border:1px solid #2b2b31; border-radius:14px; padding:16px; }
div[data-testid="stMetricLabel"] { color:#a5a5ad; }
div[data-testid="stMetricValue"] { color:#fff; }
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #303036; }
.stTabs [data-baseweb="tab"] { background:#131316; border-radius:10px 10px 0 0; color:#bbb; }
.stTabs [aria-selected="true"] { color:#fff !important; border-bottom:2px solid var(--red) !important; }
hr { border-color:#2c2c32; }
.small-note { color:#898992; font-size:.78rem; line-height:1.7; }
</style>
""", unsafe_allow_html=True)

# Top brand and page navigation
head_left, head_right = st.columns([1, 2])
with head_left:
    st.markdown('<div class="topline"><div class="brand">HISTOX<span>AI</span></div><div class="nav-label">Computational pathology / research prototype</div></div>', unsafe_allow_html=True)
with head_right:
    page = st.radio(
        "Navigation",
        ["Overview", "Our Approach", "Prediction"],
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation",
    )
    st.markdown('<div class="topline"></div>', unsafe_allow_html=True)

DEFAULT_CHECKPOINT = Path("colorectal_output/colorectal_inceptionv3_final.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    # Keep CPU resource use predictable on hosted instances.
    torch.set_num_threads(2)

# ----------------------------- Model utilities -----------------------------
@st.cache_resource
def load_model(checkpoint_bytes):
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
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return model, classes, preprocess, input_size

def get_model_scores(model, batch):
    output = model(batch)
    return output.logits if hasattr(output, "logits") else output

def prepare_image(pil_image, preprocess):
    rgb = pil_image.convert("RGB")
    return rgb, preprocess(rgb).unsqueeze(0).to(DEVICE)

def make_gradcam(model, input_tensor, target_class, image_size):
    model.eval()
    gradcam = LayerGradCam(model, model.Mixed_7c)
    x = input_tensor.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    attribution = gradcam.attribute(x, target=target_class)
    attribution = LayerAttribution.interpolate(attribution, image_size, interpolate_mode="bilinear")
    cam = np.maximum(attribution.squeeze().detach().cpu().numpy(), 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam

def make_integrated_gradients(model, input_tensor, target_class):
    ig = IntegratedGradients(model)
    attribution = ig.attribute(
        input_tensor,
        baselines=torch.zeros_like(input_tensor),
        target=target_class,
        n_steps=16,
        internal_batch_size=1
    )
    heat = np.abs(attribution.squeeze(0).detach().cpu().numpy()).sum(axis=0)
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    else:
        heat = np.zeros_like(heat)
    return heat

def overlay_heatmap(rgb_image, heatmap, alpha=0.42):
    h, w = rgb_image.shape[:2]
    heatmap = cv2.resize(heatmap.astype(np.float32), (w, h))
    colored = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip((1 - alpha) * rgb_image + alpha * colored, 0, 255))

def region_boxes(cam, image_shape, threshold=0.65):
    h, w = image_shape[:2]
    resized = cv2.resize(cam.astype(np.float32), (w, h))
    mask = np.uint8(resized >= threshold) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
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
        cv2.rectangle(canvas, (x, y), (x+w, y+h), (235, 40, 55), 2)
        cv2.circle(canvas, (x+14, y+14), 12, (235, 40, 55), -1)
        cv2.putText(canvas, str(idx), (x+10, y+19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)
    return canvas

def lime_explanation(model, image_rgb, preprocess, target_class):
    """Create a class-specific LIME explanation using memory-safe inference."""
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries, slic

        model.eval()

        def predict_fn(images):
            # LIME supplies RGB images in the [0, 1] range.
            outputs = []
            chunk_size = 4

            for batch_start in range(0, len(images), chunk_size):
                image_batch = images[batch_start:batch_start + chunk_size]
                tensors = []

                for image_array in image_batch:
                    image_array = np.asarray(image_array)
                    if image_array.dtype != np.uint8:
                        image_array = np.uint8(
                            np.clip(image_array, 0.0, 1.0) * 255.0
                        )
                    pil = Image.fromarray(image_array).convert("RGB")
                    tensors.append(preprocess(pil))

                batch_tensor = torch.stack(tensors).to(DEVICE)

                with torch.inference_mode():
                    scores = get_model_scores(model, batch_tensor)
                    batch_probabilities = torch.softmax(scores, dim=1)

                outputs.append(batch_probabilities.cpu().numpy())
                del batch_tensor, scores, batch_probabilities

            if not outputs:
                raise ValueError("LIME did not provide any perturbed images.")

            return np.concatenate(outputs, axis=0)

        # SLIC is generally more predictable and less expensive than the
        # default quickshift segmentation for hosted CPU environments.
        def segmentation_fn(image):
            return slic(
                image,
                n_segments=60,
                compactness=10,
                sigma=1,
                start_label=0,
                channel_axis=-1
            )

        image_array = np.asarray(image_rgb.convert("RGB"), dtype=np.uint8)
        lime_input = image_array.astype(np.float32) / 255.0

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            lime_input,
            predict_fn,
            labels=(int(target_class),),
            top_labels=None,
            hide_color=0,
            num_samples=120,
            batch_size=4,
            segmentation_fn=segmentation_fn,
            random_seed=42
        )

        temp, mask = explanation.get_image_and_mask(
            int(target_class),
            positive_only=False,
            num_features=8,
            hide_rest=False
        )

        visual = mark_boundaries(
            np.clip(temp, 0.0, 1.0),
            mask,
            color=(1, 0, 0),
            mode="thick"
        )
        return np.asarray(visual, dtype=np.float32), None

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

# ----------------------------- Page: Overview -----------------------------
if page == "Overview":
    st.markdown('<div class="hero-kicker">A closer look at colorectal tissue</div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown('<div class="hero-title">SEE THE<br>DETAIL.<br><em>UNDERSTAND</em><br>THE TISSUE.</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="hero-copy">
        Colorectal cancer begins in the colon or rectum. Histopathology examines
        tissue under a microscope, helping pathologists study cellular patterns
        and tissue architecture. <b>HistoxAI</b> is an academic AI project that
        explores how deep learning can classify colorectal histology image tiles
        and make model behaviour easier to inspect.
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.caption("Select **Our Approach** in the navigation above to explore the workflow.")
    with right:
        st.markdown("""
        <div class="hero-panel">
          <div class="orbit one"></div><div class="orbit two"></div><div class="orbit three"></div>
          <div class="core"></div>
          <div class="node n1"></div><div class="node red n2"></div>
          <div class="node n3"></div><div class="node red n4"></div><div class="node n5"></div>
          <div class="panel-tag">Histology / AI / Explainability</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-kicker">01 / The context</div><div class="section-title">Understanding colorectal cancer</div>', unsafe_allow_html=True)
    st.markdown('<p class="body-muted">Colorectal cancer is a disease in which abnormal cells develop in the colon or rectum. Many cases begin as growths called polyps. Screening can help detect precancerous polyps and cancer early. Diagnosis is made by qualified healthcare professionals using clinical assessment and tests such as colonoscopy and tissue biopsy.</p>', unsafe_allow_html=True)
    cards = st.columns(3)
    card_data = [
        ("01", "What happens", "Cells in the lining of the colon or rectum can grow abnormally and form a tumour. Tissue samples allow microscopic examination of these changes."),
        ("02", "Why histology matters", "A biopsy sample is processed into thin sections and stained so cellular details and tissue structures can be examined under a microscope."),
        ("03", "Why early detection matters", "Screening and timely medical assessment can identify changes earlier. New symptoms or concerns should be discussed with a qualified clinician."),
    ]
    for col, (num, title, desc) in zip(cards, card_data):
        with col:
            st.markdown(f'<div class="info-card"><div class="num">{num} / INSIGHT</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("")
    s1, s2, s3 = st.columns(3)
    for col, value, label in [
        (s1, "8", "Histology tissue classes"),
        (s2, "InceptionV3", "Deep learning backbone"),
        (s3, "XAI", "Model explanation methods"),
    ]:
        with col:
            st.markdown(f'<div class="stat"><strong>{value}</strong><span>{label}</span></div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown('<div class="small-note">Educational overview only. This project does not provide medical advice, screening, diagnosis, or treatment recommendations. Consult a healthcare professional for personal medical concerns.</div>', unsafe_allow_html=True)

# ----------------------------- Page: Approach -----------------------------
elif page == "Our Approach":
    st.markdown('<div class="section-kicker">02 / Methodology</div><div class="section-title">From tissue tile to model insight.</div>', unsafe_allow_html=True)
    st.markdown('<p class="body-muted">HistoxAI follows a computer-vision workflow: prepare labelled histology tiles, train a convolutional neural network, evaluate its classification performance, and inspect the regions that influence its predictions.</p>', unsafe_allow_html=True)

    a, b, c = st.columns(3)
    for col, n, title, desc in [
        (a, "01", "Dataset", "The colorectal histology dataset contains 5,000 image tiles across eight tissue categories, with 625 tiles per category."),
        (b, "02", "Learning", "An InceptionV3 convolutional neural network is fine-tuned to learn visual patterns from the histopathology image tiles."),
        (c, "03", "Explainability", "Grad-CAM, Integrated Gradients and LIME are used to inspect image regions that influence model outputs."),
    ]:
        with col:
            st.markdown(f'<div class="info-card"><div class="num">{n} / STAGE</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-kicker">The pipeline</div><div class="section-title">A four-stage workflow</div>', unsafe_allow_html=True)
    steps = [
        ("Data preparation", "Organise the eight labelled classes, resize images to the model input dimensions, and apply the same normalization used during training."),
        ("Model training", "Fine-tune InceptionV3 in a two-stage training process using the prepared training split."),
        ("Model evaluation", "Evaluate on the held-out tile test split using accuracy, precision, recall, F1-score, balanced accuracy and ROC-AUC."),
        ("Visual explanation", "Generate attribution visualisations to explore which regions influence the predicted class."),
    ]
    for title, desc in steps:
        st.markdown(f'<div class="step"><b style="color:#fff">{title}</b><div class="body-muted">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-kicker">Dataset classes</div><div class="section-title">Eight tissue categories</div>', unsafe_allow_html=True)
    class_names = ["01 · Tumor", "02 · Stroma", "03 · Complex", "04 · Lymphocytes",
                   "05 · Debris", "06 · Mucosa", "07 · Adipose", "08 · Empty"]
    cols = st.columns(4)
    for i, (col, name) in enumerate(zip(cols * 2, class_names)):
        with col:
            st.markdown(f'<div class="info-card" style="min-height:92px;padding:16px"><div class="num">CLASS {i+1:02d}</div><h3 style="margin:.35rem 0 0">{name}</h3></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-kicker">Reported experiment results</div><div class="section-title">Model performance</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in [
        (m1, "92.53%", "Test accuracy"),
        (m2, "0.9249", "Macro F1-score"),
        (m3, "0.9253", "Balanced accuracy"),
        (m4, "0.9938", "Macro ROC-AUC"),
    ]:
        with col:
            st.markdown(f'<div class="stat"><strong>{val}</strong><span>{label}</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="small-note">These are the previously reported results for the current experiment. The split is tile-level, not patient- or slide-level; therefore these results do not establish generalisation to independent patients or clinical readiness. The project is a research prototype only.</p>', unsafe_allow_html=True)

    st.caption("Select **Prediction** in the navigation above to test an image.")

# ----------------------------- Page: Prediction -----------------------------
elif page == "Prediction":
    st.markdown('<div class="section-kicker">03 / Interactive laboratory</div><div class="section-title">Prediction & explanation</div>', unsafe_allow_html=True)
    st.markdown('<p class="body-muted">Upload a colorectal histology image tile to view the InceptionV3 prediction, class scores, and visual explanations from multiple attribution methods.</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Model checkpoint")
        uploaded_checkpoint = st.file_uploader("Upload trained .pth checkpoint", type=["pth"])
        st.caption(f"Compute device: {DEVICE}")

    if uploaded_checkpoint is not None:
        checkpoint_bytes = uploaded_checkpoint.getvalue()
    elif DEFAULT_CHECKPOINT.exists():
        checkpoint_bytes = DEFAULT_CHECKPOINT.read_bytes()
    else:
        checkpoint_bytes = None

    if checkpoint_bytes is None:
        st.warning("The trained model checkpoint was not found. Place `colorectal_inceptionv3_final.pth` inside the `colorectal_output` folder beside `app.py`, or upload it in the sidebar.")
        st.stop()

    try:
        model, classes, preprocess, input_size = load_model(checkpoint_bytes)
    except Exception as exc:
        st.error(f"Could not load this checkpoint: {exc}")
        st.stop()

    st.caption(f"Model ready · InceptionV3 · {input_size} × {input_size} input · {len(classes)} classes")
    uploaded_image = st.file_uploader(
        "Upload a histopathology image tile",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        key="histology_image",
    )
    if uploaded_image is None:
        st.info("Choose a histology image to begin. The prediction and explanation panels will appear below.")
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

    st.markdown("#### Model output")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.image(rgb_image, caption="Uploaded histology tile", use_container_width=True)
    with right:
        st.metric("Predicted tissue class", predicted_class)
        st.metric("Model confidence", f"{confidence * 100:.2f}%")
        if confidence < 0.60:
            st.warning("The model confidence is relatively low. Treat this output as uncertain.")
        st.caption("Confidence is the model's softmax score, not a calibrated probability of clinical correctness.")

    st.markdown("#### Class probability distribution")
    sorted_indices = np.argsort(probabilities)[::-1]
    st.bar_chart({"Probability (%)": {classes[i]: float(probabilities[i] * 100) for i in sorted_indices}})

    st.markdown("---")
    st.markdown("#### Explainable AI")
    st.markdown('<p class="small-note">These visualisations describe which image regions influenced this model output. They do not establish a diagnosis or prove that a highlighted region contains a particular tissue structure.</p>', unsafe_allow_html=True)
    # Generate explanations only when requested. Prediction remains lightweight.
    original_array = np.asarray(rgb_image)

    st.markdown("#### Grad-CAM explanation")
    if st.button("Generate Grad-CAM", key="generate_gradcam"):
        with st.spinner("Generating Grad-CAM..."):
            try:
                gradcam = make_gradcam(
                    model, input_tensor, predicted_idx,
                    (input_size, input_size)
                )
                grad_overlay = overlay_heatmap(original_array, gradcam)
                _, regions = region_boxes(gradcam, original_array.shape)
                region_image = draw_regions(original_array, regions)

                tab1, tab2 = st.tabs(["Grad-CAM", "Region highlights"])
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
                        "Higher-activation areas indicate stronger relative "
                        "attribution for the selected predicted class."
                    )
                with tab2:
                    st.image(
                        region_image,
                        caption="Top high-activation regions (numbered)",
                        use_container_width=True
                    )
                    if regions:
                        for idx, (x, y, w, h, area) in enumerate(
                            regions, start=1
                        ):
                            st.write(
                                f"**Region {idx}:** location ({x}, {y}), "
                                f"size {w} × {h} pixels. This area has relatively "
                                f"high Grad-CAM activation for '{predicted_class}'."
                            )
                    else:
                        st.info(
                            "No distinct connected high-activation regions "
                            "were found at the current threshold."
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

    st.markdown("#### Integrated Gradients explanation")
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
                        ig_map,
                        caption="Integrated Gradients attribution",
                        use_container_width=True
                    )
                with c2:
                    st.image(
                        ig_overlay,
                        caption="Integrated Gradients overlay",
                        use_container_width=True
                    )
                st.caption(
                    "Attribution indicates relative influence on the model "
                    "output; it is not a verified tissue or tumor boundary."
                )
            except Exception as exc:
                st.error(f"Integrated Gradients generation failed: {exc}")
            finally:
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

        st.markdown("#### LIME superpixel explanation")
        st.markdown('<p class="small-note">LIME perturbs image superpixels and estimates which regions support or oppose the selected prediction. This may take some time.</p>', unsafe_allow_html=True)
        if st.button("Generate LIME explanation"):
            with st.spinner("Generating class-specific LIME explanation (120 samples)..."):
                lime_visual, lime_error = lime_explanation(model, rgb_image, preprocess, predicted_idx)
            if lime_visual is not None:
                st.image(lime_visual, caption="LIME superpixels: green supports, red opposes the selected class", use_container_width=True)
            else:
                st.warning(f"LIME could not be generated: {lime_error}")

    summary = {
        "predicted_class": predicted_class,
        "confidence_percent": round(confidence * 100, 2),
        "class_probabilities_percent": {
            classes[i]: round(float(probabilities[i] * 100), 2) for i in range(len(classes))
        },
        "explanation_note": "Attribution maps show model influence, not a clinical diagnosis."
    }
    st.markdown("---")
    st.download_button(
        "Download prediction summary (JSON)",
        data=json.dumps(summary, indent=2),
        file_name="histoxai_prediction.json",
        mime="application/json",
    )

st.markdown("---")
st.markdown('<div class="small-note">HISTOXAI · Academic research prototype · Not validated for patient diagnosis, screening, treatment selection, or clinical use.</div>', unsafe_allow_html=True)
