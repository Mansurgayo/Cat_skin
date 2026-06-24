import os
import io
import streamlit as st
try:
    import tensorflow as tf
    HAVE_TF = True
except Exception:
    tf = None
    HAVE_TF = False
from PIL import Image
import numpy as np
import requests
from urllib.parse import urlparse


def remote_predict(api_url, image_bytes):
    """Send image bytes to remote model API. Expects JSON with key 'predictions' or 'probs'."""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        resp = requests.post(api_url, files=files, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # try common keys
        for key in ("predictions", "probs", "probabilities", "preds"):
            if key in data:
                arr = np.asarray(data[key], dtype=float)
                return arr
        # if response is a list
        if isinstance(data, list):
            return np.asarray(data, dtype=float)
        # fallback: look for single probability
        if "prediction" in data:
            return np.asarray([data["prediction"]], dtype=float)
    except Exception:
        return None


def mock_predict(image_bytes, n_classes=4):
    """Deterministic mock prediction derived from image bytes.
    Returns a numpy array of probabilities summing to 1.
    """
    import hashlib

    h = hashlib.sha256(image_bytes).digest()
    # use first n_classes bytes to build scores
    scores = np.frombuffer(h[: n_classes], dtype=np.uint8).astype(np.float32)
    # avoid zeros
    scores = scores + 1.0
    probs = scores / scores.sum()
    return probs


def download_model_from_env():
    """Download model from environment variable MODEL_URL to model folder.
    Optional env var MODEL_FILENAME sets the target filename.
    Returns local path if downloaded or already exists, else None.
    """
    url = os.environ.get("MODEL_URL")
    if not url:
        return None
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        filename = os.environ.get("MODEL_FILENAME")
        if not filename:
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path) or None
        # fallback to first known model filename
        if not filename:
            filename = next(iter(MODEL_DISPLAY_NAMES.values()), None)
        if not filename:
            return None
        save_path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(save_path):
            return save_path
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return save_path
    except Exception:
        return None
import pandas as pd


MODEL_DIR = "model"
MODEL_DISPLAY_NAMES = {
    "Mobilenet": "mobilenetv2_cat_skin_disease.h5",
    "Efisiennet": "efficientnetb1_cat_skin_disease_final.keras",
}

# Per-model default labels (adjust if you have different mappings)
MODEL_LABELS = {
    "mobilenetv2_cat_skin_disease.h5": ["Flea_Allergy", "Health", "Ringworm", "Scabies"],
    "efficientnetb1_cat_skin_disease_final.keras": ["Flea_Allergy", "Health", "Ringworm", "Scabies"],
}


@st.cache_data
def load_keras_model(path):
    return tf.keras.models.load_model(path, compile=False)


def get_model_input_size(model):
    try:
        shape = model.input_shape
    except Exception:
        try:
            shape = model.inputs[0].shape
        except Exception:
            shape = None
    if not shape:
        return (224, 224)
    # shape may be (None, H, W, C) or (H, W, C)
    try:
        if len(shape) >= 3:
            h = int(shape[1]) if shape[1] is not None else 224
            w = int(shape[2]) if shape[2] is not None else 224
            return (h, w)
    except Exception:
        pass
    return (224, 224)


def preprocess_image(img: Image.Image, target_size, mode: str = "0-1"):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32)
    if mode == "0-1":
        arr = arr / 255.0
    elif mode == "-1-1":
        arr = (arr / 255.0 - 0.5) * 2.0
    elif mode == "model":
        # send raw pixel values (no scaling) - used when model contains Rescaling layer
        pass
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, img_arr):
    preds = model.predict(img_arr)
    preds = np.asarray(preds)
    if preds.ndim == 2:
        probs = preds[0]
    else:
        probs = preds.flatten()
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])
    return top_idx, top_prob, probs


DEFAULT_LABELS = ["Flea_Allergy", "Health", "Ringworm", "Scabies"]


def list_model_files():
    if not os.path.isdir(MODEL_DIR):
        return []
    files = []
    for f in os.listdir(MODEL_DIR):
        if f.lower().endswith(('.h5', '.hdf5', '.keras')):
            files.append(f)
    return sorted(files)


def get_model_path(display_name):
    filename = MODEL_DISPLAY_NAMES.get(display_name)
    if filename:
        return os.path.join(MODEL_DIR, filename)
    return None


def main():
    st.title("Skin Disease Classifier — Streamlit")
    st.sidebar.header("Model & Labels")

    available_models = [name for name in MODEL_DISPLAY_NAMES]
    model_choice = st.sidebar.selectbox("Select model", available_models)

    # remote inference API URL (can be set as env MODEL_API_URL in Streamlit)
    default_api = os.environ.get("MODEL_API_URL", "")
    api_url = st.sidebar.text_input("Remote inference API URL (optional)", value=default_api)

    # optional: allow user to provide public URL to download model file
    sel_filename = MODEL_DISPLAY_NAMES.get(model_choice)
    model_url_input = st.sidebar.text_input("Model file URL (optional)", value="")
    if model_url_input:
        if st.sidebar.button("Download model"):
            try:
                os.makedirs(MODEL_DIR, exist_ok=True)
                parsed = urlparse(model_url_input)
                # save to configured filename
                save_path = os.path.join(MODEL_DIR, sel_filename)
                with requests.get(model_url_input, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(save_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                st.sidebar.success(f"Downloaded model to {save_path}")
            except Exception as e:
                st.sidebar.error(f"Download failed: {e}")

    # preprocessing option
    preprocess_mode = st.sidebar.selectbox("Preprocessing", ["Auto", "0-1", "-1-1"], help="Auto detects model Rescaling; 0-1 divides by 255; -1-1 scales to [-1,1]")

    # option to use per-model default labels
    use_model_labels = st.sidebar.checkbox("Use model default labels (if available)", value=True)

    # determine default label text
    sel_filename = MODEL_DISPLAY_NAMES.get(model_choice)
    if use_model_labels and sel_filename and sel_filename in MODEL_LABELS:
        default_labels_text = "\n".join(MODEL_LABELS[sel_filename])
    else:
        default_labels_text = "\n".join(DEFAULT_LABELS)

    labels_text = st.sidebar.text_area(
        "Labels (one per line, index order)",
        value=default_labels_text,
        height=120,
    )
    labels = [l.strip() for l in labels_text.splitlines() if l.strip()]

    # debug controls
    if st.sidebar.button("Show model summary"):
        model_path_dbg = get_model_path(model_choice)
        if not model_path_dbg or not os.path.exists(model_path_dbg):
            st.sidebar.error("Model file not found for selected model.")
        else:
            try:
                m = load_keras_model(model_path_dbg)
                st.sidebar.write(f"Loaded: {model_path_dbg}")
                # capture summary
                summary_lines = []
                m.summary(print_fn=lambda s: summary_lines.append(s))
                st.sidebar.text('\n'.join(summary_lines))
                # last layer info
                try:
                    last = m.layers[-1]
                    st.sidebar.write(f"Last layer: {type(last).__name__}, output shape: {last.output_shape}")
                except Exception:
                    pass
            except Exception as e:
                st.sidebar.error(f"Failed to load model: {e}")

    st.sidebar.markdown("---")
    st.sidebar.write("Model files in `model/` folder:")
    for display_name, filename in MODEL_DISPLAY_NAMES.items():
        st.sidebar.write(f"- **{display_name}**: {filename}")

    st.markdown("---")
    st.markdown("### Upload Image")
    # centered uploader (full width of the middle column)
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"]) 

    if uploaded is None:
        st.info("Upload an image in the center area and click Predict.")
        return

    # read uploaded bytes early so we can use them for remote/mock predictions
    uploaded_bytes = uploaded.read()
    img = Image.open(io.BytesIO(uploaded_bytes))
    # make image larger and put visualization to the right
    image_col, viz_col = st.columns([3, 2])
    with image_col:
        st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Loading model and predicting..."):
            # try download from MODEL_URL env var if present
            _ = download_model_from_env()
            model_path = get_model_path(model_choice)

            model = None
            if HAVE_TF and model_path and os.path.exists(model_path):
                try:
                    model = load_keras_model(model_path)
                except Exception as e:
                    st.error(f"Failed to load local model: {e}")

            # if no local model, try remote API
            used_remote = False
            remote_probs = None
            if model is None:
                if api_url:
                    remote_probs = remote_predict(api_url, uploaded_bytes)
                    if remote_probs is not None:
                        used_remote = True
                else:
                    # no API configured and no TF model available — use mock mode
                    remote_probs = mock_predict(uploaded_bytes, n_classes=len(labels) if labels else 4)
                    used_remote = True

            if model is None and not used_remote:
                st.error(f"Model file not available and no remote API configured for {model_choice}.")
                return

            if used_remote:
                # we received probabilities directly from remote or mock
                probs = np.asarray(remote_probs, dtype=float)
                # ensure size matches labels if provided
                n = probs.size
                results = {"remote": (int(np.argmax(probs)), float(np.max(probs)), probs)}
                modes_to_test = ["remote"]
                mode_to_apply = "remote"
                target_size = (224, 224)
                st.write("Running in remote/mock mode — local TensorFlow model not used.")
            else:
                target_size = get_model_input_size(model)
                st.write("Model input shape:", getattr(model, 'input_shape', 'unknown'))

                # detect if model contains a Rescaling layer
                try:
                    has_rescaling = any((layer.__class__.__name__.lower().find('rescaling') != -1) for layer in model.layers)
                except Exception:
                    has_rescaling = False

                # detect last layer activation name
                try:
                    last = model.layers[-1]
                    act = getattr(last, 'activation', None)
                    act_name = act.__name__ if callable(act) else str(act)
                except Exception:
                    act_name = 'unknown'

                # choose preprocessing mode
                if preprocess_mode == 'Auto':
                    if has_rescaling:
                        mode_to_apply = 'model'  # model contains Rescaling; send raw pixel values
                    else:
                        mode_to_apply = '0-1'
                else:
                    mode_to_apply = preprocess_mode

                st.write(f"Detected Rescaling layer: {has_rescaling}, last activation: {act_name}, using preprocess: {mode_to_apply}")

                # prepare test modes: always compare 0-1 and -1-1; include 'model' if model has Rescaling
                modes_to_test = ["0-1", "-1-1"]
                if has_rescaling:
                    # include 'model' so user can see model-rescaling behavior
                    modes_to_test.insert(0, "model")

                results = {}
                for m in modes_to_test:
                    try:
                        arr = preprocess_image(img, target_size, mode=m)
                        # show basic diagnostics for the selected mode only
                        if m == mode_to_apply:
                            st.write("Image array shape:", arr.shape)
                            st.write(f"Image min/max: {arr.min():.6f} / {arr.max():.6f}")
                        i_idx, i_prob, i_probs = predict(model, arr)
                        results[m] = (i_idx, i_prob, i_probs)
                    except Exception as e:
                        results[m] = (None, None, None)

            # choose effective labels based on one of the results (they should share size)
            sample_probs = next((v[2] for v in results.values() if v[2] is not None), None)
            n = sample_probs.size if sample_probs is not None else 0
            if labels and len(labels) == n:
                eff_labels = labels
            elif len(DEFAULT_LABELS) == n:
                eff_labels = DEFAULT_LABELS
            else:
                eff_labels = [f"Class {i}" for i in range(n)]
                if len(labels) != n:
                    st.warning("Label count does not match model output; using generated labels.")

            # show comparison columns
            cols = st.columns(len(modes_to_test))
            for idx_col, m in enumerate(modes_to_test):
                col = cols[idx_col]
                col.subheader(f"Mode: {m}" + (" (selected)" if m == mode_to_apply else ""))
                r = results.get(m)
                if r is None or r[0] is None:
                    col.write("Failed")
                    continue
                i_idx, i_prob, i_probs = r
                name = eff_labels[i_idx] if i_idx < len(eff_labels) else f"Class {i_idx}"
                col.success(f"Top: {name} — {i_prob*100.0:.2f}%")
                # top-3
                k = min(3, i_probs.size)
                top_idx = np.argsort(i_probs)[-k:][::-1]
                for j in top_idx:
                    lbl = eff_labels[j] if j < len(eff_labels) else f"Class {j}"
                    col.write(f"{lbl} — {i_probs[j]*100.0:.2f}%")
                if i_probs.size <= 50:
                    dfm = pd.DataFrame({"Label": [eff_labels[i] for i in range(len(i_probs))], "Probability": i_probs * 100.0})
                    dfm = dfm.set_index("Label")["Probability"]
                    col.bar_chart(dfm)
                else:
                    col.write("(Detailed probabilities hidden)")

            # use results for the selected mode to present the detailed table below
            chosen = results.get(mode_to_apply)
            if chosen is None or chosen[2] is None:
                st.error("Selected preprocessing mode failed to produce predictions.")
                return
            idx, prob, probs = chosen

            # compute label name for chosen result
            label_name = eff_labels[idx] if idx < len(eff_labels) else f"Class {idx}"

            # Show results below the image area
            st.markdown("---")
            st.subheader("Results")
            pct = prob * 100.0
            st.success(f"Prediction: {label_name} — {pct:.2f}%")

            if probs is not None:
                # build DataFrame with label names and percent probabilities
                df_probs = pd.DataFrame(
                    {
                        "Label": [eff_labels[i] if i < len(eff_labels) else f"Class {i}" for i in range(len(probs))],
                        "Probability": (probs * 100.0),
                    }
                )
                df_probs["Probability"] = df_probs["Probability"].round(2)

                st.markdown("**Raw probability table**")
                st.table(df_probs.sort_values("Probability", ascending=False).reset_index(drop=True))

                # top-k summary
                k = min(3, probs.size)
                top_idx = np.argsort(probs)[-k:][::-1]
                st.markdown(f"**Top-{k} predictions**")
                for i in top_idx:
                    name = eff_labels[i] if i < len(eff_labels) else f"Class {i}"
                    st.write(f"{name} — {probs[i]*100.0:.2f}%")

                # bar chart visualization (only for reasonable class counts)
                if probs.size <= 50:
                    st.bar_chart(df_probs.set_index("Label"))
                else:
                    st.write("(Detailed probabilities hidden for large output.)")


if __name__ == "__main__":
    main()
