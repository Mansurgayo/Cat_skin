import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
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

    img = Image.open(uploaded)
    # make image larger and put visualization to the right
    image_col, viz_col = st.columns([3, 2])
    with image_col:
        st.image(img, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Loading model and predicting..."):
            model_path = get_model_path(model_choice)
            if not model_path or not os.path.exists(model_path):
                st.error(f"Model file not found for {model_choice}.")
                return
            try:
                model = load_keras_model(model_path)
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return

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
