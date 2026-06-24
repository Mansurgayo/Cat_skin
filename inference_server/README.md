# Inference server

This folder contains a FastAPI inference server for serving Keras models.

Build and run with Docker (assumes model file is available locally):

```bash
# from workspace/inference_server
docker build -t cat-skin-inference .
docker run -e MODEL_PATH=/models/mobilenetv2_cat_skin_disease.h5 -v /path/to/models:/models -p 8000:8000 cat-skin-inference
```

Endpoint:
- POST /predict — form file upload (field name `file`). Returns JSON: {"predictions": [...]}.

Set environment variable `PREPROCESS_MODE` to `0-1` or `-1-1` if needed.

Streamlit deployment notes
-------------------------

The Streamlit app in the repository is designed to avoid installing `tensorflow` on Streamlit Cloud. To deploy successfully:

- Set the Streamlit secret/env `MODEL_API_URL` to point to your running inference server's `/predict` endpoint (if you host the `inference_server` Docker image somewhere).
- Alternatively, set `MODEL_URL` to a public URL of your model file so the app can download it at runtime (requires TF in environment; not recommended on Streamlit Cloud).
- If neither is set, the app runs in a demo mock mode returning deterministic placeholder predictions so the UI works without errors.

