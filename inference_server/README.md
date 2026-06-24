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
