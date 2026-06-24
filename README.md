# Streamlit app for two Keras models

Place your Keras model files inside the `model/` folder (e.g. `.h5` or `.keras`). This workspace already contains:

- `model/efficientnetb1_cat_skin_disease_final.keras`
- `model/mobilenetv2_cat_skin_disease.h5`

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Usage:
- Select a model from the dropdown
- (Optional) Paste label names one per line in the labels box
- Upload an image and click Predict
