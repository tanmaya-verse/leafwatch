# LeafWatch - Plant stress monitoring using computer vision

Web app that analyses tomato/potato leaf images: segmentation, colour/lesion features,
stress category, top-3 possible diseases with confidence, Grad-CAM explanation,
differential-diagnosis questions, and a field-monitoring mode (heatmap, hotspots, CSV report).

## Run locally
1. Train in Google Colab with `train.py`, download `leaf_model.keras` and `class_names.json` into this folder.
2. `python -m venv venv` then activate it
3. `pip install -r requirements.txt`  (pin tensorflow to the version used in Colab)
4. `streamlit run app.py`

## Deploy
Push to GitHub, then create an app on share.streamlit.io with main file `app.py` (Python 3.11).

Results are possible causes only, not confirmed diagnoses.
