# LeafWatch - run with:  streamlit run app.py
import json, math, datetime
import numpy as np, cv2, pandas as pd, streamlit as st, tensorflow as tf
from tensorflow import keras
from PIL import Image

st.set_page_config(page_title="LeafWatch", page_icon="🍃", layout="wide")
IMG, CONF_MIN = 224, 0.60

# ------------------------------------------------------------------ knowledge base
# keyword (matched inside the class name) -> (name, pathogen, visual clues, favoured by, next steps)
DISEASES = {
    "early_blight": ("Early blight", "Fungal (Alternaria solani)",
        "Brown spots with concentric rings and a yellow halo, mostly on older leaves",
        "Warm, humid weather; plant debris",
        "Remove infected leaves, rotate crops, improve airflow"),
    "late_blight": ("Late blight", "Oomycete (Phytophthora infestans)",
        "Water-soaked dark patches, white fuzz underneath in humid conditions",
        "Cool, wet, humid weather",
        "Isolate/remove infected plants quickly; consult an agronomist urgently"),
    "bacterial_spot": ("Bacterial spot", "Bacterial (Xanthomonas)",
        "Small dark, greasy-looking spots, sometimes with yellow halo",
        "Warm, wet, splashing rain",
        "Avoid overhead watering, use clean seed, consult an expert"),
    "leaf_mold": ("Leaf mold", "Fungal (Passalora fulva)",
        "Pale yellow patches on top, olive-grey mould underneath",
        "High humidity, poor ventilation", "Increase ventilation, reduce humidity"),
    "septoria": ("Septoria leaf spot", "Fungal (Septoria lycopersici)",
        "Many small circular spots with grey centres and dark edges",
        "Wet leaves, warm weather", "Remove lower infected leaves, avoid wetting foliage"),
    "spider_mites": ("Spider mite damage", "Pest (Tetranychus urticae)",
        "Fine yellow stippling, bronzing, sometimes webbing",
        "Hot, dry conditions", "Check leaf undersides; consult expert on control options"),
    "target_spot": ("Target spot", "Fungal (Corynespora cassiicola)",
        "Brown lesions with target-like rings", "Warm, humid conditions",
        "Improve airflow, remove infected leaves"),
    "yellowleaf_curl": ("Yellow leaf curl virus", "Virus (spread by whiteflies)",
        "Upward curling, yellowing, stunted leaves", "Whitefly populations",
        "Control whiteflies, remove infected plants, use resistant varieties"),
    "mosaic": ("Mosaic virus", "Virus (mechanical/seed spread)",
        "Mottled light/dark green mosaic pattern, distorted leaves",
        "Contact spread via hands/tools", "Sanitise tools, remove infected plants"),
}

def pretty(c): return c.replace("___", " - ").replace("__", " - ").replace("_", " ")
def info(cls):
    low = cls.lower()
    for k, v in DISEASES.items():
        if k in low: return v
    return None

@st.cache_resource
def load():
    return keras.models.load_model("leaf_model.keras"), json.load(open("class_names.json"))

# ------------------------------------------------------------------ image analysis
def quality_check(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    w = []
    if cv2.Laplacian(g, cv2.CV_64F).var() < 60: w.append("Image looks blurry")
    if g.mean() < 60: w.append("Image is too dark")
    if g.mean() > 200: w.append("Image is overexposed")
    return w

def segment(rgb):
    """Leaf mask + stress mask (yellow/brown) using HSV thresholds."""
    h, s, v = [c.astype(int) for c in cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))]
    green = (h >= 36) & (h <= 85) & (s > 40) & (v > 40)
    yellow = (h >= 20) & (h < 36) & (s > 60) & (v > 100)
    brown = (h >= 5) & (h < 20) & (s > 40) & (v > 30) & (v < 200)
    leaf = (green | yellow | brown).astype(np.uint8)
    k = np.ones((5, 5), np.uint8)
    leaf = cv2.morphologyEx(cv2.morphologyEx(leaf, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n = max(int(leaf.sum()), 1)
    y_pct = 100 * (yellow & (leaf > 0)).sum() / n
    b_pct = 100 * (brown & (leaf > 0)).sum() / n
    cnt, _, stats, _ = cv2.connectedComponentsWithStats((brown & (leaf > 0)).astype(np.uint8))
    lesions = int((stats[1:, cv2.CC_STAT_AREA] > 30).sum()) if cnt > 1 else 0
    return leaf, y_pct, b_pct, lesions

def predict(model, rgb):
    x = cv2.resize(rgb, (IMG, IMG)).astype("float32")
    x = keras.applications.mobilenet_v2.preprocess_input(x)[None]
    return model.predict(x, verbose=0)[0], x

def gradcam(model, x, idx):
    gm = keras.Model(model.input, [model.get_layer("out_relu").output, model.output])
    with tf.GradientTape() as t:
        conv, p = gm(tf.convert_to_tensor(x))
        loss = p[:, idx]
    g = t.gradient(loss, conv)
    w = tf.reduce_mean(g, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv * w[:, None, None, :], axis=-1)[0]).numpy()
    return cam / (cam.max() + 1e-8)

def overlay(rgb, cam):
    r = cv2.resize(rgb, (IMG, IMG))
    heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (IMG, IMG))), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(r, 0.6, heat, 0.4, 0)

def stage1(cls, sev):
    if "healthy" in cls.lower():
        return "Possible abiotic stress" if sev >= 15 else "Healthy"
    return "Biotic stress (possible disease)"

def analyse(model, names, rgb):
    leaf, y_pct, b_pct, les = segment(rgb)
    p, x = predict(model, rgb)
    order = np.argsort(p)[::-1]
    sev = y_pct + b_pct
    return dict(p=p, x=x, order=order, cls=names[order[0]], conf=float(p[order[0]]),
                sev=sev, y=y_pct, b=b_pct, les=les, cat=stage1(names[order[0]], sev))

def followup_rules(old_new, spreading, weather):
    hints = []
    if old_new == "Older leaves first":
        hints.append("Yellowing on older leaves can mean a mobile-nutrient deficiency (nitrogen, magnesium): the plant moves them to new growth.")
    if old_new == "Newer leaves first":
        hints.append("Yellowing on new leaves points to immobile nutrients (iron, zinc, sulphur) or a viral disease.")
    if weather == "Hot and dry":
        hints.append("Hot, dry weather with browning edges suggests water/heat stress (stomatal closure, reduced transpiration).")
    if weather == "Wet and humid" and spreading == "Yes":
        hints.append("Spreading spots in wet, humid weather make a fungal/oomycete disease more likely.")
    if spreading == "No" and weather != "Wet and humid":
        hints.append("Non-spreading symptoms make an abiotic (nutrient/water) cause more likely than an infection.")
    return hints or ["Answer the questions above to refine the diagnosis."]

# ------------------------------------------------------------------ UI
st.title("🍃 LeafWatch - plant stress monitoring")
st.caption("Tomato & potato leaves · results are *possible* causes, not confirmed diagnoses.")
import os
if not (os.path.exists("leaf_model.keras") and os.path.exists("class_names.json")):
    st.error("Model files not found. Run train.py in Colab, then place `leaf_model.keras` and "
             "`class_names.json` in the same folder as app.py.")
    st.stop()
model, names = load()
tab1, tab2, tab3, tab4 = st.tabs(["Single leaf", "Field monitoring", "Research", "Limitations"])

# ---------- TAB 1: single leaf
with tab1:
    up = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
    cam_in = st.camera_input("...or take a photo")
    f = up or cam_in
    if f:
        rgb = np.array(Image.open(f).convert("RGB"))
        for w in quality_check(rgb): st.warning(f"⚠️ {w} - consider retaking the photo.")
        r = analyse(model, names, rgb)
        c1, c2, c3 = st.columns(3)
        c1.image(rgb, caption="Uploaded", use_container_width=True)
        leaf, _, _, _ = segment(rgb)
        c2.image(leaf * 255, caption="Leaf segmentation", use_container_width=True)
        cam = gradcam(model, r["x"], int(r["order"][0]))
        c3.image(overlay(rgb, cam), caption="Grad-CAM: where the model looked", use_container_width=True)

        st.subheader(f"Stage 1: {r['cat']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Severity (leaf area affected)", f"{r['sev']:.1f}%")
        m2.metric("Yellowing / Browning", f"{r['y']:.0f}% / {r['b']:.0f}%")
        m3.metric("Lesions", r["les"])
        m4.metric("Model confidence", f"{r['conf']*100:.0f}%")

        if r["conf"] < CONF_MIN:
            st.error("Uncertain result - confidence is low. Please retake the photo (plain background, good light).")
        else:
            st.subheader("Stage 2: possible conditions (top 3)")
            for i in r["order"][:3]:
                st.write(f"**{pretty(names[i])}** - {r['p'][i]*100:.0f}%")
                st.progress(float(r["p"][i]))
            d = info(r["cls"])
            if d and "healthy" not in r["cls"].lower():
                st.subheader(f"Disease card: {d[0]}")
                st.table(pd.DataFrame({"": ["Pathogen type", "Visual clues", "Favoured by", "General next steps"],
                                       "Details": [d[1], d[2], d[3], d[4]]}).set_index(""))
            st.info("🧪 Biology note: yellowing (chlorosis) means chlorophyll loss, so less light is captured and "
                    "photosynthesis drops. Brown lesions are dead (necrotic) tissue where cells have been killed "
                    "by pathogens or severe stress.")

        st.subheader("Differential-diagnosis questions")
        q1, q2, q3 = st.columns(3)
        a = q1.selectbox("Which leaves show symptoms?", ["Not sure", "Older leaves first", "Newer leaves first"])
        b = q2.selectbox("Is it spreading to other plants?", ["Not sure", "Yes", "No"])
        c = q3.selectbox("Recent weather?", ["Normal", "Wet and humid", "Hot and dry"])
        for h in followup_rules(a, b, c): st.write("• " + h)
        st.warning("These are possible causes only. Consult an agronomist before any chemical treatment.")

# ---------- TAB 2: field monitoring
with tab2:
    st.write("Upload several photos, one per field zone, to simulate large-scale scouting.")
    files = st.file_uploader("Zone photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch")
    hot = st.slider("Hotspot threshold (% leaf area affected)", 5, 60, 25)
    gps = st.text_input("Field GPS tag (optional, e.g. 12.9716, 77.5946)")
    if files:
        rows = []
        for i, fl in enumerate(files):
            rgb = np.array(Image.open(fl).convert("RGB"))
            r = analyse(model, names, rgb)
            rows.append(dict(zone=f"Z{i+1}", file=fl.name, condition=pretty(r["cls"]),
                             stage1=r["cat"], confidence=round(r["conf"], 2),
                             severity_pct=round(r["sev"], 1), hotspot=r["sev"] >= hot))
        df = pd.DataFrame(rows)
        n = len(df); cols = math.ceil(math.sqrt(n)); rws = math.ceil(n / cols); S = 110
        grid = np.full((rws * S, cols * S, 3), 255, np.uint8)
        for i, row in df.iterrows():
            sv = row.severity_pct
            col = (60, 180, 75) if sv < 10 else (240, 200, 40) if sv < hot else (220, 50, 50)
            y0, x0 = (i // cols) * S, (i % cols) * S
            grid[y0 + 3:y0 + S - 3, x0 + 3:x0 + S - 3] = col
            cv2.putText(grid, row.zone, (x0 + 12, y0 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(grid, f"{sv:.0f}%", (x0 + 12, y0 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        st.image(grid, caption="Field heatmap: green = healthy, yellow = watch, red = hotspot")
        if df.hotspot.any():
            st.error(f"🚨 Hotspot alert: {', '.join(df[df.hotspot].zone)} need urgent inspection.")
        else:
            st.success("No hotspots above threshold.")
        st.subheader("Treatment priority (spot-treat the top zones, not the whole field)")
        pr = df.sort_values("severity_pct", ascending=False).reset_index(drop=True)
        pr.index += 1; st.dataframe(pr)

        # spread tracking across scans
        hist = st.session_state.setdefault("history", [])
        if st.button("Save this scan to history"):
            hist.append(dict(time=datetime.datetime.now().strftime("%H:%M:%S"),
                             mean_severity=round(df.severity_pct.mean(), 1), hotspots=int(df.hotspot.sum())))
        if hist:
            h = pd.DataFrame(hist); st.subheader("Spread tracking")
            st.line_chart(h.set_index("time")["mean_severity"])
            if len(h) > 1:
                d = h.mean_severity.iloc[-1] - h.mean_severity.iloc[-2]
                st.write(f"Change since previous scan: **{d:+.1f}%** ({'spreading' if d > 0 else 'improving'})")

        out = df.assign(scan_time=datetime.datetime.now().isoformat(timespec="seconds"), gps=gps)
        st.download_button("⬇️ Download report (CSV)", out.to_csv(index=False), "leafwatch_report.csv")

# ---------- TAB 3: research
with tab3:
    st.write("**Generalisation test:** upload your own phone photos, choose the true label, and compare with "
             "the dataset accuracy printed by `train.py`.")
    rf = st.file_uploader("Phone photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="res")
    if rf:
        recs = []
        for fl in rf:
            rgb = np.array(Image.open(fl).convert("RGB"))
            r = analyse(model, names, rgb)
            true = st.selectbox(f"True label for {fl.name}", names, format_func=pretty, key=fl.name)
            recs.append(dict(file=fl.name, true=pretty(true), predicted=pretty(r["cls"]),
                             confidence=round(r["conf"], 2), correct=(true == r["cls"])))
        d = pd.DataFrame(recs); st.dataframe(d)
        st.metric("Phone-photo accuracy", f"{100 * d.correct.mean():.0f}%")
        st.caption("Compare with dataset accuracy: the gap is your generalisation result. "
                   "Also test different lights/backgrounds and review Grad-CAM for background focus.")

# ---------- TAB 4: limitations
with tab4:
    st.markdown("""
**Supported:** tomato and potato leaves, conditions in the PlantVillage dataset.

**Limitations**
- Trained on lab-style images, so accuracy usually drops on cluttered field photos.
- Abiotic stress (nutrient/water/heat) has no trained class; it is flagged from colour features and follow-up questions.
- Visually similar conditions can be confused.
- Results are *possible* causes, never confirmed diagnoses. Consult an agronomist before chemical treatment.
- Field monitoring is simulated with per-zone photos, not drone mapping.
""")
