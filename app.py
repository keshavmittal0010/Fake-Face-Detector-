import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import math
from scipy import fftpack
import pandas as pd

st.set_page_config(page_title="Fake Face Detector (Prototype)", layout="wide", page_icon="🕵️‍♂️")

st.markdown("""
<style>
.hero {
    background: linear-gradient(90deg, #0f172a, #0ea5e9);
    color: white;
    padding: 18px 24px;
    border-radius: 12px;
    margin-bottom: 14px;
}
.hero h1 { margin: 0; font-size: 28px; }
.hero p { margin: 4px 0 0 0; color: rgba(255,255,255,0.9); }
.stFileUploader { border-radius: 10px; }
div.stButton > button {
    background: linear-gradient(90deg,#06b6d4,#8b5cf6);
    color: white; border: none;
    padding: 8px 16px;
    border-radius: 10px;
    box-shadow: 0 6px 18px rgba(99,102,241,0.18);
}
.result-card {
    background: linear-gradient(180deg, #ffffff, #f8fafc);
    padding: 12px; border-radius: 10px;
    box-shadow: 0 6px 20px rgba(2,6,23,0.06);
}
.muted { color: #6b7280; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🕵️‍♂️ Fake Face Detector — Student Prototype</h1>
  <p>Lightweight demo using OpenCV and simple heuristics.</p>
</div>
""", unsafe_allow_html=True)

haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_xml)

def to_cv2_image(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def detect_faces(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))

def laplacian_variance(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def high_freq_energy(img_gray):
    f = fftpack.fft2(img_gray)
    fshift = fftpack.fftshift(f)
    mag = np.abs(fshift)
    h, w = img_gray.shape
    crow, ccol = h//2, w//2
    L = int(min(h,w) * 0.08)
    mask = np.ones_like(mag)
    mask[crow-L:crow+L, ccol-L:ccol+L] = 0
    return (mag * mask).sum() / (mag.sum() + 1e-9)

def color_hist_distance(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    left, right = hsv[:, :w//2], hsv[:, w//2:]
    hl = cv2.calcHist([left], [0,1], None, [50,60], [0,180,0,256])
    hr = cv2.calcHist([right], [0,1], None, [50,60], [0,180,0,256])
    cv2.normalize(hl, hl); cv2.normalize(hr, hr)
    return cv2.compareHist(hl, hr, cv2.HISTCMP_BHATTACHARYYA)

def analyze_face(face_bgr):
    face = cv2.resize(face_bgr, (224,224))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    lv = laplacian_variance(gray)
    hf = high_freq_energy(gray)
    ch = color_hist_distance(face)
    lv_score = np.tanh(lv / 150.0)
    hf_score = np.tanh(hf * 5.0)
    ch_score = 1.0 - math.exp(-ch * 6.0)
    fake_score = 0.6 * (1 - lv_score) + 0.25 * (1 - hf_score) + 0.15 * ch_score
    fake_score = max(0.0, min(1.0, fake_score))
    return {"laplacian_var": lv, "high_freq_energy": hf, "color_hist_dist": ch, "fake_score": fake_score}

def visualize_faces(img, faces, anns):
    img_copy = img.copy()
    for (x,y,w,h), a in zip(faces, anns):
        score = a["fake_score"]
        if score > 0.7:
            color = (0,0,255)
        elif score > 0.5:
            color = (0,165,255)
        else:
            color = (0,200,0)
        cv2.rectangle(img_copy, (x,y), (x+w,y+h), color, 3)
        cv2.putText(img_copy, f"{score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img_copy

col1, col2 = st.columns([1,2])
with col1:
    st.header("Upload & Settings")
    file = st.file_uploader("Upload image or video (≤10s).", type=["png","jpg","jpeg","mp4","mov","webm","avi"])
    st.markdown("<div class='muted'>Uses OpenCV Haar face detection and simple metrics.</div>", unsafe_allow_html=True)
    run_btn = st.button("Analyze")
    with st.expander("How it works"):
        st.write("It checks sharpness, texture, and color symmetry to estimate if a face looks real or AI-generated.")

with col2:
    st.header("Result")
    result_area = st.empty()

if not file:
    st.info("Upload an image or short video to begin.")
    st.stop()

if run_btn:
    progress = st.progress(0)
    name = file.name.lower()
    is_video = any(name.endswith(x) for x in [".mp4",".mov",".webm",".avi"])

    if not is_video:
        image = Image.open(file).convert("RGB")
        img_cv2 = to_cv2_image(image)
        faces = detect_faces(img_cv2)
        progress.progress(40)

        if len(faces) == 0:
            result_area.warning("No faces detected. Try a clearer image.")
        else:
            anns = [analyze_face(img_cv2[y:y+h, x:x+w]) for (x,y,w,h) in faces]
            vis = visualize_faces(img_cv2, faces, anns)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            with result_area.container():
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.image(vis_rgb, caption="Detected faces (scores shown)", use_column_width=True)
                data = [{"Face": f"Face {i+1}", "Fake Score": round(a['fake_score'],3),
                         "Sharpness": round(a['laplacian_var'],1),
                         "High-Freq Energy": round(a['high_freq_energy'],4),
                         "Color Diff": round(a['color_hist_dist'],4)} for i,a in enumerate(anns)]
                st.dataframe(pd.DataFrame(data))
                overall = max(a["fake_score"] for a in anns)
                label = "LIKELY FAKE" if overall > 0.5 else "LIKELY REAL"
                st.markdown(f"<h3><strong>Overall: {label}</strong> — Score: {overall:.3f}</h3>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            progress.progress(100)

    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(file.read())
        cap = cv2.VideoCapture(tmp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        inds = np.linspace(0, total-1, min(8,total)).astype(int)
        anns_all = []
        for i in inds:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: continue
            faces = detect_faces(frame)
            anns = [analyze_face(frame[y:y+h, x:x+w]) for (x,y,w,h) in faces]
            anns_all.append((frame, faces, anns))
        cap.release()

        if not anns_all:
            result_area.warning("No faces found in video.")
        else:
            imgs = []
            scores = []
            for f, faces, anns in anns_all:
                vis = visualize_faces(f, faces, anns)
                imgs.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                scores += [a["fake_score"] for a in anns]
            with result_area.container():
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.image(imgs, width=250)
                overall = max(scores) if scores else 0
                label = "LIKELY FAKE" if overall > 0.5 else "LIKELY REAL"
                st.markdown(f"<h3><strong>Overall Video: {label}</strong> — Score: {overall:.3f}</h3>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='muted'>Tip: Use clear, frontal faces for best results.</div>", unsafe_allow_html=True)
