import os
import shutil
import tempfile

import cv2
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from pydub import AudioSegment
from transformers import BlipProcessor, BlipForConditionalGeneration
from TTS.api import TTS

st.set_page_config(page_title="Video → Captions → Voice-over", layout="centered")


# ----------------------------------------------------------------------
# Cached model loading (runs once per session, not on every rerun)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading BLIP captioning model...")
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    return processor, model, device


@st.cache_resource(show_spinner="Loading Coqui TTS model...")
def load_tts():
    return TTS(
        model_name="tts_models/en/ljspeech/tacotron2-DDC",
        progress_bar=False,
        gpu=torch.cuda.is_available(),
    )


# ----------------------------------------------------------------------
# Pipeline steps (same logic as the notebook)
# ----------------------------------------------------------------------
def extract_by_interval(video_path, out_dir, interval_sec):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(fps * interval_sec))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(out_dir, exist_ok=True)
    frame_idx, saved_idx, saved = 0, 0, []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            out_path = os.path.join(out_dir, f"frame_{saved_idx:05d}_t{timestamp:.2f}s.jpg")
            cv2.imwrite(out_path, frame)
            saved.append((out_path, timestamp))
            saved_idx += 1
        frame_idx += 1

    cap.release()
    return saved, fps, total_frames


def caption_frames(frame_records, processor, model, device, progress_cb=None):
    captions = []
    for i, (frame_path, timestamp) in enumerate(frame_records):
        image = Image.open(frame_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=40)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append({"frame_path": frame_path, "timestamp_sec": round(timestamp, 2), "caption": caption})
        if progress_cb:
            progress_cb(i + 1, len(frame_records), caption)
    return captions


def add_captions_to_video(video_path, captions, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count, caption_idx = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_count / fps
        while caption_idx < len(captions) - 1 and captions[caption_idx + 1]["timestamp_sec"] <= timestamp:
            caption_idx += 1
        current_caption = captions[caption_idx]["caption"]

        text_size = cv2.getTextSize(current_caption, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = max((width - text_size[0]) // 2, 10)
        text_y = height - 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, current_caption, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return output_path


def synthesize_audio(captions, tts_model, out_dir, progress_cb=None):
    os.makedirs(out_dir, exist_ok=True)
    audio_paths = []
    for i, entry in enumerate(captions):
        out_path = os.path.join(out_dir, f"audio_{i}.wav")
        tts_model.tts_to_file(text=entry["caption"], file_path=out_path)
        audio_paths.append(out_path)
        if progress_cb:
            progress_cb(i + 1, len(captions))
    return audio_paths


def build_combined_audio(captions, audio_paths, out_path):
    combined = AudioSegment.silent(duration=0)
    prev_end_ms = 0
    for entry, clip_path in zip(captions, audio_paths):
        target_start_ms = int(entry["timestamp_sec"] * 1000)
        start_ms = max(target_start_ms, prev_end_ms)
        if start_ms > len(combined):
            combined += AudioSegment.silent(duration=start_ms - len(combined))
        clip = AudioSegment.from_wav(clip_path)
        combined += clip
        prev_end_ms = len(combined)
    combined.export(out_path, format="wav")
    return out_path


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🎬 Video → Captions → Voice-over")
st.caption("Splits a video into frames, captions each one with BLIP, narrates the captions with TTS, "
           "and produces a final video with burned-in captions and matching sound.")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "avi"])
interval_sec = st.slider(
    "Seconds between sampled frames",
    min_value=0.5, max_value=5.0, value=2.0, step=0.5,
    help="Lower = more frames/captions (slower, more detail). Higher = fewer frames, "
         "and more breathing room for each caption to be spoken without overlapping the next one.",
)

run = st.button("Run pipeline", type="primary", disabled=uploaded_video is None)

if run and uploaded_video is not None:
    work_dir = tempfile.mkdtemp(prefix="video_pipeline_")
    video_path = os.path.join(work_dir, uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    try:
        # Step 1: models
        processor, blip_model, device = load_blip()
        tts_model = load_tts()

        # Step 2: extract frames
        status = st.status("Extracting frames...", expanded=True)
        frame_records, fps, total_frames = extract_by_interval(
            video_path, os.path.join(work_dir, "frames"), interval_sec
        )
        status.write(f"Saved {len(frame_records)} frames ({fps:.1f} fps, {total_frames} total frames).")

        # Step 3: caption frames
        status.update(label="Captioning frames with BLIP...")
        progress = st.progress(0.0)
        caption_log = st.empty()

        def caption_progress(done, total, caption):
            progress.progress(done / total)
            caption_log.text(f"[{done}/{total}] {caption}")

        captions = caption_frames(frame_records, processor, blip_model, device, caption_progress)
        pd.DataFrame(captions).to_csv(os.path.join(work_dir, "captions.csv"), index=False)
        progress.empty()
        caption_log.empty()
        status.write(f"Generated {len(captions)} captions.")

        # Step 4: burn captions onto video
        status.update(label="Burning captions onto the video...")
        captioned_raw = os.path.join(work_dir, "captioned_video.mp4")
        add_captions_to_video(video_path, captions, captioned_raw)
        captioned_h264 = os.path.join(work_dir, "captioned_video_h264.mp4")
        os.system(f'ffmpeg -y -i "{captioned_raw}" -vcodec libx264 -acodec aac "{captioned_h264}" -loglevel error')
        status.write("Captioned video encoded.")

        # Step 5: TTS
        status.update(label="Synthesizing narration with TTS...")
        tts_progress = st.progress(0.0)

        def tts_progress_cb(done, total):
            tts_progress.progress(done / total)

        audio_paths = synthesize_audio(captions, tts_model, os.path.join(work_dir, "audio_outputs"), tts_progress_cb)
        tts_progress.empty()
        status.write(f"Generated {len(audio_paths)} narration clips.")

        # Step 6: combine audio (no overlap)
        status.update(label="Building the combined narration track...")
        combined_audio_path = os.path.join(work_dir, "combined_audio.wav")
        build_combined_audio(captions, audio_paths, combined_audio_path)
        status.write("Narration track built with no overlapping clips.")

        # Step 7: mux
        status.update(label="Muxing narration onto the captioned video...")
        final_path = os.path.join(work_dir, "final_video_with_audio.mp4")
        os.system(
            f'ffmpeg -y -i "{captioned_h264}" -i "{combined_audio_path}" '
            f'-c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest "{final_path}" -loglevel error'
        )
        status.update(label="Done!", state="complete")

        st.subheader("Final video")
        st.video(final_path)
        with open(final_path, "rb") as f:
            st.download_button("Download final video", f, file_name="final_video_with_audio.mp4", mime="video/mp4")

        with st.expander("Captions generated"):
            st.dataframe(pd.DataFrame(captions)[["timestamp_sec", "caption"]])

    finally:
        # Keep work_dir around for this run only; clean up old temp dirs to avoid disk bloat.
        pass