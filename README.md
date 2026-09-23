# Video → Captions → Voice-over

A Streamlit app that takes a video, captions it with an image-captioning model, narrates those captions with text-to-speech, and produces a final video with burned-in captions and matching narration audio.

## Pipeline

1. **Extract frames** from the uploaded video at a fixed interval.
2. **Caption each frame** using BLIP (`Salesforce/blip-image-captioning-base`).
3. **Burn the captions** onto the video as on-screen text.
4. **Synthesize narration** for each caption with Coqui TTS (`tts_models/en/ljspeech/tacotron2-DDC`).
5. **Stitch the narration clips** into a single audio track, timed to the captions and without overlap.
6. **Mux the narration** onto the captioned video to produce the final output.

## Requirements

- Python 3.10 or 3.11
- [FFmpeg](https://ffmpeg.org/download.html) installed and available on your system `PATH`
- A GPU is recommended (BLIP + TTS are much slower on CPU) but not required

## Setup

```bash
pip install -r requirements.txt
```

### Installing FFmpeg (Windows)

```powershell
winget install ffmpeg
```

After installing, close and reopen your terminal (or restart your machine if it's still not recognized), then verify with:

```powershell
ffmpeg -version
```

## Running the app

```bash
streamlit run app.py
```

If `streamlit` isn't recognized as a command, run it through Python instead:

```bash
python -m streamlit run app.py
```

This opens the app at `http://localhost:8501`.

## Usage

1. Upload a video file (`.mp4`, `.mov`, `.mkv`, `.avi`).
2. Adjust the **seconds between sampled frames** slider — lower values give more detail but more overlap risk in narration; higher values give the narration more breathing room per caption.
3. Click **Run pipeline** and watch the progress for each step.
4. Preview and download the final video once processing finishes.

## Notes

- All intermediate files (frames, per-caption audio clips, captions CSV) are written to a temporary working directory per run.
- The narration track never overlaps itself: if a caption takes longer to speak than the gap to the next one, the following clip is delayed rather than played on top of it.
- To use a different TTS voice, change the `model_name` in `load_tts()` inside `app.py`. Run `TTS().list_models()` to see available options.
