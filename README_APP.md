# Video Deepfake Detection Prototype

Dataset source: Kaggle  
https://www.kaggle.com/datasets/kanzeus/realai-video-dataset

This project is a small prototype for testing a visual-odometry-style pipeline on real and AI-generated videos.

![AI video preview](assets/ai_19_preview.gif)

## Setup

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Launch the interface

```powershell
python -m streamlit run python_visual_odometry\app.py
```

You can also use:

```powershell
launch_visual_odometry_app.bat
```

## Notebooks

- `python_visual_odometry/Detection_DF.ipynb`  
  Test the pipeline on a single video or a single pair of frames.

- `python_visual_odometry/Detection_DF_scaling.ipynb`  
  Run the same idea on a larger part of the dataset and export aggregated statistics.

## Notes

- The first run may download the `LiheYoung/depth_anything_vitb14` model.
- Extracted frames are saved in `*_frames_spaced` folders or in the output folder chosen by the notebook/app.
- Depth cache files are stored in `python_visual_odometry/depth_anything_cache/vitb`.
