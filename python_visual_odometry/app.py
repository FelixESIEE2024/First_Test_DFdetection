from __future__ import annotations

import json

import streamlit as st

from video_pipeline import DEVICE, DEPTH_MODEL_NAME, run_video_pipeline


st.set_page_config(
    page_title="Video Depth Pose Pipeline",
    page_icon="ðŸŽžï¸",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(181, 82, 51, 0.10), transparent 28%),
            radial-gradient(circle at top right, rgba(32, 91, 90, 0.08), transparent 26%),
            #f5efe6;
        color: #1f2933;
    }
    .stApp,
    .stApp p,
    .stApp li,
    .stApp label,
    .stApp div,
    .stApp span,
    .stMarkdown,
    .stText,
    .stAlert,
    .stMetric,
    .st-emotion-cache-10trblm,
    .st-emotion-cache-16idsys,
    .st-emotion-cache-1kyxreq {
        color: #1f2933 !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #163534, #224847);
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="base-input"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] [role="slider"] {
        color: #ffffff !important;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(255,250,242,0.96), rgba(247,239,228,0.94));
        border: 1px solid #d9cfc0;
        border-radius: 24px;
        padding: 22px 26px;
        box-shadow: 0 18px 48px rgba(54, 39, 28, 0.08);
        margin-bottom: 1rem;
    }
    .hero-card h1,
    .hero-card p {
        color: #1f2933 !important;
    }
    .metric-card {
        background: rgba(255,250,242,0.96);
        border: 1px solid #d9cfc0;
        border-radius: 18px;
        padding: 14px 16px;
    }
    code {
        background: rgba(32, 91, 90, 0.08) !important;
        color: #1f2933 !important;
        border-radius: 6px;
        padding: 0.12rem 0.35rem;
    }
    .stCode,
    .stCode pre,
    .stCode code {
        color: #f8fafc !important;
    }
    input, textarea {
        color: #1f2933 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
      <h1>Video â†’ depth â†’ pose â†’ error map pipeline</h1>
      <p>
        This interface follows the logic of <code>New_version_nb.ipynb</code>:
        you only provide a video path, the app extracts frames, computes the keyframe depth
        with <code>Depth-Anything</code>, converts it to inverse depth, estimates the relative pose,
        and then displays the useful maps together with the <code>mean squared error</code>.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Input")
    video_path = st.text_input("Video path", placeholder=r"C:\...\my_video.mp4")
    output_dir = st.text_input(
        "Frame output directory (optional)",
        placeholder="Leave empty to use <video>_frames_spaced",
    )

    st.header("Options")
    frame_count = st.slider("Number of exported frames", min_value=2, max_value=20, value=10, step=1)
    target_frame_index = st.slider(
        "Target frame index",
        min_value=1,
        max_value=max(1, frame_count - 1),
        value=1,
        step=1,
    )
    min_gap = st.slider("Minimum gap between two frames", min_value=1, max_value=120, value=10, step=1)
    start_frame = st.slider("Starting video frame", min_value=0, max_value=500, value=0, step=1)
    resize_width = st.slider(
        "Optional resize width (0 = original width)",
        min_value=0,
        max_value=1920,
        value=0,
        step=1,
    )
    use_cache = st.checkbox("Reuse depth map cache", value=True)

    run_clicked = st.button("Analyze video", type="primary", use_container_width=True)


def render_results(result: dict) -> None:
    metrics = result["metrics"]
    intrinsics = result["intrinsics"]
    total_pixels = intrinsics["width"] * intrinsics["height"]

    a, b, c, d = st.columns(4)
    a.metric("Photometric MSE", f"{metrics['mean_squared_error']:.6f}")
    b.metric("Valid pixels", f"{metrics['valid_reprojected_pixels']} / {total_pixels}")
    c.metric("Final error lvl 2", f"{metrics['final_error_lvl2']:.6f}")
    d.metric("Depth valid ratio", f"{metrics['depth_valid_ratio']:.3f}")

    st.markdown("### Summary")
    st.markdown(
        f"""
        - Video: `{result["video_path"]}`
        - Exported frames: `{len(result["frame_paths"])}` in `{result["frames_dir"]}`
        - Keyframe / target: `scene_000` â†’ `scene_{result["target_index"]:03d}`
        - Selected video indices: `{result["selected_video_indices"]}`
        - Detected FPS: `{result["fps"]:.3f}`
        - Depth model: `{DEPTH_MODEL_NAME}` on `{DEVICE}`
        - Estimated intrinsics: `fx={intrinsics["fx"]:.2f}`, `fy={intrinsics["fy"]:.2f}`, `cx={intrinsics["cx"]:.2f}`, `cy={intrinsics["cy"]:.2f}`
        - Valid reprojected pixels: `{metrics["valid_reprojected_pixels"]} / {total_pixels}`
        """
    )

    st.markdown("### Visualizations")
    col1, col2, col3 = st.columns(3)
    col1.image(result["visuals"]["keyframe_image"], caption="Keyframe", use_container_width=True)
    col2.image(result["visuals"]["target_image"], caption="Target frame", use_container_width=True)
    col3.image(result["visuals"]["depth_map"], caption="Depth map", use_container_width=True)

    col4, col5, col6 = st.columns(3)
    col4.image(result["visuals"]["inverse_depth_map"], caption="Inverse depth", use_container_width=True)
    col5.image(result["visuals"]["squared_difference_map"], caption="Squared difference map", use_container_width=True)
    col6.image(result["visuals"]["residual_error_map_lvl2"], caption="Residual error map lvl 2", use_container_width=True)

    st.markdown("### Extracted frames")
    gallery_columns = st.columns(min(5, len(result["frame_paths"])))
    for index, (image_path, caption) in enumerate(result["extracted_gallery"]):
        gallery_columns[index % len(gallery_columns)].image(image_path, caption=caption, use_container_width=True)

    st.markdown("### Estimated pose")
    st.code(result["pose_matrix_text"], language="text")

    st.markdown("### Estimated intrinsic matrix")
    st.code(result["intrinsics_matrix_text"], language="text")

    st.markdown("### Detailed metrics")
    payload = {
        "metrics": result["metrics"],
        "intrinsics": result["intrinsics"],
        "total_pixel_count": total_pixels,
        "total_pixel_count_formula": f'{intrinsics["width"]} x {intrinsics["height"]} = {total_pixels}',
        "selected_video_indices": result["selected_video_indices"],
        "selected_timestamps_seconds": result["selected_timestamps_seconds"],
    }
    st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")

    st.markdown("### Solver logs")
    st.text(result["logs"] or "No additional logs.")


if run_clicked:
    if not video_path.strip():
        st.error("Please enter a video path first.")
    else:
        try:
            with st.spinner("Extracting frames, estimating depth, and computing pose..."):
                result = run_video_pipeline(
                    video_path=video_path.strip(),
                    output_dir=output_dir.strip() or None,
                    frame_count=frame_count,
                    target_frame_index=target_frame_index,
                    min_gap=min_gap,
                    start_frame=start_frame,
                    resize_width=resize_width or None,
                    use_cache=use_cache,
                    save_cache=True,
                )
            st.success("Analysis completed.")
            render_results(result)
        except Exception as exc:
            st.error(str(exc))
else:
    st.info("Enter a video path in the sidebar, then start the analysis.")
