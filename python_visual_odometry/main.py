import argparse
import copy
from pathlib import Path

import cv2
import numpy as np

import camera
import frameData
import pose_estimator_gauss_newton

def get_available_indices(dataset_dir):
  image_dir = dataset_dir / "images"
  indices = sorted(int(path.stem.split("_")[-1]) for path in image_dir.glob("scene_*.png"))
  if not indices:
    raise FileNotFoundError(f"No dataset images found in {image_dir}")
  return indices

def load_invdepth_from_depth(depth_path):
  depth = np.load(depth_path).astype(np.float32)
  invDepth = np.zeros_like(depth, dtype=np.float32)
  valid = depth > 1e-6
  invDepth[valid] = 1.0 / depth[valid]
  invDepthVar = np.ones_like(depth, dtype=np.float32)
  invDepthVar[~valid] = 1e6
  return invDepth, invDepthVar

def parse_args():
  parser = argparse.ArgumentParser(description="Run the dense monocular visual odometry demo.")
  parser.add_argument(
      "--dataset",
      default="dataset/desktop_dataset",
      help="Path to the dataset directory containing images/ and poses/.",
  )
  parser.add_argument(
      "--depth-dir",
      default=None,
      help="Optional path to the directory containing depth .npy files. Defaults to <dataset>/depth.",
  )
  parser.add_argument(
      "--max-frames",
      type=int,
      default=None,
      help="Optional limit on the number of frames to process.",
  )
  parser.add_argument(
      "--no-display",
      action="store_true",
      help="Disable OpenCV windows. Useful for testing on headless environments.",
  )
  return parser.parse_args()

def main():
  args = parse_args()
  dataset_dir = Path(args.dataset)
  image_dir = dataset_dir / "images"
  depth_dir = Path(args.depth_dir) if args.depth_dir else dataset_dir / "depth"
  indices = get_available_indices(dataset_dir)
  if args.max_frames is not None:
    indices = indices[:args.max_frames]
  if not indices:
    raise ValueError("No frames selected.")

  width = 640
  height = 480
  fx = 481.20
  fy = 480.0
  cx = 319.5
  cy = 239.5

  frame = frameData.frameData()
  cam = camera.camera(fx,fy,cx,cy,width,height)
  pose_gn = pose_estimator_gauss_newton.pose_estimator_gauss_newton(cam, show_debug = not args.no_display)
  keyframe = None

  print(f"Processing {len(indices)} frame(s) from {dataset_dir}")
  print(f"Using depth maps from {depth_dir}")

  for step, imIndex in enumerate(indices):
    image_path = image_dir / f"scene_{imIndex:03d}.png"
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
      raise FileNotFoundError(f"Could not read image: {image_path}")

    if step == 0:
      depth_path = depth_dir / f"scene_{imIndex:03d}_depth.npy"
      if not depth_path.exists():
        raise FileNotFoundError(f"Could not find depth map for keyframe: {depth_path}")
      invDepth, invDepthVar = load_invdepth_from_depth(depth_path)
      frame.setImage(image)
      keyframe = copy.deepcopy(frame)
      keyframe.setInvDepth(invDepth, invDepthVar)
      print(f"[frame {imIndex:03d}] loaded depth map {depth_path.name}")
    else:
      frame.setImage(image)
      print(f"[frame {imIndex:03d}] estimating pose")
      pose_gn.optPose(frame, keyframe)

    if not args.no_display:
      cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
      cv2.imshow("frame", frame.image[1])
      cv2.namedWindow("invdepth", cv2.WINDOW_NORMAL)
      cv2.imshow("invdepth", keyframe.invDepth[1])
      cv2.namedWindow("invdepthVar", cv2.WINDOW_NORMAL)
      cv2.imshow("invdepthVar", np.sqrt(keyframe.invDepthVar[1]) * 10.0)
      cv2.waitKey(30)

  if not args.no_display:
    cv2.destroyAllWindows()

  print("Run completed successfully.")

if __name__ == "__main__":
  main()
