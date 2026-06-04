import cv2
import numpy as np


def load_depth_npy(path, point=None, invert=True):
	depth = np.load(path)

	if point is not None:
		y, x = point
		return float(depth[y, x])

	depth_min = float(depth.min())
	depth_max = float(depth.max())
	if depth_max == depth_min:
		img = np.zeros_like(depth, dtype=np.uint8)
	else:
		norm = (depth - depth_min) / (depth_max - depth_min)
		if invert:
			norm = 1.0 - norm
		img = (norm * 255.0).astype(np.uint8)

	return img


depth_path = 'C:\\Users\\conqu\\Videos\\Detection_Deep_Fake\\Github_clone\\Depth-Anything\\assets\\output\\scene_000_depth.npy'

depth_img = load_depth_npy(depth_path)
cv2.imwrite('C:\\Users\\conqu\\Videos\\Detection_Deep_Fake\\Github_clone\\Depth-Anything\\assets\\output\\scene_0005_depth.png', depth_img)
print(depth_img)
print(depth_img.shape, depth_img.dtype)

point_value = load_depth_npy(depth_path, point=(0, 0))
print(point_value)
