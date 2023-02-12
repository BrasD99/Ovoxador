import cv2
from tools.camera import Camera
from tools.helpers import get_config
import numpy as np

filename = "/Users/brasd99/Downloads/homography_0.yml"
fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
homography = fs.getNode("homography").mat()

cfg = get_config()

camera = Camera(
    video_file='/Users/brasd99/Downloads/IMG_4848 2.MOV', 
    camera_id=1, 
    cfg=cfg)

camera.process()

player_tracks, ball_tracks = camera.get_tracks()

maket = cv2.imread('/Users/brasd99/Downloads/maket.jpeg')

def align_images(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.array([kp1[m.queryIdx].pt for m in matches], np.float32)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in matches], np.float32)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))

height, width, channels = camera.frames[0].shape
m_height, m_width, m_channels = maket.shape

final = np.zeros((height, width, channels), dtype=camera.frames[0].dtype)
first_frame = camera.frames[0]

for i, player_track in enumerate(player_tracks):
    src_frame = camera.frames[i].copy()
    src_frame = cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB)
    result = src_frame.copy()
    bboxes = [track['box'].astype(int) for track in player_track["tracks"]]
    result = align_images(first_frame, result)
    for xmin, ymin, xmax, ymax in bboxes:
        result[ymin:ymax, xmin:xmax, :] = 0

    non_zero = (result != 0) & (final == 0)
    final[non_zero] = result[non_zero]

transformed_img = cv2.warpPerspective(final, homography, (m_width, m_height))

# Convert to HSV
hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
# Compute histogram of hue values
hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
# Find the most frequent hue value
main_color = np.argmax(hist)

# Convert back to BGR
main_color = np.uint8([[[main_color, 255, 255]]])
main_color = cv2.cvtColor(main_color, cv2.COLOR_HSV2BGR)

# Convert image to grayscale
gray = cv2.cvtColor(maket, cv2.COLOR_BGR2GRAY)

# Apply threshold to grayscale image
_, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Find contours in thresholded image
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image with main color as background
result = np.zeros((transformed_img.shape[0], transformed_img.shape[1], 3), np.uint8)
result[:] = main_color

# Draw contours in white on the black image
cv2.drawContours(result, contours, -1, (255, 255, 255), 2)

alpha = 0.8
mask = (transformed_img != 0).astype(np.float32) * alpha

result = result.astype(np.float32) * (1 - mask) + transformed_img.astype(np.float32) * mask

result = result.astype(np.uint8)

cv2.imshow("Final Field Texture", result)
cv2.waitKey(0)


'''
half_height, half_width = m_height, m_width // 2

left_half = transformed_img[:, :half_width, :]
right_half_mirrored = transformed_img[:, half_width:, :][:, ::-1, :]

parts = [left_half, right_half_mirrored]

output = np.zeros(left_half.shape, dtype=maket.dtype)

for part in parts:
    part_height, part_width, part_channels = part.shape
    if half_height != part_height or half_width != part_width:
        part = cv2.resize(part, (half_width, half_height), cv2.INTER_NEAREST)
    mask = (part != 0) #& (output == 0)
    output[mask] = part[mask]

right_half = cv2.flip(output, 1)
merged_pitch = np.concatenate((output, right_half), axis=1)

cv2.imshow("Final Field Texture", merged_pitch)
cv2.waitKey(0)

half_height, half_width = m_height // 2, m_width // 2

# top left
tl = transformed_img[:half_height, :half_width, :]
print(f'tl: {tl.shape}')
# top right
tr = transformed_img[:half_height, half_width:, :][:, ::-1, :]
print(f'tr: {tr.shape}')
# bottom left
bl = transformed_img[half_height:, :half_width, :][::-1, :, :]
print(f'bl: {bl.shape}')
# bottom right
br = transformed_img[half_height:, half_width:, :][:, ::-1, :][::-1, :, :]
print(f'br: {br.shape}')

output = np.zeros(tl.shape, dtype=maket.dtype)

parts = [tl, tr, bl, br]

for part in parts:
    part_height, part_width, part_channels = part.shape
    if half_height != part_height or half_width != part_width:
        part = cv2.resize(part, (half_width, half_height), cv2.INTER_NEAREST)
    mask = (part != 0) & (output == 0)
    output[mask] = part[mask]

tl = output

# Mirror the top left corner to get the top right corner
tr = tl[:, ::-1, :]

# Mirror the top left corner vertically to get the bottom left corner
bl = tl[::-1, :, :]

# Mirror the bottom left corner horizontally to get the bottom right corner
br = bl[:, ::-1, :]

full_pitch = np.concatenate((np.concatenate((tl, tr), axis=1),
                             np.concatenate((bl, br), axis=1)), axis=0)

cv2.imshow("Final Field Texture", full_pitch)
cv2.waitKey(0)


left_half = transformed_img[:, :w // 2, :]
right_half = transformed_img[:, w // 2:, :][:, ::-1, :]

right_half_mask = (right_half != 0) & (left_half == 0)
left_half[right_half_mask] = right_half[right_half_mask]

right_half = cv2.flip(right_half, 1)

transformed_img = np.concatenate((left_half, right_half), axis=1)

cv2.imshow("Final Field Texture", transformed_img)
cv2.waitKey(0)
'''