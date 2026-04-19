import cv2 as cv
import numpy as np
from harris import harris, non_maximum_suppression
# from skimage.feature import hog
# from scipy.ndimage import uniform_filter

def my_hog_blcok(patch, pixels_per_cell=(8,8)):
    gx = cv.Sobel(patch, cv.CV_32F, 1, 0, ksize=1)
    gy = cv.Sobel(patch, cv.CV_32F, 0, 1, ksize=1)

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (np.arctan2(gy, gx) * (180 / np.pi)) % 360
    orientations = 9
    per_bin = 360 // orientations

    h, w = patch.shape
    cell_h, cell_w = pixels_per_cell
    n_cells_h = h // cell_h
    n_cells_w = w // cell_w

    histogram = np.zeros((n_cells_h, n_cells_w, orientations), dtype=np.float32)

    for i in range(orientations):
        mask = (orientation >= i * per_bin) & (orientation < (i + 1) * per_bin)
        histogram[:, :, i] += cv.boxFilter(magnitude * (orientation / per_bin - i) * mask.astype(np.float32),
                                          -1, (cell_w, cell_h), normalize=False)[cell_h//2::cell_h, cell_w//2::cell_w]
        last = (i-1+orientations) % orientations
        if i == 0:
            now = orientations
        else:
            now = i
        mask = (orientation > last * per_bin) & (orientation <= now * per_bin)
        histogram[:, :, i] += cv.boxFilter(magnitude * (now - orientation / per_bin) * mask.astype(np.float32),
                                          -1, (cell_w, cell_h), normalize=False)[cell_h//2::cell_h, cell_w//2::cell_w]

    result = histogram.flatten()
    result = result / (np.linalg.norm(result) + 1e-6)
    return result


def get_dominant_angle(patch):
    gx = cv.Sobel(patch, cv.CV_32F, 1, 0, ksize=1)
    gy = cv.Sobel(patch, cv.CV_32F, 0, 1, ksize=1)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (np.arctan2(gy, gx) * (180 / np.pi)) % 360

    # 36个bin的方向直方图
    hist, _ = np.histogram(orientation, bins=36, range=(0, 360),
                           weights=magnitude)
    dominant_bin = np.argmax(hist)
    return dominant_bin * 10


def hog_feature(gray, kps, patch_size=16):
    descriptors = []
    valid_kp = []

    half = patch_size // 2

    for kp in kps:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        if x-half < 0 or y-half < 0 or x+half >= gray.shape[1] or y+half >= gray.shape[0]:
            continue

        patch = gray[y-half:y+half, x-half:x+half]

        angle = get_dominant_angle(patch)
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        ys, xs = np.mgrid[-half:half, -half:half] 
        xs_rot = (xs * cos_a - ys * sin_a + x).astype(np.float32)
        ys_rot = (xs * sin_a + ys * cos_a + y).astype(np.float32)
        patch = cv.remap(gray,
                        xs_rot, ys_rot,
                        interpolation=cv.INTER_LINEAR,
                        borderMode=cv.BORDER_REFLECT)# patch[i,j]对应原图中(x+xs_rot[i,j], y+ys_rot[i,j])的像素值
        # print(patch.shape)

        feature = my_hog_blcok(patch, pixels_per_cell=(4, 4))
        descriptors.append(feature)
        valid_kp.append(kp)

    return np.array(descriptors, dtype=np.float32), valid_kp

def hog_match_rotate(gray1, gray2):
    R = harris(gray1, blockSize=3, ksize=3, k=0.06, gassian=True)
    corners = non_maximum_suppression(R, threshold=0.01)
    kp1 = [cv.KeyPoint(float(x), float(y), 1) for y, x in zip(*np.where(corners))]
    R = harris(gray2, blockSize=3, ksize=3, k=0.06, gassian=True)
    corners = non_maximum_suppression(R, threshold=0.01)
    kp2 = [cv.KeyPoint(float(x), float(y), 1) for y, x in zip(*np.where(corners))]

    des1, kp1 = hog_feature(gray1, kp1)
    des2, kp2 = hog_feature(gray2, kp2)
    # print(des1.shape, des2.shape)

    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    print(len(good_matches))
    return good_matches, kp1, kp2


if __name__ == "__main__":
    img1 = cv.imread('images/1/uttower1.jpg')
    img2 = cv.imread('images/1/uttower2_rotate.jpg')

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    matches, kp1, kp2 = hog_match_rotate(gray1, gray2)

    result = cv.drawMatches(img1, kp1, img2, kp2,
                            matches, None,
                            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('Hog Matches', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # cv.imwrite('results/1/uttower_match_hog_rotate.png', result)