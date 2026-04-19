import numpy as np
import cv2 as cv
from sift import sift_match 
from hog import hog_match
from hog_rotate import hog_match_rotate
from scipy.linalg import null_space

def svd_transform(source, target):
    mu_p = np.mean(source, axis=0)
    mu_q = np.mean(target, axis=0)
    H = (source - mu_p).T @ (target - mu_q)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_q - R @ mu_p
    return R, t

def RANSAC(source, target, num_iterations=500):
    """
    source: [N, 2]
    target: [N, 2]

    return:
        R: [2, 2]
        t: [2,]
    """
    best_inliers = []
    best_R = None
    best_t = None

    # distances = np.linalg.norm(source[:] - target[:], axis=1)
    # threshold = np.mean(distances) * 2
    threshold = 2.8

    for i in range(num_iterations):
        indices = np.random.choice(source.shape[0], 3, replace=False)
        R, t = svd_transform(source[indices], target[indices])
        
        transformed = (R @ source.T).T + t
        distances = np.linalg.norm(transformed - target, axis=1)
        inliers = np.where(distances < threshold)[0]

        if i%100 == 0:
            print(f"Iteration {i}, Inliers: {len(best_inliers)}, distances: {distances.mean():.2f}")

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R = R
            best_t = t

    best_R, best_t = svd_transform(source[best_inliers], target[best_inliers])
    return best_R, best_t

def homography_transform(source, target):
    A = []
    for (x, y), (u, v) in zip(source, target):
        A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
    A = np.array(A)
    # H_flat = null_space(A)[:, 0]
    # H = H_flat.reshape(3, 3)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H

def homography_RANSAC(source, target, num_iterations=500):
    """
    source: [N, 2]
    target: [N, 2]
    """
    if source.shape[0] < 4:
        H = homography_transform(source, target)
        return H
    best_inliers = []
    best_H = None

    # distances = np.linalg.norm(source[:] - target[:], axis=1)
    # threshold = np.mean(distances) * 2
    threshold = 2.4

    X = np.hstack((source, np.ones((source.shape[0], 1)))).T  # [3, N]

    for i in range(num_iterations):
        indices = np.random.choice(source.shape[0], 4, replace=False)
        H = homography_transform(source[indices], target[indices])
        
        transformed = H @ X
        transformed = (transformed[:2, :] / transformed[2, :]).T  # [N, 2]
        distances = np.linalg.norm(transformed - target, axis=1)
        inliers = np.where(distances < threshold)[0]

        if i%100 == 0:
            print(f"Iteration {i}, Inliers: {len(best_inliers)}, distances: {distances.mean():.2f}")

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    best_H = homography_transform(source[best_inliers], target[best_inliers])
    return best_H

def affine(img1, img2, description_type='sift'):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    if description_type == 'sift':
        matches, kp1, kp2 = sift_match(gray1, gray2)
    elif description_type == 'hog':
        matches, kp1, kp2 = hog_match(gray1, gray2)
    elif description_type == 'hog_rotate':
        matches, kp1, kp2 = hog_match_rotate(gray1, gray2)
    else:
        raise ValueError("Invalid description type. Choose from 'sift', 'hog', or 'hog_rotate'.")
    
    source = np.array([kp2[m.trainIdx].pt for m in matches])
    target = np.array([kp1[m.queryIdx].pt for m in matches])
    R, t = RANSAC(source, target)
    M = np.hstack((R, t.reshape(-1, 1))) # [2, 3]
    # print(M)
    result = melt(img1, img2, M, is_homography=False)

    # result = cv.warpAffine(img2, M, (img1.shape[1]*2, img1.shape[0]))
    # result[0:img1.shape[0], 0:img1.shape[1]] = img1
    # cv.imshow("Melted Image", result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cv.imwrite("results/test/uttower_stitching_sift_selfaffine.png", result)
    return result

def cal_range(H, h, w, is_homography=True):
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    corners_homogeneous = np.hstack((corners, np.ones((4, 1)))).T  # [3, 4]
    transformed_corners = H @ corners_homogeneous
    if is_homography:
        transformed_corners /= transformed_corners[2, :]

    w_min, h_min = transformed_corners[:2, :].min(axis=1)
    w_max, h_max = transformed_corners[:2, :].max(axis=1)
    return int(w_min), int(w_max), int(h_min), int(h_max)

def melt(img1, img2, H, is_homography=True):
    h, w = img2.shape[:2]
    w_min, w_max, h_min, h_max = cal_range(H, h, w, is_homography)
    # print(f"Transformed range: w: [{w_min}, {w_max}], h: [{h_min}, {h_max}]")
    # print(img1.shape, img2.shape)

    # 计算输出图像的尺寸，确保能够容纳变换后的img2和原始img1
    warp_w = max(w_max, img1.shape[1])
    warp_h = max(h_max, img1.shape[0])
    if is_homography:
        warp2 = cv.warpPerspective(img2, H, (warp_w, warp_h))
    else:
        warp2 = cv.warpAffine(img2, H, (warp_w, warp_h))
    warp1 = np.zeros_like(warp2)
    warp1[0:img1.shape[0], 0:img1.shape[1]] = img1

    mask1 = np.all(warp1 > 0, axis=-1)
    mask2 = np.all(warp2 > 0, axis=-1)
    overlap = mask1 & mask2
    # cv.imshow("overlap", overlap.astype(np.uint8)*255)
    # cv.imshow("mask1", mask1.astype(np.uint8)*255)

    alpha3 = np.zeros_like(overlap, dtype=np.float32)
    for i in range(overlap.shape[0]):
        row = overlap[i]  # [W]
        cols = np.where(row == 1)[0]
        
        if len(cols) == 0:
            continue
        
        left, right = cols[0], cols[-1] 
        n = right - left + 1
        alpha3[i, left:right+1] = np.linspace(1, 0, n)
    alpha3 = alpha3[:, :, np.newaxis]  # [H, W, 1]

    result = np.zeros_like(warp1, dtype=np.float32)
    result[mask1] = warp1[mask1].astype(np.float32)
    result[mask2] = warp2[mask2].astype(np.float32)
    result[overlap] = (alpha3[overlap] * warp1[overlap].astype(np.float32) +
                    (1 - alpha3[overlap]) * warp2[overlap].astype(np.float32))

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def homography(img1, img2, description_type='sift', is_print=False):
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 匹配特征点并计算单应矩阵
    if description_type == 'sift':
        matches, kp1, kp2 = sift_match(gray1, gray2)
    elif description_type == 'hog':
        matches, kp1, kp2 = hog_match(gray1, gray2)
    elif description_type == 'hog_rotate':
        matches, kp1, kp2 = hog_match_rotate(gray1, gray2)
    else:
        raise ValueError("Invalid description type. Choose from 'sift', 'hog', or 'hog_rotate'.")
    
    source = np.array([kp2[m.trainIdx].pt for m in matches])
    target = np.array([kp1[m.queryIdx].pt for m in matches])
    # H, _ = cv.findHomography(source, target, cv.RANSAC)
    H = homography_RANSAC(source, target)

    result = melt(img1, img2, H, is_homography=True)

    if is_print:
        print_result = cv.warpPerspective(img2, H, (img1.shape[1]*2, img1.shape[0]))
        print_result[0:img1.shape[0], 0:img1.shape[1]] = img1
        cv.imshow("Homography Stitched Image_debug", print_result)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # cv.imwrite("results/test/uttower_stitching_sift_selfhom.png", print_result)
    # cv.imshow("Warped Image", result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cv.imwrite("results/1/uttower_stitching_hog_selfhom.png", result)
    return result

if __name__ == "__main__":
    img1 = cv.imread('images/1/yosemite1.jpg')
    img2 = cv.imread('images/1/yosemite2.jpg')
    img3 = cv.imread('images/1/yosemite3.jpg')
    img4 = cv.imread('images/1/yosemite4.jpg')

    # img1 = cv.imread('images/1/uttower1.jpg')
    # img2 = cv.imread('images/1/uttower2.jpg')

    # affine
    # result = affine(img1, img2, description_type='sift')
    # result = affine(result, img3, description_type='sift')
    # result = affine(result, img4, description_type='sift')
    # cv.imshow("Affine Stitched Image", result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cv.imwrite("results/1/yosemite_stitching_affine.png", result)

    # homography
    result = homography(img1, img2, description_type='sift')
    result = homography(result, img3, description_type='sift')
    result = homography(result, img4, description_type='sift')
    cv.imshow("Homography Stitched Image", result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # cv.imwrite("results/1/yosemite_stitching_testhog.png", result)