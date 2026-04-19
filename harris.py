import cv2 as cv
import numpy as np
from scipy.ndimage import maximum_filter

def harris(gray, blockSize=2, ksize=3, k=0.04, gassian=False):
    Ix = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=ksize)
    Iy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=ksize)
    # Ix = cv.Scharr(gray, cv.CV_64F, 1, 0)
    # Iy = cv.Scharr(gray, cv.CV_64F, 0, 1)

    Ix2 = Ix ** 2 # [H, W]
    Iy2 = Iy ** 2
    Ixy = Ix * Iy

    if gassian:
        Ix2 = cv.GaussianBlur(Ix2, (blockSize, blockSize), 0)
        Iy2 = cv.GaussianBlur(Iy2, (blockSize, blockSize), 0)
        Ixy = cv.GaussianBlur(Ixy, (blockSize, blockSize), 0)
    
    else:
        kernel = np.ones((blockSize, blockSize), dtype=np.float32) / (blockSize * blockSize)
        Ix2 = cv.filter2D(Ix2, -1, kernel)
        Iy2 = cv.filter2D(Iy2, -1, kernel)
        Ixy = cv.filter2D(Ixy, -1, kernel)

    R = (Ix2 * Iy2 - Ixy ** 2) - k * (Ix2 + Iy2) ** 2

    return R

def non_maximum_suppression(R, threshold):
    # Thresholding
    threshold = threshold * R.max()
    R_max = maximum_filter(R, size=3)
    corners = (R == R_max) & (R > threshold)
    # print(corners)
    return corners

if __name__ == "__main__":
    img = cv.imread("images/1/uttower2.jpg")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 预处理
    # gray = cv.GaussianBlur(gray, (3,3), 0)
    # kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    # gray = cv.filter2D(gray, -1, kernel)
    # blur = cv.GaussianBlur(gray, (3,3), 0)
    # gray = cv.addWeighted(gray, 1.5, blur, -0.5, 0)
    # cv.imshow("Preprocess", gray)

    gray = np.float32(gray)
    # R = cv.cornerHarris(gray, blockSize=3, ksize=3, k=0.06)
    R = harris(gray, blockSize=3, ksize=3, k=0.06, gassian=True)

    corners = non_maximum_suppression(R, threshold=0.01)
    # print(corners.shape) # (410, 615)
    print(len(np.where(corners)[0]))
    for y, x in zip(*np.where(corners)):
        cv.circle(img, (x, y), 1, (0, 0, 255), -1)

    cv.imshow("Harris Corners", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # cv.imwrite("results/1/uttower2.png", img)