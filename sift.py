import cv2 as cv
import numpy as np

def sift_match(gray1, gray2):
    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(len(kp1), len(kp2))
    print(des1.shape, des2.shape)
    # img1= cv.drawKeypoints (gray1,kp1,gray1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imshow( 'sift_keypoints.jpg' , img1)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    print(len(good_matches))
    # print(good_matches[0].distance, good_matches[0].queryIdx, good_matches[0].trainIdx)

    return good_matches, kp1, kp2

if __name__ == "__main__":
    img1 = cv.imread('images/1/uttower1.jpg')
    img2 = cv.imread('images/1/uttower2.jpg')

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    matches, kp1, kp2 = sift_match(gray1, gray2)

    result = cv.drawMatches(img1, kp1, img2, kp2,
                            matches[:100], None,
                            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('SIFT Matches', result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # cv.imwrite('results/test/uttower_match_sift_rotate.png', result)