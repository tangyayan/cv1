from affine import affine, homography
import cv2 as cv

def main(imgs, description_type='sift', conversion_type='homography'):
    """
    Args:
        imgs: List of input images to be stitched.
        description_type: Type of feature descriptor to use ('sift', 'hog', or 'hog_rotate').
        conversion_type: Type of transformation to use for stitching ('affine' or 'homography').
    """
    result = imgs[0]
    for i in range(1, len(imgs)):
        if conversion_type == 'affine':
            result = affine(result, imgs[i], description_type=description_type)
        elif conversion_type == 'homography':
            result = homography(result, imgs[i], description_type=description_type)
        else:
            raise ValueError("Invalid conversion type. Choose from 'affine' or 'homography'.")
    
    cv.imshow("Final Stitched Image", result)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # cv.imwrite("results/1/yosemite_stitching_homography.png", result)

if __name__ == "__main__":
    img1 = cv.imread('images/1/yosemite1.jpg')
    img2 = cv.imread('images/1/yosemite2.jpg')
    img3 = cv.imread('images/1/yosemite3.jpg')
    img4 = cv.imread('images/1/yosemite4.jpg')
    imgs = [img1, img2, img3, img4]

    # img1 = cv.imread('images/1/uttower1.jpg')
    # img2 = cv.imread('images/1/uttower2.jpg')
    # imgs = [img1, img2]

    main(imgs, description_type='sift', conversion_type='homography')