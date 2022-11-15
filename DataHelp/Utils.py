from skimage import io
import skimage

def Translate(image, xTransform, yTransform):
    tr = skimage.transform.EuclideanTransform(translation=[xTransform, yTransform])
    img = skimage.transform.warp(image, tr)
    return img

def GetImageSize(image, path=True):
    # Get image dimensions before processing further
    if path:
        image = io.imread(image)
    imageHeight = len(image)
    imageWidth = len(image[0])

    return imageHeight, imageWidth