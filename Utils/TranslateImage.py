import skimage

# Utility to translate image (original + xTransform, original + yTransform)
def TranslateImage(image, xTransform, yTransform):
    tr = skimage.transform.EuclideanTransform(translation=[xTransform, yTransform])
    img = skimage.transform.warp(image, tr)
    return img