import skimage

# Utility to get size of image
def GetImageSize(image, path=True):
    # Get image dimensions before processing further
    if path:
        image = skimage.io.imread(image)
    imageHeight = len(image)
    imageWidth = len(image[0])

    return imageHeight, imageWidth