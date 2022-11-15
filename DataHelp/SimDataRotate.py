import numpy as np
import skimage
import os
    
def RotateImagesInDir(origImageDir, newImageDir):
    angleIncrement = 1 # How many degrees between each rotation

    if os.path.exists(newImageDir):
        if len(os.listdir(newImageDir)) != 0:
            raise RuntimeError("Path already exists - will not overwrite")
    else:
        os.mkdir(newImageDir)

    classIncludesZone = True

    numImages = len(os.listdir(origImageDir))
    cnt = 1

    for image in os.listdir(origImageDir):
        try:
            img = skimage.io.imread(f"{origImageDir}/{image}", as_gray=True)
            print(f"Augmenting {image} ({cnt}/{numImages})")
            cnt += 1
        except ValueError:
            print(f"Failed to open {image}")
            cnt += 1

        if classIncludesZone:
            saveDir = f"{newImageDir}/{os.path.splitext(image)[0]}"
            os.mkdir(saveDir)
        else:
            saveDir = newImageDir

        for angle in range(0, 180, angleIncrement):
            newImg = skimage.util.img_as_ubyte(skimage.transform.rotate(img, angle))
            skimage.io.imsave(f"{saveDir}/{os.path.splitext(image)[0]}_{angle}D.png", newImg, check_contrast=False)

def main():
    RotateImagesInDir("In", "Out")

if __name__ == "__main__":
    main()
