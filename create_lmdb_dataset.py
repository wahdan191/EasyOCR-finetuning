import fire
import os
import lmdb
import cv2
import numpy as np

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where images are stored
        outputPath : LMDB output path
        gtFile     : list of image paths and labels
        checkValid : if True, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1073741824)  # Adjust map size as needed
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    print(f"Total samples in {gtFile}: {nSamples}")

    for i in range(nSamples):
        try:
            if '.png' in datalist[i]:
                imagePath, label = datalist[i].strip('\n').split('.png')
                imagePath += '.png'
            elif '.jpg' in datalist[i]:
                imagePath, label = datalist[i].strip('\n').split('.jpg')
                imagePath += '.jpg'
            else:
                print(f"Skipping line {i + 1} in {gtFile}: Invalid format - No .png or .jpg found")
                continue
        except ValueError:
            print(f"Skipping line {i + 1} in {gtFile}: Invalid format")
            continue
        
        imagePath = os.path.join(inputPath, imagePath)

        if not os.path.exists(imagePath):
            print(f'{imagePath} does not exist')
            continue
        
        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print(f'{imagePath} is not a valid image')
                continue

        imageKey = f'image-{cnt:09}'
        labelKey = f'label-{cnt:09}'
        cache[imageKey.encode()] = imageBin
        cache[labelKey.encode()] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print(f'Written {cnt} / {nSamples}')
        
        cnt += 1

    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print(f'Created dataset with {nSamples} samples')

if __name__ == '__main__':
    fire.Fire(createDataset)
