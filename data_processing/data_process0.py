# Make little dataset for verifying no bug
# sample less user
import os
import random
import cv2
from PIL import Image
import numpy as np

OULU_DATABASE_PATH = '/mnt/hdd.user/datasets/FAS/Oulu-NPU'
SMALL_OULU_DATABASE_PATH = '/workspace/Face-Anti-Spoofing-Neural-Network/SMALLOULU'
SUBDIRS = {"train": "Train_files", "test": "Test_files", "dev": "Dev_files"}
USERS = {"train": [i for i in range(1, 21)], "dev": [i for i in range(21, 36)], "test": [i for i in range(36, 56)]}

folder = {}
images = {}
label = {}
indice = 0

def resize_center_eyes(frame,lx,ly,rx,ry):
    img = np.array(frame).transpose(1,0,2)
    res = np.zeros((256,256,3),dtype=float)
    center = (0.5*(lx+rx),0.5*(ly+ry))
    for x in range(256):
        realx = int((x-128)*3 + center[0])
        for y in range(256):
            realy = int((y-100)*3 + center[1])
            res[x,y,:]=np.mean(np.mean(img[realx-1:realx+2,realy-1:realy+2,::-1],axis=0),axis=0)
    res=res.transpose(1,0,2)
    img_png=res
    return res/255,Image.fromarray(img_png.astype('uint8'))

def getAttribute(filename):
    if len(filename) < 12:
        return (None, None, None, None)
    phone = filename[0]
    session = filename[2]
    user = filename[4:6]
    access_type = filename[7]
    return (phone, session, user, access_type)

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f

def getFrames(dirpath, savepath, framenum):
    global indice
    global images
    global folder
    global label

    for onefile in findAllFile(dirpath):
        phone, session, user, access_type = getAttribute(onefile)
        file_type = onefile[-3:]
        fileId = onefile[:-4]

        if (file_type != 'avi'):
            continue
        
        print(onefile)
        # print("access_type: {}, type(access_type): {}".format(access_type, type(access_type)))

        if int(user) < 10:
            user = '0' + user
        cap = cv2.VideoCapture(os.path.join(dirpath, fileId+'.avi'))

        lines = []
        with open(os.path.join(dirpath, fileId+'.txt'), 'r') as f:
            wholelines = f.readlines()
            assert framenum <= len(wholelines)
            sample_idxs = random.sample(range(0, len(wholelines)), framenum)
            sample_idxs.sort()
            print(sample_idxs)
            for i in sample_idxs:
                lines.append(wholelines[i])
        
        currentFrame = 0
        while(currentFrame < framenum):
            ret, frame = cap.read()

            list = lines[currentFrame].split(',')
            lx = int(list[1])
            ly = int(list[2])
            rx = int(list[3])
            ry = int(list[4])

            images[str(indice)], pil_img = resize_center_eyes(frame, lx, ly, rx, ry)

            # print(images[str(indice)])
            # print(pil_img)
            
            # name = '/Volumes/G-DRIVE mobile USB/Train_files_OULU/Train_files/'+str(phone)+'_'+str(session)+'_'+user+'_'+str(file)+'_'+str(currentFrame)+'.png'
            name = os.path.join(savepath, fileId+'_'+str(currentFrame)+'.png')
            pil_img.save(name)
            folder[str(indice)] = name

            print(name)

            if access_type == "1":
                # print("indice:{}, name:{}, label 1".format(indice, name))
                label[str(indice)] = 1
            else:
                label[str(indice)] = 0

            indice += 1

            currentFrame += 1

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    devpath_small = os.path.join(SMALL_OULU_DATABASE_PATH, SUBDIRS["dev"])
    testpath_small = os.path.join(SMALL_OULU_DATABASE_PATH, SUBDIRS["test"])
    trainpath_small = os.path.join(SMALL_OULU_DATABASE_PATH, SUBDIRS["train"])
    print("======== Processing train files =======")
    getFrames(devpath_small, trainpath_small, 5)
    print("======== Processing dev files =======")
    getFrames(devpath_small, devpath_small, 5)
    print("======== Processing test files =======")
    getFrames(devpath_small, testpath_small, 5)

    np.savez("images_small2.npz",**images)
    np.savez("label_small2.npz",**label)
    np.savez("folder_small2.npz",**folder)