# Make little dataset for verifying no bug
# sample less user
import os
import random

ORIGIN_OULU_DATABASE_PATH = '/mnt/hdd.user/datasets/FAS/Oulu-NPU'
SMALL_OULU_DATABASE_PATH = '/workspace/Face-Anti-Spoofing-Neural-Network/SMALLOULU'
SUBDIRS = {"train": "Train_files", "test": "Test_files", "dev": "Dev_files"}
USERS = {"train": [i for i in range(1, 21)], "dev": [i for i in range(21, 36)], "test": [i for i in range(36, 56)]}
# FRAME_PATH = '/workspace/Face-Anti-Spoofing-Neural-Network/SMALLOULU_FRAME'

# train_users = [i for i in range(1, 21)]
# dev_users = [i for i in range(21, 36)]
# test_users = [i for i in range(36, 56)]

# 新建SMALL OULU DATABASE的路径
if not os.path.exists(SMALL_OULU_DATABASE_PATH):
    os.mkdir(SMALL_OULU_DATABASE_PATH)


def list2file(l, savepath):
    with open(savepath, 'w') as f:
        for i in l:
            f.write(i + "\n")

def file2list(loadpath):
    l = []
    with open(loadpath, 'r') as f:
        for i in f.readlines():
            i = i.strip()
            if len(i) > 0:
                l.append(i)
    return l

def generateSmallDataset(originpath, num, dirtype):
    file_list = []

    user_list = random.sample(USERS[dirtype], num)
    for phone in range(1, 7):
        for session in range(1, 4):
            for user in user_list:
                for access_type in range(1, 6):
                    nom = str(user)
                    if user < 10:
                        nom = '0' + nom
                    fileId = str(phone)+'_'+str(session)+'_'+nom+'_'+str(access_type)
                    # print(fileId)
                    assert os.path.exists(os.path.join(os.path.join(ORIGIN_OULU_DATABASE_PATH, SUBDIRS[dirtype]), fileId+".avi"))
                    file_list.append(fileId)
    
    savepath = os.path.join(os.path.join(SMALL_OULU_DATABASE_PATH, SUBDIRS[dirtype]), dirtype+".txt")
    list2file(file_list, savepath)
    return file_list

if __name__ == '__main__':
    ptype = "train"
    file_list = generateSmallDataset(os.path.join(ORIGIN_OULU_DATABASE_PATH, SUBDIRS[ptype]), 2, ptype)
    print(file_list)

    # file_list2 = file2list(os.path.join(os.path.join(SMALL_OULU_DATABASE_PATH, SUBDIRS["train"]), "train"+".txt"))
    # print(file_list2)

    # devpath = os.path.join(ORIGIN_OULU_DATABASE_PATH, SUBDIRS["dev"])
    # getFrames(devpath, 5, devpath)