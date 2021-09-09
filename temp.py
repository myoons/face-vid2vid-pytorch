from glob import glob


if __name__ == '__main__':
    folders = glob('/home/nas2_userF/dataset/Voxceleb2/**')
    print(folders)