import glob
import os
import shutil

class Subset_dataset:
    '''
        This class creates subset of the original dataset. Some of the labels contain one or 2 images, 
        so to carry out our experiment we are selecting only those labels which have more than some prespecified 
        number of images provided through the variable count(default is 3). 
        
        src -> directory if the original dataset
        dest -> target directory where the subset of the dataset will be stored
        count -> minimum of images per label
    '''
    def __init__(self, src, dest, count=3):
        self.src = src
        self.dest = dest
        self.count = count
    
    def create(self):
        listdir = os.listdir(self.src)
        for dir in listdir:
            file = glob.glob(self.src+ '/' +dir + '/*.jpg')
            if len(file) >= count:
                if not os.path.isdir(self.dest):
                    os.mkdir(self.dest)
                # if not os.path.isdir(os.path.join(self.dest, dir)):
                #     os.mkdir(os.path.join(self.dest, dir))
                shutil.copytree(self.src+ '/' + dir, self.dest+ '/' + dir)


if __name__ == '__main__':
    src = '../lfw'
    dest = '../dataset'
    count = 3
    dataset = Subset_dataset(src, dest, count)
    dataset.create()
    
