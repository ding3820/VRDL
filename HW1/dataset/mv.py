import os
import shutil
import random

path = "train"

num = []

for root, dirs, files in os.walk(path):
    
    for d in dirs:
        pa = os.path.join(root,d)
        num = len([name for name in os.listdir(pa)])
        img_list = random.sample(list(range(num)), 10)
        

        for i in img_list:
                
            image_name = "image_" + str(format(i, '04d')) + '.jpg'
            print(os.path.join(pa,image_name))
            print("val/"+d)
                
            shutil.move(os.path.join(pa,image_name), "val/"+d)