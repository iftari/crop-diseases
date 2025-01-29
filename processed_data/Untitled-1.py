# %%
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io

%matplotlib inline


# %%
path = r'data\Brown_spot\DSC_0100.jpg'
image = io.imread(path)
type(image), image.shape
plt.imshow(image)

# %%

type(image),image.shape



# %%
plt.imshow(image)

# %%
image.shape[0]*image.shape[1]

# %%
path=r'data\Brown_spot\DSC_0100.jpg'
image=io.imread(path,as_gray=True)
image

# %%
plt.imshow(image)

# %%
image.shape[0]*image.shape[1]

# %%
from skimage.transform import rescale, resize
scimg = rescale(image, 1/2)
image.shape, scimg.shape

# %%
plt.imshow(scimg)

# %%
plt.imshow(image)

# %%
resizedimg = resize(scimg, output_shape=(228,228))
resizedimg.shape

# %%
plt.imshow(resizedimg)

# %%
import os
os.getcwd()
# data\Bacterial_leaf_blight\DSC_0365.JPG
os.listdir('data')
os.listdir('data\Bacterial_leaf_blight')



# %%
# Trnasformatiom
class Transform:
    def __init__(self, dirs: list[str]) -> None:
        self.from_path = 'data'
        self.to_path = 'processed_data'
        self.dirs = dirs
        self.cls = {dirs[i]: i for i in range(len(dirs))}
        self.images_path = [
            (f'data\\{dir}\\{img}', f'processed_data\\{img.split(".")[0]}_{self.cls[dir]}.{img.split(".")[1]}') for dir in self.dirs for img in os.listdir(f'data\\{dir}')]
        print(len(self.images_path))
    
    def transform(self):
        for img, pimg in self.images_path:
            image = io.imread(img, as_gray=True)
            scimg = rescale(image, 1/2)
            resized_img = resize(scimg, (228,228))
            io.imsave(pimg, resized_img)
        return True

# %%
#for dir in ['Bacterial_leaf_blight','Brown_spot','Leaf_smut']:
    #print(f'data\\{dir}')

# %%
#for dir in ['Bacterial_leaf_blight','Brown_spot','Leaf_smut']:
    #for img in os.listdir(f'data\\{dir}'):
        #print(f'data\\{dir}\\{img})

# %%
for dir in ['Bacterial_leaf_blight','Brown_spot', 'Leaf_smut']:
    for img in os.listdir(f'data\\{dir}'):
        img_path = img.split('.')
        img_path = img_path[0]+f'_0.'+img_path[1]
        print((f'data\\{dir}\\{img}', f'processed_data\\{img_path}'))


# %%
tf_image = Transform(['Bacterial_leaf_blight','Brown_spot', 'Leaf_smut'])


# %%
tf_image.transform()

# %%
! pip install -U scikit-image
! pip install --upgrade pip


