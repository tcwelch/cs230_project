from PIL import Image
import numpy as np
import os
from pathlib import Path


def resize(img,dim_x,dim_y):
    return img.resize((dim_x,dim_y),Image.ANTIALIAS)

def flatten_pixels(img_list):
    return [color_val for pixel in img_list for color_val in pixel]

def get_batch(txt_path,dim_x,dim_y):
	"""
	inputs:
		txt_path = path from where this file lives ending in txt file name
		dim_x = resized image width
		dim_y = resized image height

	returns:
		X = size (m,n_H,n_W,n_C)
		y = size (m) - encoded as 1 - 6
	"""

	data_mix = open(txt_path,'r')
	examples = data_mix.readlines()

	entry_dir = Path('archive/Garbage classification/Garbage classification')
	cardboard_dir_path = entry_dir / 'cardboard' # y label = 1
	glass_dir_path = entry_dir / 'glass' # y label = 2
	metal_dir_path = entry_dir / 'metal' # y label = 3
	paper_dir_path = entry_dir / 'paper' # y label = 4
	plastic_dir_path = entry_dir / 'plastic' # y label = 5
	trash_dir_path = entry_dir / 'trash' # y label = 6
	batch_x = []
	batch_y = []
	for line in examples:
		string = line.strip()
		words = string.split(' ')
		image = words[0]
		indx = words[1]
		if int(indx) == 1:
			with Image.open(glass_dir_path / image) as img:
				resized_img = resize(img,dim_x,dim_y)
				batch_x.append(np.asarray(resized_img))
				batch_y.append(np.array([1,0,0,0,0,0]))
		elif int(indx) == 2:
		    with Image.open(paper_dir_path / image) as img:
		    	resized_img = resize(img,dim_x,dim_y)
		    	batch_x.append(np.asarray(resized_img))
		    	batch_y.append(np.array([0,1,0,0,0,0]))
		elif int(indx) == 3:
			with Image.open(cardboard_dir_path / image) as img:
				resized_img = resize(img,dim_x,dim_y)
				batch_x.append(np.asarray(resized_img))
				batch_y.append(np.array([0,0,1,0,0,0]))
		elif int(indx) == 4:
			with Image.open(plastic_dir_path / image) as img:
				resized_img = resize(img,dim_x,dim_y)
				batch_x.append(np.asarray(resized_img))
				batch_y.append(np.array([0,0,0,1,0,0]))
		elif int(indx) == 5:
			with Image.open(metal_dir_path / image) as img:
				resized_img = resize(img,dim_x,dim_y)
				batch_x.append(np.asarray(resized_img))
				batch_y.append(np.array([0,0,0,0,1,0]))
		elif int(indx) == 6:
			with Image.open(trash_dir_path / image) as img:
				resized_img = resize(img,dim_x,dim_y)
				batch_x.append(np.asarray(resized_img))
				batch_y.append(np.array([0,0,0,0,0,1]))
	return np.array(batch_x), np.array(batch_y)


def main():
	X_test, y_test = get_batch('archive/one-indexed-files-notrash_test.txt',224,224)
	print("X_test.shape")
	print(X_test.shape)
	print("y_test.shape")
	print(y_test.shape)
	print('y_test')
	print(y_test)


if __name__ == "__main__":
    main()