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
		y = size (m) 
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
	return batch_x, batch_y

def get_aug(X,y,num_to_add):
	X_aug = []
	y_aug = []
	y_indx = []
	#converting to categorical list (0-5)
	for i in range(len(y)):
		indx = np.argmax(y[i])
		y_indx.append(indx)

	for i in range(len(num_to_add)):
		#creates a set of all X's with label i
		X_to_aug = [X[idx] for idx in range(len(y_indx)) if y_indx[idx] == i]

		#adds randomly augmented samples to X_aug from class with number specified by num_to_add(i)
		for j in range(num_to_add[i]):
			#chooses random sample from X_to_aug and removes it
			aug_choice = np.random.choice(range(len(X_to_aug)))
			example_to_aug = X_to_aug[aug_choice]

			#chooses random augmentaiton to example_to_remove and adds it to X_aug
			rand_aug = np.random.randint(5)
			if rand_aug == 0:
				X_aug.append(np.rot90(example_to_aug, k=1, axes=(0, 1)))
			elif rand_aug == 1:
				X_aug.append(np.rot90(example_to_aug, k=2, axes=(0, 1)))
			elif rand_aug == 2:
				X_aug.append(np.rot90(example_to_aug, k=3, axes=(0, 1)))
			elif rand_aug == 3:
				X_aug.append(np.flip(example_to_aug,axis=0))
			elif rand_aug == 4:
				X_aug.append(np.flip(example_to_aug,axis=1))
			aug_label = np.array([0,0,0,0,0,0])
			aug_label[i] = 1
			y_aug.append(aug_label)
			# X_to_aug.pop(aug_choice)

	return X_aug,y_aug




def main():
	X, y = get_batch('archive/one-indexed-files-notrash_train.txt',224,224)
	print("len(X)")
	print(len(X))
	print("X[0].shape")
	print(X[0].shape)
	print("len(y)")
	print(len(y))
	print("y[0].shape")
	print(y[0].shape)

	aug_X,aug_y = get_aug(X,y,[3,0,0,2,0,0])
	print("aug_X[0].shape")
	print(aug_X[0].shape)
	print("aug_y[0].shape")
	print(aug_y[0].shape)


	X.extend(aug_X)
	y.extend(aug_y)
	X_test = np.array(X)
	y_test = np.array(y)

	print("X_test.shape")
	print(X_test.shape)
	print("y_test.shape")
	print(y_test.shape)


if __name__ == "__main__":
    main()