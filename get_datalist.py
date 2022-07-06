import os,shutil
import cv2
import glob
import random


def Write_TXT(txt_file, data_list):
	''' 
	write train and test txt file list
	'''
	random.shuffle(data_list)
	fd  = open(txt_file, 'w')
	for i in range(len(data_list)):
		fd.write(data_list[i] + '\n')
	fd.close()

def Get_List(file_path):
	'''
	'''
	datalist = []
	dirlist = glob.glob(file_path + '/*')
	for dir in dirlist:
		imglist = glob.glob(dir + '/*.*')
		for img in imglist:
			label = img.split('/')[2]
			data = img + ' ' + label
			cv_img = cv2.imread(img.replace(' ',''))
			if cv_img is None:
				continue
			datalist.append(data)
	return datalist


if __name__=='__main__':
	test_path  = 'Datasets/test'
	test_list = Get_List(test_path)
	Write_TXT('maskface_test_list.txt', test_list)
	
	train_path = 'Datasets/train'
	train_list = Get_List(train_path)	
	Write_TXT('maskface_train_list.txt', train_list)

