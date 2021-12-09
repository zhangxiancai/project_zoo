import os
import glob
import cv2
import shutil

pathlist = glob.glob('data/*')
for path in pathlist:
	imglist = glob.glob('{}/*.jpg'.format(path))
	for img in imglist:
		basepath = img.split('/')[-1]
		if 'masked' in img:
			out = 'masked/{}'.format(basepath)
		else:
			out = 'unmask/{}'.format(basepath)
		shutil.move(img, out)
