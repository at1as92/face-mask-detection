import sys, os
import numpy as np
from math import *
from PIL import Image

import tarfile 
  
# # open file 
# file = tarfile.open('originalPics.tar') 
  
# # extracting file 
# try:
# 	file.extractall('./images', numeric_owner=True)
# except PermissionError as e:
# 	print(e)

# file.close()


def filterCoordinate(c,m):
	if c < 0:
		return 0
	elif c > m:
		return m
	else:
		return c


ellipse_filename = 'FDDB-folds/FDDB-fold-01-ellipseList.txt'
rect_filename = 'FDDB-folds/FDDB-fold-01-rectList.txt'

with open(ellipse_filename) as f:
	lines = [line.rstrip('\n') for line in f]

f = open(rect_filename,'w')
i = 0

#C:\SDRE-CV\Mini_Project\FDDB\images2\2002\08\11\big
while i < len(lines):
	img_file = 'images/' + lines[i] + '.jpg'
	img = Image.open(img_file)
	w = img.size[0]
	h = img.size[1]
	num_faces = int(lines[i+1])
	for j in range(num_faces):
		ellipse = lines[i+2+j].split()[0:5]
		a = float(ellipse[0])
		b = float(ellipse[1])
		angle = float(ellipse[2])
		centre_x = float(ellipse[3])
		centre_y = float(ellipse[4])
		
		tan_t = -(b/a)*tan(angle)
		t = atan(tan_t)
		x1 = centre_x + (a*cos(t)*cos(angle) - b*sin(t)*sin(angle))
		x2 = centre_x + (a*cos(t+pi)*cos(angle) - b*sin(t+pi)*sin(angle))
		# x_max = filterCoordinate(max(x1,x2),w)
		# x_min = filterCoordinate(min(x1,x2),w)
		x_max = int(filterCoordinate(max(x1,x2),w))
		x_min = int(filterCoordinate(min(x1,x2),w))
		
		if tan(angle) != 0:
			tan_t = (b/a)*(1/tan(angle))
		else:
			tan_t = (b/a)*(1/(tan(angle)+0.0001))
		t = atan(tan_t)
		y1 = centre_y + (b*sin(t)*cos(angle) + a*cos(t)*sin(angle))
		y2 = centre_y + (b*sin(t+pi)*cos(angle) + a*cos(t+pi)*sin(angle))
		# y_max = filterCoordinate(max(y1,y2),h)
		# y_min = filterCoordinate(min(y1,y2),h)
		y_max = int(filterCoordinate(max(y1,y2),h))
		y_min = int(filterCoordinate(min(y1,y2),h))
	
		text = img_file + ',' + str(x_min) + ',' + str(y_min) + ',' + str(x_max) + ',' + str(y_max)	+ '\n'
		print(text)
		f.write(text)

	i = i + num_faces + 2


f.close()
