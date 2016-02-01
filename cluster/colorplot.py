# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:26:20 2016

@author: Oliver
"""

from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,title,show

# the function that I'm going to plot
def z_func(x,y, labels, DATA):
	total = 0
	for la in set(labels):
		dist = 0
		
		for

		total += exp(-(dist))
 
x = arange(-3.0,3.0,0.1)
y = arange(-3.0,3.0,0.1)
X,Y = meshgrid(x, y) # grid of point
Z = z_func(X, Y) # evaluation of the function on the grid

im = imshow(Z,cmap=cm.BuPu) # drawing the function
# adding the Contour lines with labels
cset = contour(Z,linewidths=2,cmap=cm.afmhot)
clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
colorbar(im) # adding the colobar on the right
# latex fashion title
title('$a_i = \sum_{j} \exp(-d(f_i, f_j))$')
show()