'''
Color control and choice 
========================

Colormap Selection 
------------------
A number of colormap obtained from SciVis (`https://sciviscolor.org/`). 


Utilities 
---------
**convert_colormap**

Detailed Description:
---------------------

Convert colormap in `xml` format into `matplotlib` format

'''
import os
import sys
from xml.dom import minidom
import matplotlib as col
import numpy as np
from lxml import etree

def make_cmap(xml):
    '''
    Convert colormap from `PARAVISION` and `SCIVISION` 
    https://sciviscolor.org/outlier-focused-colormaps/
    
    Parameters
    ==========
    file: str
        Input file colormap in xml format 
    
    Returns
    =======
    colormap:    
        Colormap object
    '''

    vals = load_xml(xml)
    colors = vals['color_vals']
    position = vals['data_vals']


    if len(position) != len(colors):
        sys.exit('position length must be the same as colors')

    cdict = {'red':[], 'green':[], 'blue':[]}

    if position[0] != 0:
        cdict['red'].append((0, colors[0][0], colors[0][0]))
        cdict['green'].append((0, colors[0][1], colors[0][1]))
        cdict['blue'].append((0, colors[0][2], colors[0][2]))
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    if position[-1] != 1:
        cdict['red'].append((1, colors[-1][0], colors[-1][0]))
        cdict['green'].append((1, colors[-1][1], colors[-1][1]))
        cdict['blue'].append((1, colors[-1][2], colors[-1][2]))
    cmap = col.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap


def load_xml(xml):
    
    try:
        xmldoc = etree.parse(xml)
    except IOError as e:
        print ('The input file is invalid. It must be a colormap xml file. Go to https://sciviscolor.org/home/colormaps/ for some good options')
        print ('Go to https://sciviscolor.org/matlab-matplotlib-pv44/ for an example use of this script.')
        sys.exit()
    
    data_vals=[]
    color_vals=[]
    for s in xmldoc.getroot().findall('.//Point'):
        data_vals.append(float(s.attrib['x']))
        color_vals.append((float(s.attrib['r']),float(s.attrib['g']),float(s.attrib['b'])))
    return {'color_vals':color_vals, 'data_vals':data_vals}