'''
Color control and choice 
========================

Colormap Selection 
------------------
The following colormaps obtained from SciVis (`https://sciviscolor.org/`) are available in `zapata/SciVis_colormaps`.

.. image:: ../resources/colormap.png
        :scale: 100 %
        :align: center

Utilities 
---------
**convert_colormap**    

    *Convert colormap in `xml` format into `matplotlib` format*

'''
import os
import sys
from xml.dom import minidom
import matplotlib as col
import numpy as np
from lxml import etree

def make_cmap(xml):
    '''
    | Convert colormap from `PARAVISION` and `SCIVISION` 
    | https://sciviscolor.org/outlier-focused-colormaps/
    
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
    '''
    Load colormap in `xml` format
    '''
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

def plot_cmap(colormap):
    '''
    Show colormap
    '''

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig=plt.figure(figsize=(8,1))
    map=fig.add_subplot(111)
    map.set_frame_on(False)
    map.get_xaxis().set_visible(False)
    plt.title(colormap.name)
    map.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)
    map.imshow(gradient, aspect='auto', cmap=plt.get_cmap(colormap))
    plt.show(fig)
    return

def viewcmap(DIR):
    '''
    Show colormaps in directory `DIR`

    Colormaps are in `xml` format
    '''
    for i in sorted(os.listdir(DIR)):
        filename, file_extension = os.path.splitext(i)
        if file_extension == '.xml':
            tt=zcol.make_cmap(COLOR +'/'+ i)
            tt.name = filename
            plot_cmap(tt)
    return
def _showcolormap():
    '''
    Create picture of all colormap in directory COLOR
    '''
    fil = []
    for i in sorted(os.listdir(COLOR)):
        filename, file_extension = os.path.splitext(i)
        if file_extension == '.xml':
            fil.append(i)
    nplot=len(fil)
    nplot2 = int(nplot/2)
    fig,ax=plt.subplots(nrows=int(nplot/2),ncols=2,figsize=(12,32))
    for i in range(0,int(nplot/2)):

            tt=zcol.make_cmap(COLOR +'/'+ fil[i])
            filename, file_extension = os.path.splitext(fil[i])
            tt.name = filename
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
    
            axs = ax[i,0]
            axs.set_frame_on(False)
            axs.get_xaxis().set_visible(False)
            axs.set_title(tt.name)
            axs.get_yaxis().set_visible(False)
            axs.imshow(gradient, aspect='auto', cmap=tt)
    for i in range(0,int(nplot/2)):

            tt=zcol.make_cmap(COLOR +'/'+ fil[i+nplot2])
            filename, file_extension = os.path.splitext(fil[i+nplot2])
            tt.name = filename
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
    
            axs = ax[i,1]
            axs.set_frame_on(False)
            axs.get_xaxis().set_visible(False)
            axs.set_title(tt.name)
            axs.get_yaxis().set_visible(False)
            axs.imshow(gradient, aspect='auto', cmap=tt)
    fig.tight_layout(pad=0.5)
    fig.show()
    plt.savefig('colormap.png')
    return
