#!/usr/bin/env python
#-*- coding:utf-8 -*-
from __future__ import print_function

__author__ = 'seelviz'

import matplotlib
# Force matplotlib to not use any Xwindows backend:
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from plotly.offline import download_plotlyjs
from plotly.graph_objs import *
from plotly import tools
import plotly

import nibabel as nb

import os
#os.chdir('C:/Users/L/Documents/Homework/BME/Neuro Data I/Data/')

import csv,gc  # garbage memory collection :)

import numpy as np
from numpy import genfromtxt

import pickle

from collections import namedtuple

import csv
import re
import time
import seaborn as sns

# nd stuff
from ndreg import *
import ndio.remote.neurodata as neurodata

from collections import OrderedDict

class atlasregiongraph(object):
    """Class for generating the color coded atlas region graphs"""

    def __init__(self, token, atlas_path = None):
        self._token = token
        self._atlas_path = atlas_path

	# make an atlas directory if none exists
        if not os.path.exists(self._token + '/' + 'atlas'):
            os.makedirs(self._token + '/' + 'atlas')

    def align_brain(self):
        """
        This function is for getting the aligned brain nii file.  If the aligned brain
        is already aquired, just skipt to generate_atlas_region_graph().

        """
        refToken = "ara_ccf2"
        refImg = imgDownload(refToken)
        prefix = self._token + '/atlas/'

        refAnnoImg = imgDownload(refToken, channel="annotation")

        randValues = np.random.rand(1000,3)
        randValues = np.concatenate(([[0,0,0]],randValues))
        randCmap = matplotlib.colors.ListedColormap (randValues)

        # Using the token =================
        inToken = self._token
        nd = neurodata()

        inImg = imgDownload(inToken, resolution=5)
	rawCopy = inImg

	(values, bins) = np.histogram(sitk.GetArrayFromImage(inImg), bins=100, range=(0,500))
	plt.plot(bins[:-1], values)

	counts = np.bincount(values)
	maximum = np.argmax(counts)

	lowerThreshold = maximum
	upperThreshold = sitk.GetArrayFromImage(inImg).max()+1

	inImg = sitk.Threshold(inImg,lowerThreshold,upperThreshold,lowerThreshold) - lowerThreshold
	#==========================

        inImg.SetSpacing([0.01872, 0.01872, 0.005])
        inImg_download = inImg

        inImg = imgResample(inImg, spacing=refImg.GetSpacing())

        inImg = imgReorient(inImg, "LAI", "RSA")

        inImg_reorient = inImg

        spacing=[0.25,0.25,0.25]
        refImg_ds = imgResample(refImg, spacing=spacing)

        inImg_ds = imgResample(inImg, spacing=spacing)

        affine = imgAffineComposite(inImg_ds, refImg_ds, iterations=100, useMI=True, verbose=True)

        inImg_affine = imgApplyAffine(inImg, affine, size=refImg.GetSize())

        inImg_ds = imgResample(inImg_affine, spacing=spacing)
        (field, invField) = imgMetamorphosisComposite(inImg_ds, refImg_ds, alphaList=[0.05, 0.02, 0.01], useMI=True, iterations=100, verbose=True)
        inImg_lddmm = imgApplyField(inImg_affine, field, size=refImg.GetSize())

        ##################
        # Reverse orientation
        ########

        invAffine = affineInverse(affine)
        invAffineField = affineToField(invAffine, refImg.GetSize(), refImg.GetSpacing())
        invField = fieldApplyField(invAffineField, invField)
        inAnnoImg = imgApplyField(refAnnoImg, invField,useNearest=True, size=inImg_reorient.GetSize())

        inAnnoImg = imgReorient(inAnnoImg, "RSA", "LAI")

        inAnnoImg = imgResample(inAnnoImg, spacing=inImg_download.GetSpacing(), size=inImg_download.GetSize(), useNearest=True)

        self._atlas_path = prefix + self._token + "_atlas.nii"

        imgWrite(inAnnoImg, self._atlas_path)

    def generate_atlas_region_graph(self, atlas_path = None):
        self._atlas_path = atlas_path if atlas_path != None else self._atlas_path
        atlas = nb.load(self._atlas_path)    # <- atlas .nii image
        atlas_data = atlas.get_data()

        csvfile = self._token + '/' + self._token + 'localeq.csv'  # <- regular csv from the .nii to csv step

        bright_points = genfromtxt(csvfile, delimiter=',')

        locations = bright_points[:, 0:3]

        regions = [atlas_data[l[1], l[0], l[2]] for l in locations]

        outfile = open(self._token + '/' + self._token + 'localeq.region.csv', 'w')
        infile = open(csvfile, 'r')
        for i, line in enumerate(infile):
            line = line.strip().split(',')
            outfile.write(",".join(line) + "," + str(regions[i]) + "\n")    # adding a 5th column to the original csv indicating its region (integer)
        infile.close()
        outfile.close()

        print(len(regions))
        print(regions[0:10])

    def generate_atlas_region_plotly(self):
        font = {'weight' : 'bold',
            'size'   : 18}

        matplotlib.rc('font', **font)

        ### load data
        thedata = np.genfromtxt(self._token + '/' + self._token + 'localeq.region.csv', delimiter=',', dtype='int', usecols = (0,1,2,4), names=['x','y','z','region'])

        region_dict = OrderedDict()
        for l in thedata:
            trace = 'trace' + str(l[3])
            if trace not in region_dict:
                region_dict[trace] = np.array([[l[0], l[1], l[2], l[3]]])
            else:
                tmp = np.array([[l[0], l[1], l[2], l[3]]])
                region_dict[trace] = np.concatenate((region_dict.get(trace, np.zeros((1,4))), tmp), axis=0)

        current_palette = sns.color_palette("husl", 396)
        # print current_palette

        data = []
        for i, key in enumerate(region_dict):
            trace = region_dict[key]
            tmp_col = current_palette[i]
            tmp_col_lit = 'rgb' + str(tmp_col)
            trace_scatter = Scatter3d(
                x = trace[:,0], 
                y = trace[:,1],
                z = trace[:,2],
                mode='markers',
                marker=dict(
                    size=1.2,
                    color=tmp_col_lit, #'purple',                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.15
                )
            )
            
            data.append(trace_scatter)
            

        layout = Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
            paper_bgcolor='rgb(0,0,0)',
            plot_bgcolor='rgb(0,0,0)'
        )

        fig = Figure(data=data, layout=layout)
        plotly.offline.plot(fig, filename= self._token + '/' + self._token + "_region_color.html")

