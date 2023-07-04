# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:52:06 2021

@author: Mehrdad
"""


from sklearn import preprocessing
import pandas as pd
import numpy
import time
time_start = time.perf_counter()

MOF = pd.read_csv("MOF_METAL_1000.csv")
d = preprocessing.normalize(MOF)
numpy.savetxt("MOF_METAL_1000_Normalised.csv", d, delimiter=",")



time_elapsed = (time.perf_counter() - time_start)

print ("%5.1f secs" % (time_elapsed))
