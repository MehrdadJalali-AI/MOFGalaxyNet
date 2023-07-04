# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:46:24 2021

@author: Mehrdad
"""
import numpy as np
import networkx as nx
import time
time_start = time.perf_counter()

# -------DIRECTED Graph, Unweighted-----------
# Unweighted directed graph:
a = np.loadtxt('Wieghted_2000.csv', delimiter=',', dtype=float)
D = nx.DiGraph(a)
nx.write_weighted_edgelist(D, 'EdgesList_2000.csv', delimiter=',')  # output


time_elapsed = (time.perf_counter() - time_start)

print ("%5.1f secs" % (time_elapsed))


time_elapsed = (time.perf_counter() - time_start)

print ("%5.1f secs" % (time_elapsed))
