import numpy as np
import math
import csv
# x=math.sqrt
import time
time_start = time.perf_counter()

rsize=2000
csize=2000

matrix = np.zeros((rsize,csize))

param= 0.1

with open('Matrix_SMILEMETAL_NOPLD_2000.csv') as csv_file:
    data = list(csv.reader(csv_file))
    line_count = 0
    for row in range(rsize):
        for col in range(csize):
           if float(data[row][col]) >= float(param) and row!=col:
               matrix[row][col] = float(data[row][col])  
           else:
               matrix[row][col] =0

with open('Wieghted_2000.csv', 'w', newline='', encoding='utf-8') as f:
    employee_writer = csv.writer(f)
    employee_writer.writerows(matrix)



