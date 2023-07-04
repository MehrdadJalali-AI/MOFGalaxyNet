import numpy as np
import math
import csv
# x=math.sqrt
import time
from scipy.spatial import distance
from rdkit import Chem,DataStructs
from scipy import spatial
from rdkit import Chem


rsize=2000
csize=2000
alpha=0.1

def WB(list1):
    return str(list1).replace("['",'').replace("']",'')

time_start = time.perf_counter()

def similarity(row1,row2):
 
    mol1 = Chem.MolFromSmiles(WB(row1))
    mol2 = Chem.MolFromSmiles(WB(row2))
    try:
      fp1 = Chem.RDKFingerprint(mol1)
      fp2 = Chem.RDKFingerprint(mol2)
      return DataStructs.TanimotoSimilarity(fp1,fp2)
    except (ValueError, TypeError):
      return 0
    
matrix = np.zeros((rsize, csize))

with open('/content/drive/MyDrive/Research/MOF/MOFSMILES/SMILES_METAL_2000_NoPLD.csv') as csv_file:
    data = list(csv.reader(csv_file))
    line_count = 0
    for row in range(rsize):
        for col in range(csize):
         matrix[row][col]= (alpha*(1-spatial.distance.cosine(list(np.float_(data[row][0:6])),list(np.float_(data[col][0:6]))))+(1-alpha)*similarity(data[row][8],data[col][8]))
        #  matrix[row][col]= similarity(data[row][8],data[col][8])

with open('/content/drive/MyDrive/Research/MOF/MOFSMILES/Matrix_SMILEMETAL_NOPLD_2000.csv', 'w', newline='', encoding='utf-8') as f:
    employee_writer = csv.writer(f)
    employee_writer.writerows(matrix)


time_elapsed = (time.perf_counter() - time_start)

print ("Finisheeeeeeeeeeeeeeeeeeeeeeed")
