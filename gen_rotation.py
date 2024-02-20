import numpy as np
import pickle
import numpy.linalg as lg 
import math

p = pickle.load(open('../mesh_files/partial_icosphere_1.pkl','rb'))
V = p['V']
F = p['F']

F1 = F[79].flatten()
res = []
for i in range(80):
	F2 = F[i].flatten()
	V1 = V[F1].T
	V2 = V[F2].T

	M = V2.dot(lg.inv(V1)).round(8).flatten()
	res.append(M)


res = np.asarray(res)
print(res.shape)
np.save('RM.npy',res)


