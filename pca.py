# pca implementation

import numpy as np


# Encoder: f(x) = DT * x
def encodePoint(pt, D):
	return np.matmul(D.transpose(), pt)

# Decoder: g(f(x)) = DDT * x
def decodePoint(pt, D):
	return np.matmul(D, pt)

def findError(encodedPoints, D):
	decodedPoints = []
	for point in encodedPoints:
		decodedPoints.append(decodePoint(point, D))
	
	decodedPoints = np.array(decodedPoints)
	
	err = 0.
	for index, pt in enumerate(decodedPoints):
		err += np.linalg.norm(decodedPoints[index] - encodedPoints[index])
	return err
	
def pcaReduction(points, dimension, getErrorRate):
	u,s,vh = np.linalg.svd(points, full_matrices=True, compute_uv=True)

	vectors = np.asarray(sorted(np.column_stack((s,vh)), key = lambda g: -g[0]))

	
	D = vectors[np.ix_([x for x in xrange(dimension)], [x for x in xrange(1,vectors.shape[1])])]
	D = D.transpose()	# Turn D into correctly shaped matrix
	
	encodedPoints = []
	
	
	for point in points:
		encodedPoints.append(encodePoint(np.asarray(point), D))
	
	encodedPoints = np.array(encodedPoints)
	
	if getErrorRate == True:
		print "Error rate:", findError(encodedPoints, D)
		
	return encodedPoints, D

def main():
	test = np.array([[1, 5, 2], [2, 9, 3], [3, 13, 4], [4, 17, 5], [5, 21, 6], [6, 25, 7]])
	encodedPoints, encoder = pcaReduction(test, 1, True)
	
	print "Encoded Points:", encodedPoints
	print "Encoder Matrix:", encoder
	

if __name__ == '__main__':
	main()