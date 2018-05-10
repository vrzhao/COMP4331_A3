import scipy.io as spio
import numpy as np
import math

# import the data and convert it into a usable format
DBSCAN_dict = spio.loadmat('DBSCAN-Points.mat', squeeze_me=True)
dbscan = DBSCAN_dict['Points']

# Function for calculating the Euclidian distance
def distance(a,b):
    return math.sqrt(abs(a[0]-b[0])**2 + abs(a[1]-b[1])**2)

# Function for finding neighboring points
def findNeighbors(DB, P, eps):
   neighbors = []
   for i in range(0,len(DB)):
       # if two points have a distance less than the eps than they are neighbors
       if(distance(DB[P],DB[i]) < eps):
           neighbors.append(i)

   return neighbors

# My DBSCAN implimentation
def DBSCAN_(DB, eps, minPts):
    # initialize the label list to equal the length of the database
    # I use the number 0 to denote that the point currently has not been given a label, that the label is currently undefined
    labels = [0] * len(DB)

    # set the current label to 1
    C = 1

    # iterate through the database and find the neighbors of each point in the databse
    for i in range(0,len(DB)):
        # check to see if the point has already been checked, if so skip it
        if (not (labels[i]== 0)):
            continue
        # find the neighbors of a given point
        neighbors = findNeighbors(DB,i,eps)
        # if the point does not have enough neighbors it is labeled as noise, we use -1 to denote a noise label
        if (len(neighbors) < minPts):
            labels[i] = -1
        else:
            # If it has enough neighbors, it is a core point and we add the point to the current cluster
            labels[i] = C

            # we iterate throught labels[i]'s neighbors to see if they are also a core point or border point
            # by using a while loop, we can change the number of time the function loops by changing the length of neighbors within the loop.
            j = 0
            while (j < len(neighbors)):
                P = neighbors[j]
                # if a point was previously labeled a noise point but is a neighbor to the current point, it is a border point of the current cluster. Relabel the point.
                if (labels[P] == -1):
                    labels[P] == C
                # check to see if the current point has been given a label already
                if (labels[P] == 0):
                    labels[P] = C
                    # find the neighbors of the current point
                    P_neighbors = findNeighbors(DB, P, eps)
                    # if the currently point has a number of neighbors greater than or equal to minPts, it is another core point and we add its neighbors to be checked in later loops.
                    if (len(P_neighbors) >= minPts):
                        neighbors = neighbors + P_neighbors
                j += 1

            # increment the current label number now that have finished forming the previous cluster
            C += 1

    return labels

my_labels = DBSCAN_(dbscan,0.15,3)

# sklearn DBSCAN
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.15, min_samples=3).fit(dbscan)
sk_labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(sk_labels)) - (1 if -1 in sk_labels else 0)

# Plot sklearn DBSCAN
import matplotlib.pyplot as plt

# creates a set of all the unique labels
unique_labels = set(sk_labels)
# creates an array that contains the sizes and colors for each unique label
size = [8]*len(unique_labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col, size in zip(unique_labels, colors, size):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
        size = 5

    # create an array of boolean values so that we only plot points in the same cluster each iteration
    is_label_k = (sk_labels == k)

    xy = dbscan[is_label_k]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=size)

plt.title('sklearn Estimated number of clusters: %d' % n_clusters_)
plt.show()



# Plot My DBSCAN
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(my_labels)) - (1 if -1 in my_labels else 0)

# creates a set of all the unique labels
unique_labels = set(my_labels)
# creates an array that contains the sizes and colors for each unique label
size = [8]*len(unique_labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col, size in zip(unique_labels, colors,size):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
        size = 5

    # create an array of boolean values so that we only plot points in the same cluster each iteration
    is_label_k = np.zeros_like(my_labels, dtype=bool)
    for i in range(0, len(dbscan)):
        is_label_k[i] = (my_labels[i] == k)

    xy = dbscan[is_label_k]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=size)

plt.title('My DBSCAN Estimated number of clusters: %d' % n_clusters_)
plt.show()

# function for testing if the resulting labels are identical
def checkIdentical():
    normalized_sk_lables = [0] * 500
    for i in range(0, 500):
        if sk_labels[i] == -1:
            normalized_sk_lables[i] = -1
        else:
            # as i started my labels at 1, i need to shift all non-noise labels up by a value of 1
            normalized_sk_lables[i] = sk_labels[i] + 1

    print normalized_sk_lables == my_labels

checkIdentical()