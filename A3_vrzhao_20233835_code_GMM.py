import scipy.io as spio
from scipy.stats import norm
import pandas as pd
import numpy as np
import math


# read in the data
GMM_dict = spio.loadmat('GMM-Points.mat', squeeze_me=True)
gmm = GMM_dict['Points']

# function for plotting results
import matplotlib.pyplot as plt
def plot(labels, iteration):
    unique_labels = set(labels)
    size = [8] * len(unique_labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col, size in zip(unique_labels, colors, size):
        class_member_mask = np.zeros_like(labels, dtype=bool)
        for i in range(0, len(gmm)):
            class_member_mask[i] = (labels[i] == k)

        xy = gmm[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=size)

    if(iteration == 0):
        plt.title('GMM')
    else:
        plt.title('GMM Iteration: %d' % iteration)
    plt.show()

# Function calculates the probability of each point being in a given cluster based on the mean, sd, and lam
def probability(point, mean, sd, lam):
    p = lam
    for i in range(len(point)):
        p *= norm.pdf(point[i], mean[i], sd[i][i])

    return p

# Function gives each point a label based on the probability it is in either cluster
def expectation(gmm):
    cluster = [0,0]
    labels = [0] * len(gmm)

    # calculates the probability each point has of being in cluster 1 and 2
    p_cluster1 = probability([gmm[:, 0], gmm[:, 1]], list(parameters['mean1']), list(parameters['sd1']),parameters['lam'][0])
    p_cluster2 = probability([gmm[:, 0], gmm[:, 1]], list(parameters['mean2']), list(parameters['sd2']),parameters['lam'][1])

    # give each point a label based on which cluster it has a higher probability of being in,
    for i in range(0,gmm.shape[0]):
        if p_cluster1[i] > p_cluster2[i]:
            labels[i] = 1
            cluster[0] += 1
        else:
            labels[i] = 2
            cluster[1] += 1
    return cluster, labels

# Function maximizes the parameters based on the results of the last iteration
def maximization(cluster, parameters,labels):
    percent_cluster1 = cluster[0] / float(len(gmm))
    percent_cluster2 = 1 - percent_cluster1
    # initializes two arrays that will contain all the x and y values for points that are in cluster 1
    array1x = [0]*cluster[0]
    array1y = [0]*cluster[0]
    array1_index = 0

    # initializes two arrays that will contain all the x and y values for points that are in cluster 2
    array2x = [0]*cluster[1]
    array2y = [0]*cluster[1]
    array2_index = 0

    # puts each point's x and y value into an array based on its label
    for i in range(0, len(gmm)):
        if(labels[i] == 1):
            array1x[array1_index] = gmm[i, 0]
            array1y[array1_index] = gmm[i, 1]
            array1_index +=1
        else:
            array2x[array2_index] = gmm[i, 0]
            array2y[array2_index] = gmm[i, 1]
            array2_index +=1

    # uses the previously created x and y value arrays to calculates our new parameters
    parameters['lam'] = [percent_cluster1, percent_cluster2]
    parameters['mean1'] = [np.mean(array1x), np.mean(array1y)]
    parameters['mean2'] = [np.mean(array2x), np.mean(array2y)]
    parameters['sd1'] = [[np.std(array1x), 0], [0, np.std(array1y)]]
    parameters['sd2'] = [[np.std(array2x), 0], [0, np.std(array2y)]]
    return parameters

# calculates the distance between two points
def distance(a,b):
    return math.sqrt(abs(a[0]-b[0])**2 + abs(a[1]-b[1])**2)

# check the amount of change in mean1 and mean2 compared to the previous iteration
def check_change(new_parameters, parameters):
    mean1_change = distance(new_parameters['mean1'],parameters['mean1'])
    mean2_change = distance(new_parameters['mean2'],parameters['mean2'])
    return mean1_change + mean2_change

# set initial parameters
mean1 = [3,0]
sd1 = [[1, 0], [0, 1]]
mean2 = [0,-2]
sd2 = [[1, 0], [0, 1]]
lam = [0.5,0.5]

data = {'mean1': mean1, 'sd1': sd1, 'mean2': mean2, 'sd2': sd2, 'lam':lam}
parameters = pd.DataFrame(data=data)

final_labels = [0]*len(gmm)

change = 1
iteration = 1

# continues running until the amount of change falls below 0.01
while change > 0.01:
    new_parameters = parameters.copy()
    cluster, labels = expectation(gmm)
    parameters = maximization(cluster,new_parameters.copy(),labels)
    change = check_change(new_parameters,parameters)
    final_labels = labels
    # plots the results of each iteration
    plot(labels,iteration)
    iteration += 1

print parameters

# plot true values
plot(gmm[:, 2], 0)
