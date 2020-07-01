import math
from typing import List

import numpy as np
from sklearn.datasets import load_wine, load_boston, load_iris, load_digits
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes


def get_weight(x, m,a):
    new_wi=pow(np.subtract(x, m), 2)/(2*pow(a,2))
    # sum=[]
    # for item in new_wi:
    #     sum.append(np.exp(item))
    return np.exp(-pow(np.subtract(x, m), 2)/(2*pow(a,2)))


def get_label_dict(X_train, y_train) -> dict:
    return {label: X_train[y_train == label] for label in np.unique(y_train)}


def get_mean(dict_part, part):
    m = np.mean(dict_part, axis=0)
    return {part: m}


def get_index(x, mu1):
    new_mu1 = mu1
    equal=False
    while not equal:
        num = np.zeros((1, x.shape[1]), float)
        den = np.zeros((1, x.shape[1]), float)
        for xi in x:
            wi = get_weight(xi, new_mu1,math.sqrt(x.shape[1] * (np.pi) * 2))
            num = num +(np.multiply(wi , xi))
            den = den+ wi
        i=new_mu1
        new_mu1 = np.divide(num ,den)
        subtract=np.exp(np.subtract(i,new_mu1))
        equal = (np.round(subtract,decimals=11)==1).all()

    return new_mu1


def get_mean_for_each_class(x, y):
    means = []
    mean_shift = []
    dictionary = get_label_dict(x, y)
    for label in np.unique(y):
        mu1 = get_mean(dictionary[label], label)
        means.append(mu1)
        mean_shift.append({label: get_index(dictionary[label], mu1[label])})

    return means, mean_shift


def euclidean(x, m):
    x_minus_m = x - m
    euclid = np.dot(x_minus_m, x_minus_m.T)
    return euclid


def mean_shift(x, m, a):
    new_x = np.array(pow(np.subtract(x, m), 2) / (2 * (a * a)))[0]
    new_x=np.exp(new_x)
    mean_shift_distance=np.dot(new_x, new_x.T)
    return math.sqrt(mean_shift_distance)


def cluster_on_euclidean_distance(x, means: List[dict] = None):
    euclid = []
    for mean in means:
        euclid.append(euclidean(x, list(mean.values())[0]))
        # mahal.append(mahalanobis(x, list(mean.values())[0]))
    min_dist = euclid.index(min(euclid))
    return list(means[min_dist].keys())[0]


def cluster_on_mean_shift(x, means: List[dict] = None, n=100):
    meanshift = []
    for mean in means:
        meanshift.append(mean_shift(x, list(mean.values())[0], math.sqrt(n * (np.pi) * 2)))
        # mahal.append(mahalanobis(x, list(mean.values())[0]))
    min_dist = meanshift.index(np.min(meanshift))
    return list(means[min_dist].keys())[0]


def classification(x, y):
    dist = []
    mean_shift_distance=[]
    new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(x, y, test_size=0.9, random_state=True)
    means, mean_shift = get_mean_for_each_class(new_X_train, new_y_train)
    print("indicator mean")
    len_x=len(new_X_test)
    for i in range(0, len_x):
        a = cluster_on_euclidean_distance(new_X_test[i], means=means)
        b=cluster_on_mean_shift(new_X_test[i],means=mean_shift,n=len_x)
        dist.append(a)
        mean_shift_distance.append(b)
    get_confusion_matrix(y_test=new_y_test, dist=dist)
    print("indicator find using meanshift distance")
    get_confusion_matrix(y_test=new_y_test, dist=mean_shift_distance)



def get_confusion_matrix(y_test, dist):
    print('Confusion Matrix :')
    print(confusion_matrix(y_test, dist))
    print('Accuracy Score :', accuracy_score(y_test, dist))


# x, y = load_wine(return_X_y=True)
x,y=load_digits(return_X_y=True)
classification(x, y)
