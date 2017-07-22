#implementaion of kmeans algorithm to group t-shirts based on their size, given the length and width of each
#initial centroids are calculated randomly and graph has been plotted for the same
#-----------------------------------------------


import math
import random
import copy
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

cnum=3
totaldata=100
samplept1=random.randrange(0,101)
samplept2=random.randrange(0,101)
samplept3=random.randrange(0,101)
Max=math.pow(10,10)

samples=[[random.randrange(20,70),random.randrange(15,80)] for i in range(101)]
X=np.array(samples)
#plt.scatter(X[:,0],X[:,1],s=10,linewidths=2)
#plt.show()
data=[]
centroids=[]
centroids1=[]
colors=["g.","r.","c."]
for p in samples:
    print(p)

class DataPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def set_x(self, x):
        self.x = x

    def get_x(self):
        return self.x

    def set_y(self, y):
        self.y = y

    def get_y(self):
        return self.y

    def set_cluster(self, clusterNumber):
        self.clusterNumber = clusterNumber

    def get_cluster(self):
        return self.clusterNumber


class Centroids:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def set_x(self, x):
        self.x = x

    def get_x(self):
        return self.x

    def set_y(self, y):
        self.y = y

    def get_y(self):
        return self.y


def initialize_centroids():
    p1= Centroids(samples[samplept1][0],samples[samplept1][1])
    centroids.append(p1)
    p1 = Centroids(samples[samplept2][0], samples[samplept2][1])
    centroids.append(p1)
    p1 = Centroids(samples[samplept3][0], samples[samplept3][1])
    centroids.append(p1)

    print("Centroids\leader initialized at:")
    for i in range(3):
        print "(", centroids[i].get_x(), ", ", centroids[i].get_y(), ")"
        print()


def initialize_data():

    for i in range(totaldata):

        newitem = DataPoint(samples[i][0], samples[i][1])
        if i==samplept1:
            newitem.set_cluster(i)
        elif i==samplept2:
            newitem.set_cluster(i)
        elif i==samplept3:
            newitem.set_cluster(i)
        else:
            newitem.set_cluster(None)

        data.append(newitem)
'''    for i in data:
        print("(", i.get_x(), ", ", i.get_y(),",",i.get_cluster(), ")")
        print()
'''

def get_distance(dx,dy,cx,cy):
    return math.sqrt(math.pow(cx-dx,2)+math.pow(cy-dy,2))

def recalculate_centroid():
    X=0
    Y=0
    totalincluster=0

    for j in range(cnum):
        X=0
        Y=0
        totalincluster=0
        for k in range(totaldata):
            if(data[k].get_cluster()==j):
                X+=data[k].get_x()
                Y+=data[k].get_y()
                totalincluster+=1

        if(totalincluster>0):
            centroids[j].set_x(X/totalincluster)
            centroids[j].set_y(Y/totalincluster)


def update_clusters():

    for i in range(totaldata):
        min=Max
        newcluster=0
        for j in range(cnum):
            dist=get_distance(data[i].get_x(),data[i].get_y(),centroids[j].get_x(),centroids[j].get_y())
            if(dist<min):
                min=dist
                newcluster=j
        data[i].set_cluster(newcluster)



def perform_kmeans():
    global centroids1
    initialize_centroids()
    initialize_data()
    c1=copy.copy(centroids[0])
    c2=copy.copy(centroids[1])
    c3=copy.copy(centroids[2])
    update_clusters()
    recalculate_centroid()

    while ((c1!=centroids[0]) or (c2!=centroids[1]) or (c3!=centroids[2])):
        c1 = centroids[0]
        c2 = centroids[1]
        c3 = centroids[2]
        recalculate_centroid()
        update_clusters()
	centroids1=[[c1.get_x(),c1.get_y()],[c2.get_x(),c2.get_y()],[c3.get_x(),c3.get_y()]]


def print_result():
    for i in range(cnum):
        print "Shirt size:",i+1,"includes leader:(",centroids[i].get_x(),",",centroids[i].get_y(),")"
        for j in range(totaldata):
            if(data[j].get_cluster()==i):
                print "(",data[j].get_x(),",",data[j].get_y(),")"
        print()
	for i in range(totaldata):
		plt.plot(X[i][0],X[i][1],colors[data[i].get_cluster()],markersize=10)
	cent=np.array(centroids1)
	plt.scatter(cent[:,0],cent[:,1],marker='x',s=100,linewidths=10)
    plt.show()



perform_kmeans()
print_result()



