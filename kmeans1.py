#implementation of Kmeans algorithm to determine cluster of students based on cgpa obtained by each
#Initial centroids are determined randomly
#-------------------------------------------------------------


import math
import random
import copy
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
samples=[]

f=open("input_data.txt")
str=f.read()
for line in str.split('\n'):
    x=line.split()
    if x:
        temp=[' '.join(x[:-1]),float(x[-1])]
        samples.append(temp)

cnum=3
totaldata=len(samples)
samplept1=random.randrange(0,totaldata)
samplept2=random.randrange(0,totaldata)
samplept3=random.randrange(0,totaldata)
Max=math.pow(10,10)


X=np.array([[a[1],0.0] for a in samples])
#plt.scatter(X[:,0],X[:,1],s=10,linewidths=2)
#plt.show()
data=[]
centroids=[]
centroids1=[]
colors=["g.","r.","c."]
for p in samples:
    print(p)

class DataPoint:
    def __init__(self,name, x, y=0.0):
        self.name=name
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
	
    def get_name(self):
        return self.name

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
    p1= Centroids(samples[samplept1][1],0.0)
    centroids.append(p1)
    p1 = Centroids(samples[samplept2][1], 0.0)
    centroids.append(p1)
    p1 = Centroids(samples[samplept3][1], 0.0)
    centroids.append(p1)

    print "Centroids\leader initialized at:"
    for i in range(3):
        print centroids[i].get_x()


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
            centroids[j].set_x(round(X/totalincluster,2))
            centroids[j].set_y(round(Y/totalincluster,2))


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
        print "---------------------------------------------"
        print "Cluster:",i+1,"includes leader:",centroids[i].get_x()
        for j in range(totaldata):
            if(data[j].get_cluster()==i):
                print data[j].get_name(),":",data[j].get_x()
	for i in range(totaldata):
		plt.plot(X[i][0],X[i][1],colors[data[i].get_cluster()],markersize=5)
	cent=np.array(centroids1)
	plt.scatter(cent[:,0],cent[:,1],marker='x',s=100,linewidths=10)
    plt.show()



perform_kmeans()
print_result()



