import csv
import pickle
import os
import codecs
import numpy as np
from urllib.request import urlopen
import matplotlib.pyplot as plt
class TravelingSalesmanProblem:
    def __init__(self, name):
        self.name = name
        self.locations = []
        self.distances = []
        self.tspSize = 0
        self.__initData()

    def __len__(self):
        return self.tspSize

    def __createData(self):
            self.locations = []
            with urlopen("http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/" + self.name + ".tsp") as f:
                reader = csv.reader(codecs.iterdecode(f, 'utf-8'), delimiter=" ", skipinitialspace=True)
                for row in reader:
                    if row[0] in ('DISPLAY_DATA_SECTION', 'NODE_COORD_SECTION'):
                        break
                for row in reader:
                    if row[0] != 'EOF':
                        del row[0]
                        self.locations.append(np.asarray(row, dtype=np.float32))
                    else:
                        break
            self.tspSize = len(self.locations)
            print("length = {}, locations = {}".format(self.tspSize,
            self.locations))
            self.distances = [[0] * self.tspSize for _ in
            range(self.tspSize)]
            for i in range(self.tspSize):
                for j in range(i + 1, self.tspSize):
                    distance = np.linalg.norm(self.locations[j] -
                    self.locations[i])
                    self.distances[i][j] = distance
                    self.distances[j][i] = distance
                    print("{}, {}: location1 = {}, location2 = {} => distance = {}".format(i, j, self.locations[i], self.locations[j],distance))
            if not os.path.exists("tsp-data"):
                os.makedirs("tsp-data")
            pickle.dump(self.locations, open(os.path.join("tsp-data", self.name + "-loc.pickle"), "wb"))
            pickle.dump(self.distances, open(os.path.join("tsp-data", self.name + "-dist.pickle"), "wb"))

    def __initData(self):
        try:
            self.locations = pickle.load(open(os.path.join("tsp-data", self.name + "-loc.pickle"), "rb"))
            self.distances = pickle.load(open(os.path.join("tsp-data", self.name + "-dist.pickle"), "rb"))
        except (OSError, IOError):
            pass
        if not self.locations or not self.distances:
            self.__createData()
        self.tspSize = len(self.locations)
        
    def getTotalDistance(self, indices):
        distance = self.distances[indices[-1]][indices[0]]
        for i in range(len(indices) - 1):
            distance += self.distances[indices[i]][indices[i + 1]]
        return distance
    def plotData(self, indices):
        plt.scatter(*zip(*self.locations), marker='.', color='red')
        locs = [self.locations[i] for i in indices]
        locs.append(locs[0])
        plt.plot(*zip(*locs), linestyle='-', color='blue')
        return plt

def main():
    tsp = TravelingSalesmanProblem("eil51")
    optimalSolution = [1,22,8,26,31,28,3,36,35,20,2,29,21,16,50,34,30,9,49,10,39,33,45,15,44,42,40,19,41,13,25,14,24,43,7,23,48,6,27,51,46,12,47,18,4,17,37,5,38,11,32]
    print("Problem name: " + tsp.name)
    print("Optimal solution = ", optimalSolution)
    print("Optimal distance = ", tsp.getTotalDistance(optimalSolution))
    # plot the solution:
    plot = tsp.plotData(optimalSolution)
    plot.show()

if __name__ == "__main__":
    main()