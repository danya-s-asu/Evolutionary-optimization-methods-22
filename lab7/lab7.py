import random
POPULATION_SIZE = 10
MAX_GENERATIONS = 10

GRAPH = [[0,1,0,1,0,0,1],
        [0,0,1,1,0,0,0],
        [0,1,0,1,0,0,0],
        [0,0,1,0,1,0,0],
        [0,1,0,0,0,1,0],
        [0,0,1,0,0,0,1],
        [0,0,0,0,0,0,0]]
FEROMON= [[0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0]]

index=[]
bots=[]

for i in range(len(GRAPH)):
    for j in range(len(GRAPH)):
        if GRAPH[i][j]>0:
            index.append(j)
    if len(index)>0:
        for j in range(len(index)):
            FEROMON[i][index[j]]=1/len(index)
        index.clear()

def Bots(nbot):
    for _ in range(nbot):
        bots.append(Bot())

class Bot:
    def __init__(self):
        self.Way=[0]
    
    def FindWay(self):
        p=0
        for _ in range(len(FEROMON)*len(FEROMON[0])):
            c=dict()
            tmp=0
            for i in range(len(GRAPH)):
                if GRAPH[p][i]>0:
                    tmp+=FEROMON[p][i]
                    c[i] = tmp
            if len(c)>0:
                for i in range(len(GRAPH)):
                    if i in c:
                        if c.get(i)>random.random():
                            p=i
                            self.Way.append(p)
                            break                    
            else:
                break
        for i in range(len(GRAPH)):
            if i not in self.Way:
                return False
        return True
    
    def resetFeromon(self):
        p=0
        n=[]
        for i in range(len(self.Way)):
            if (self.Way[i] not in n) and (FEROMON[p][self.Way[i]]>0):
                FEROMON[p][self.Way[i]]+=1/len(self.Way)
                n.append(self.Way[i])
                p=self.Way[i]
        ind=[]
        for i in range(len(GRAPH)):
            sum = 0
            for j in range(len(GRAPH)):
                if FEROMON[i][j]>0:
                    ind.append(j)
                    sum+=FEROMON[i][j]
            for j in range(len(ind)):
                FEROMON[i][ind[j]] = (FEROMON[i][ind[j]]  * 100) / (sum*100)
            ind.clear()
    
    def BestChanseWay(self):
        s = "0"
        p = 0
        maxi = 0
        m = -100
        for _ in range(len(GRAPH)-1):
            for  i in range(len(GRAPH)):
                if FEROMON[p][i] > m:
                    maxi = i
                    m = FEROMON[p][i]

            s+="->"+str(maxi)
            p = maxi
            m = -100
        return s
            
Bots(POPULATION_SIZE)

for i in range(MAX_GENERATIONS):
    for j in range(len(bots)):
        if bots[j].FindWay():
            bots[j].resetFeromon();
for i in range(len(FEROMON)):
    for j in range(len(FEROMON[i])):
        FEROMON[i][j]=float(round(FEROMON[i][j],1))

print("Graph")
print(*GRAPH, sep="\n")
print("feromons")
print(*FEROMON,sep="\n")
print(bots[0].BestChanseWay())
