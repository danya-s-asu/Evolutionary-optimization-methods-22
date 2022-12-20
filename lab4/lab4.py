from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import matplotlib.pyplot as plt
import numpy as np
# константы генетического алгоритма
POPULATION_SIZE = 200 # количество индивидуумов в популяции
P_CROSSOVER = 0.5 # вероятность скрещивания
P_MUTATION = 0.5 # вероятность мутации индивидуума
MAX_GENERATIONS = 200 # максимальное количество поколений
HALL_OF_FAME_SIZE = 10 # Зал славы
RANDOM_SEED = 40
CHROM_LENGTH = 81
random.seed(RANDOM_SEED)
sudoku = [0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0]
items = [(5,5),
        (16,6),(17,1),
        (19,8),(20,6),(21,5),(26,9),
        (28,5),(29,7),(34,4),
        (39,1),(45,6),
        (46,6),(47,9),(52,7),(53,5),
        (57,9),(58,6),(59,4),
        (64,1),(65,2),(66,3),(68,9),
        (74,5),(78,2),(79,9)]
def insertnumber(sud):
    for i in items:
        sud[i[0]-1]=i[1]
    return sud

def printit(sud):
    r=0
    for i in sud:
        print(i, end =" ")
        r+=1
        if r==9:
            r=0
            print('\n') 

def getValue(Ind):
    erorrvalue=0
    for i in range(9):
        erorrvalue+=9-len(set(Ind[9*i:9*i+9]))
    for i in range(9):
        erorrvalue+=9-len(set(Ind[0+i:81:9]))
    for i in range(3):
        for j in range(3):
            erorrvalue+=9-len(set(Ind[3*i+27*j:3*i+3+27*j]+Ind[3*i+27*j+9:3*i+12+27*j]+Ind[3*i+27*j+18:3*i+21+27*j]))
    return erorrvalue,

def randomPoint():
    return random.randint(1,9)

toolbox = base.Toolbox()
toolbox.register("randomPoint", randomPoint)
creator.create("FitnessMax", base.Fitness,weights=(-1,))
creator.create("Individual", list,fitness=creator.FitnessMax)

toolbox.register("individualCreator",tools.initRepeat,creator.Individual, toolbox.randomPoint, CHROM_LENGTH)
toolbox.register("populationCreator", tools.initRepeat,list, toolbox.individualCreator)
toolbox.register("evaluate", getValue)

toolbox.register("select", tools.selTournament,tournsize=3)
toolbox.register("mate", tools.cxTwoPoint) 
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.01)

# create initial population (generation 0):
population = toolbox.populationCreator(n=POPULATION_SIZE)


# prepare the statistics object:
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)

# define the hall-of-fame object:
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

# perform the Genetic Algorithm flow with hof feature added:
population, logbook = algorithms.eaSimple(population, toolbox,
cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
stats=stats, halloffame=hof, verbose=True)
# print best solution found:
best = hof.items[0]
printit(best)
print("-- Best Ever Individual = ", best)
print("-- Best Ever Fitness = ", best.fitness.values[0])
print("-- Knapsack Items = ")

# extract statistics:
maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
# plot statistics:
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over Generations')
plt.show()