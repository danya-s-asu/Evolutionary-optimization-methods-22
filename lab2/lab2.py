from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import matplotlib.pyplot as plt
import numpy as np
# константы генетического алгоритма
POPULATION_SIZE = 15 # количество индивидуумов в популяции
P_CROSSOVER = 0.9 # вероятность скрещивания
P_MUTATION = 0.1 # вероятность мутации индивидуума
MAX_GENERATIONS = 30 # максимальное количество поколений
HALL_OF_FAME_SIZE = 10 # Зал славы
BOUND_LOW, BOUND_UP = -2.048, 2.048
CHROM_LENGTH = 2    # длина хромосомы
RANDOM_SEED = 50
random.seed(RANDOM_SEED)

def randomPoint(a, b):
    return [random.uniform(a, b), random.uniform(a, b)]

toolbox = base.Toolbox()
toolbox.register("randomPoint", randomPoint, BOUND_LOW, BOUND_UP)
creator.create("FitnessMax", base.Fitness,weights=(-1,))
creator.create("Individual", list,fitness=creator.FitnessMax)

toolbox.register("individualCreator",tools.initIterate,creator.Individual, toolbox.randomPoint)
toolbox.register("populationCreator", tools.initRepeat,list, toolbox.individualCreator)

def ffunction(x, y):
        return (1-x)*(1-x)+100*((y-x*x)*(y-x*x))

def function(individual):
    x, y = individual 
    f = (1-x)*(1-x)+100*((y-x*x)*(y-x*x))
    return f,

toolbox.register("evaluate", function)
toolbox.register("select", tools.selTournament,tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit,indpb=1.0/CHROM_LENGTH)
population = toolbox.populationCreator(n=POPULATION_SIZE)
fitnessValues = list(map(toolbox.evaluate, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

fitnessValues = [individual.fitness.values[0] for individual in population]
maxFitnessValues = []
meanFitnessValues = []
stats = tools.Statistics(lambda ind:ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
population, logbook =algorithms.eaSimple(population, toolbox,cxpb=P_CROSSOVER, 
            mutpb=P_MUTATION,ngen=MAX_GENERATIONS, stats=stats, halloffame=hof,verbose=True)
            
maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")
plt.plot(maxFitnessValues, c='r')
plt.plot(meanFitnessValues, c='g')
plt.xlabel('Генерация')
plt.ylabel('Максимальный/средний фитнес функции')
plt.title('Максимальный и средний результат генерации')
plt.show()
print(f'Индивидуумы в зале славы', *hof.items,sep="\n")
print(f'Лучший индивидуум \n x={hof.items[0][0]}, y={hof.items[0][1]}, f={ffunction(hof.items[0][0], hof.items[0][1])}, геном={hof.items[0]}')

def main():
    fig = plt.figure()
    plt.title('(1-x)*(1-x)+100*((y-x*x)*(y-x*x))')
    ax = plt.axes(projection='3d')

    #cordiates for spiral
    
    x = np.linspace(-2.048, 2.048, 10)
    y = np.linspace(-2.048, 2.048, 10)
  
    X, Y = np.meshgrid(x, y) 
    Z = ffunction(X, Y) 
    ax = plt.axes(projection ='3d')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, cmap='cool')
    ax.scatter(1, 1, ffunction(1, 1), marker='x', color='red', zorder=1, s=30)
    for i in hof.items:
        ax.scatter(i[0], i[1], ffunction(i[0], i[1]), c='g', s=10)
    ax.view_init(elev=30, azim=45)
    ax.scatter(hof.items[0][0], hof.items[0][1], ffunction(hof.items[0][0], hof.items[0][1]), c='black', s=10)
    plt.show()

if __name__ == "__main__":
    main()