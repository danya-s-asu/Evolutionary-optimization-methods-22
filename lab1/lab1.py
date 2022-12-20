from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import matplotlib.pyplot as plt
import numpy as np
# константы генетического алгоритма
POPULATION_SIZE = 75 # количество индивидуумов в популяции
P_CROSSOVER = 0.9 # вероятность скрещивания
P_MUTATION = 0.1 # вероятность мутации индивидуума
MAX_GENERATIONS = 30 # максимальное количество поколений
HALL_OF_FAME_SIZE = 10 # Зал славы
BOUND_LOW, BOUND_UP = -20, -2.3 # границы
EPS = 0.001 # точность

toolbox = base.Toolbox()
toolbox.register("zeroOrOne", random.randint, 0, 1)
creator.create("FitnessMax", base.Fitness,weights=(1,))
creator.create("Individual", list,fitness=creator.FitnessMax)

def power_of_two(n):
    return len(bin(int(n)))-2

def chrom_length(low, up, eps):
    return power_of_two((up-low)/eps)

CHROM_LENGTH = chrom_length(BOUND_LOW, BOUND_UP, EPS)
toolbox.register("individualCreator",tools.initRepeat,creator.Individual, toolbox.zeroOrOne,CHROM_LENGTH)
toolbox.register("populationCreator", tools.initRepeat,list, toolbox.individualCreator)

def chrom_to_number(individual, low, up, length):
    x = int(''.join([str(numeric_string) for numeric_string in individual]),2)
    return low+x*(up-low)/(2**length-1)

def function(individual):
    x = chrom_to_number(individual, BOUND_LOW, BOUND_UP, CHROM_LENGTH)
    return np.cos(2*x)/(x**2),

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
stats.register("max", np.max)
stats.register("avg", np.mean)

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
population, logbook =algorithms.eaSimple(population, toolbox,cxpb=P_CROSSOVER, 
            mutpb=P_MUTATION,ngen=MAX_GENERATIONS, stats=stats, halloffame=hof,verbose=True)
            
maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
plt.plot(maxFitnessValues, c='r')
plt.plot(meanFitnessValues, c='g')
plt.xlabel('Генерация')
plt.ylabel('Максимальный/средний фитнес функции')
plt.title('Максимальный и средний результат генерации')
plt.show()
print(f'Индивидуумы в зале славы', *hof.items,sep="\n")
print(f'Лучший индивидуум \n x={chrom_to_number(hof.items[0], BOUND_LOW, BOUND_UP, CHROM_LENGTH)}, f={function(hof.items[0])[0]}, геном={hof.items[0]}')

def main():
    plt.title('cos(2*x)/(x^2) x∈[-20, -2.3]')
    x = np.linspace(-20, -2.3, 20000)
    plt.plot(x, np.cos(2*x)/(x**2))
    plt.scatter(-2.97971404, 0.10677771245, c='r', s=20, marker='x')
    for i in hof.items:
        plt.scatter(chrom_to_number(i, BOUND_LOW, BOUND_UP, CHROM_LENGTH), function(i)[0], c='g', s=10)
    plt.scatter(chrom_to_number(hof.items[0], BOUND_LOW, BOUND_UP, CHROM_LENGTH), function(hof.items[0])[0], c='b', s=20)
    plt.show()

if __name__ == "__main__":
    main()