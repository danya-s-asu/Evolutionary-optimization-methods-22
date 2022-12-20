from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap import benchmarks
from deap import cma

import matplotlib.pyplot as plt
import numpy as np
MAX_GENERATIONS = 30
# Problem size
N=2
def ffunction(x, y):
        return (1-x)*(1-x)+100*((y-x*x)*(y-x*x))
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("evaluate", benchmarks.rastrigin)
np.random.seed(128)
  
strategy = cma.Strategy(centroid=[10.0]*N, sigma=10.0, lambda_=20*N)
toolbox.register("generate", strategy.generate, creator.Individual)
toolbox.register("update", strategy.update)
toolbox.register("evaluate", benchmarks.rosenbrock)

hof = tools.HallOfFame(10)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
population, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=MAX_GENERATIONS, stats=stats, halloffame=hof)

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