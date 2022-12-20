import operator
import random
import matplotlib.pyplot as plt
import numpy as np
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
HALL_OF_FAME_SIZE = 10 # Зал славы
MAX_GENERATIONS = 1000
POPULATION_SIZE=200

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", np.ndarray, fitness=creator.FitnessMin, speed=list, smin=None, smax=None, best=None)
def ffunction(x, y):
        return (1-x)*(1-x)+100*((y-x*x)*(y-x*x))
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(np.random.uniform(pmin, pmax, size)) 
    part.speed = np.random.uniform(smin, smax, size)
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, phi1, phi2):
    u1 = np.random.uniform(0, phi1, len(part))
    u2 = np.random.uniform(0, phi2, len(part))
    v_u1 = u1 * (part.best - part)
    v_u2 = u2 * (best - part)
    part.speed += v_u1 + v_u2
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part += part.speed
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2, pmin=-2.048, pmax=2.048, smin=-10, smax=10)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=1.0, phi2=1.0)
toolbox.register("evaluate", benchmarks.rosenbrock)
population = toolbox.population(POPULATION_SIZE)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
logbook = tools.Logbook()
best = None
for g in range(MAX_GENERATIONS):
    for part in population:
        part.fitness.values = toolbox.evaluate(part)
        if part.best is None or part.best.fitness < part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values
        if best is None or best.fitness < part.fitness:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values    

    for part in population:
        toolbox.update(part, best)
    logbook.record(gen=g, evals=len(population), **stats.compile(population))
    print(logbook.stream)
hof=best
maxFitnessValues, meanFitnessValues = logbook.select("min", "avg")
plt.plot(maxFitnessValues, c='r')
plt.plot(meanFitnessValues, c='g')
plt.xlabel('Генерация')
plt.ylabel('Максимальный/средний фитнес функции')
plt.title('Максимальный и средний результат генерации')
plt.show()
print(f'Лучший индивидуум \n x={hof[0]}, y={hof[1]}, f={ffunction(hof[0], hof[1])}, геном={hof}')

def main():
    fig = plt.figure()
    plt.title('(1-x)*(1-x)+100*((y-x*x)*(y-x*x))')
    ax = plt.axes(projection='3d')    
    x = np.linspace(-2.048, 2.048, 10)
    y = np.linspace(-2.048, 2.048, 10)
    X, Y = np.meshgrid(x, y) 
    Z = ffunction(X, Y) 
    ax = plt.axes(projection ='3d')
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, cmap='cool')
    ax.scatter(1, 1, ffunction(1, 1), marker='x', color='red', zorder=1, s=30)
    ax.view_init(elev=30, azim=45)
    ax.scatter(hof[0], hof[1], ffunction(hof[0], hof[1]), c='black', s=10)
    plt.show()

if __name__ == "__main__":
    main()