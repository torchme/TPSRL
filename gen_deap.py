import array
import random
import pandas as pd
import numpy
import wandb
import time

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

numpy.set_printoptions(threshold=100)

CONFIG = {
    "pathToDf": "data/",
    "num_iter": 101,
    "num_sample_df": 26,
    "num_population": 300,
}

df = pd.read_csv(CONFIG['pathToDf'] + 'dist_vologda_matrix.csv', index_col='Unnamed: 0')
df = df.iloc[2:CONFIG['num_sample_df']+2, 2:CONFIG['num_sample_df']+2]  # Убрать при деплое, используется для ограничения размерности
IND_SIZE = df.shape[0]  # N
distance_map = df.values  # M


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="i", fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)

# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evalTSP(individual):
    distance = distance_map[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        distance += distance_map[gene1][gene2]
    return distance,

def compute_time(argument, *args, **kwargs):
    return time.time()

toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalTSP)
#toolbox.register("time", compute_time)


def main():
    random.seed(169)

    pop = toolbox.population(n=CONFIG["num_population"])

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    stats.register("time", compute_time)
    pop, logbook = algorithms.eaSimple(
        pop, toolbox, 0.7, 0.2, CONFIG['num_iter'], stats=stats, halloffame=hof
    )

    return pop, stats, hof, logbook


def wandb_register(logbook):
    result_df = pd.DataFrame(logbook)
    result_df['time_iter'] = [0, *result_df['time'].diff().dropna()]
    result_df['time'] = result_df['time_iter'].cumsum()
    result_df = result_df.T
    run = wandb.init(project=f"TSPRL", entity="torchme")
    wandb.run.name = f"TSP/GA-{CONFIG['num_sample_df']}x{CONFIG['num_sample_df']}"

    for i in result_df.columns:
        #start_time = time.time()
        print()
        wandb.log(
            {
                "gen:": result_df.loc["gen", i],
                "nevals": result_df.loc["nevals", i],
                "distance": result_df.loc["avg", i],
                #'mean': result_df.loc['mean', i],
                "std": result_df.loc["std", i],
                "min": result_df.loc["min", i],
                "max": result_df.loc["max", i],
                "time one iteration": result_df.loc["time_iter", i],
                "total time": result_df.loc["time", i]
            }
        )
    run.finish()


if __name__ == "__main__":
    start_time = time.time()
    pop, stats, hof, logbook = main()
    wandb_register(logbook)
