import array
import random
import pandas as pd
import numpy
import wandb

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

numpy.set_printoptions(threshold=100)

CONFIG = {
    "pathToDf": "data/",
    "num_iter": 31,
    "num_sample_df": 8,
    "num_population": 300,
}

df = pd.read_csv(CONFIG['pathToDf'] + 'dist_vologda_matrix.csv', index_col='Unnamed: 0')
df = df.iloc[:CONFIG['num_sample_df'], :CONFIG['num_sample_df']]  # Убрать при деплое, используется для ограничения размерности
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

toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalTSP)


def main():
    random.seed(169)

    pop = toolbox.population(n=CONFIG["num_population"])

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaSimple(
        pop, toolbox, 0.7, 0.2, 40, stats=stats, halloffame=hof
    )

    return pop, stats, hof, logbook


def wandb_register(logbook):
    result_df = pd.DataFrame(logbook).T
    run = wandb.init(project=f"TSPRL", entity="torchme")
    wandb.run.name = f"TPSGA-{CONFIG['num_sample_df']}x{CONFIG['num_sample_df']}"

    for i in result_df.columns:
        print()
        wandb.log(
            {
                "gen:": result_df.loc["gen", i],
                "nevals": result_df.loc["nevals", i],
                "avg": result_df.loc["avg", i],
                #'mean': result_df.loc['mean', i],
                "std": result_df.loc["std", i],
                "min": result_df.loc["min", i],
                "max": result_df.loc["max", i],
            }
        )
    run.finish()


if __name__ == "__main__":
    pop, stats, hof, logbook = main()
    wandb_register(logbook)
