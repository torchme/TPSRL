import pandas as pd
import numpy as np
import random


PATH_DF = 'data/'

df = pd.read_csv(PATH_DF + 'dist_vologda_matrix.csv', index_col='Unnamed: 0').iloc[8:16+8, 8:16+8]

matrix = df.values

#indexes = df.index
cities = df.columns
encode_cities_dict = dict(zip(cities, [i for i in range(len(cities))]))
encode_cities = list(range(len(cities)))

ONE_MAX_LENGTH = len(encode_cities)

POPULATION_SIZE = 5000
P_CROSSOVER = 0.4
P_MUTATION = 0.1
MAX_GENERATIONS = 500

START_INDEX = 0

def populationCreator(n=0):
    population_sample_indexes = []
    population_sample = []
    population_sample_total_dist = []
    for i in range(n):
        sample_list_indexes, sample_list = createSampleList()
        population_sample_indexes.append(sample_list_indexes)
        population_sample.append(sample_list)
        population_sample_total_dist.append(countTotalDistance(sample_list))
    return population_sample_indexes, population_sample, population_sample_total_dist


def countTotalDistance(sample_list):
    total_dist = 0
    for i, j in sample_list:
        total_dist += matrix[i, j]

    return total_dist

def createSampleList():
    sample_list = random.sample(encode_cities, ONE_MAX_LENGTH)
    sample_list.remove(START_INDEX)
    _ = [START_INDEX]
    _ += sample_list
    sample_list.append(START_INDEX)

    return _, list(zip(_, sample_list))


def crossing_one_sample(firstSample, secondSample, P_CROSSOVER):
    length_sample = len(firstSample)
    index_crossing = int(P_CROSSOVER * len(firstSample)+1)
    firstCrossSameple = firstSample[:index_crossing]
    firstCrossSameple += [i for i in secondSample[index_crossing:] if i not in firstCrossSameple]
    if firstCrossSameple != length_sample:
        firstCrossSameple += [i for i in firstSample if i not in firstCrossSameple]

    secondCrossSameple = secondSample[:index_crossing]
    secondCrossSameple += [i for i in firstSample[index_crossing:] if i not in secondCrossSameple]
    if secondCrossSameple != length_sample:
        secondCrossSameple += [i for i in secondSample if i not in secondCrossSameple]

    #print(firstSample, secondSample)
    #print(index_crossing)
    #print(firstCrossSameple, secondCrossSameple)

    return firstCrossSameple, secondCrossSameple

#   crossing_one_sample(population_sample_indexes[0], population_sample_indexes[1], P_CROSSOVER)

def count_sample_list(sample_indexes):
    #print(sample_indexes)
    _ = [START_INDEX]
    _ = sample_indexes
    sample_indexes = sample_indexes + [START_INDEX]
    return list(zip(_, sample_indexes[1:]))
    #print(sample_indexes + [START_INDEX])


def get_sample_mutation(sample, N_MUTATIONS):
    #print(sample)
    count = 0
    _sample = sample.copy()
    _sample.remove(START_INDEX)
    mutations_indexes = random.sample(_sample, N_MUTATIONS) # for i in range(N_MUTATIONS)]
    print(_sample)
    if N_MUTATIONS % 2 == 0:
        while count < len(mutations_indexes):
            _ = _sample[count]
            _sample[count] = _sample[count+1]
            _sample[count+1] = _
            count += 2
    else:
        #Нужно дофиксить
        while count < len(mutations_indexes)-1:
            _sample[mutations_indexes[count]] = _sample[mutations_indexes[count+1]]
            count += 1
        _sample[mutations_indexes[count]] = _sample[mutations_indexes[0]]
    print(mutations_indexes)
    print(_sample)
    #print(sample)


def make_df(population_sample_indexes, population_sample, population_sample_total_dist):
    df = pd.DataFrame()
    df['indexes'] = population_sample_indexes
    df['sample'] = population_sample
    df['total_dist'] = population_sample_total_dist
    return df.sort_values(by='total_dist', ascending=True)

def get_inference(num_iter):
    population_sample_indexes, population_sample, population_sample_total_dist = populationCreator(POPULATION_SIZE)
    total_df = make_df(population_sample_indexes, population_sample, population_sample_total_dist)
    #print(f'Генерация')
    #print()
    #print(total_df.head(3))
    sub_sample_size = POPULATION_SIZE//2
    sub_sample_1 = total_df.sample(n=sub_sample_size)['indexes'].to_list()  # указать вручную 1/2 от POPULATION SIZE
    sub_sample_2 = total_df.sample(n=sub_sample_size)['indexes'].to_list()  # указать вручную 1/2 от POPULATION SIZE
    #sub_sample_1 = get_sample_mutation(sub_sample_1, 2)
    for num_iteration in range(num_iter):
        cross_population_sample_indexes = []

        for i in range(len(sub_sample_1)):
            # print(f'до {_1[i]} / {_2[i]}')
            sub_sample_1[i], sub_sample_2[i] = crossing_one_sample(sub_sample_1[i], sub_sample_2[i], P_CROSSOVER)
            cross_population_sample_indexes.append(sub_sample_1[i])
            cross_population_sample_indexes.append(sub_sample_2[i])

        cross_population_sample = [count_sample_list(i) for i in cross_population_sample_indexes]
        cross_population_sample_total_dist = [countTotalDistance(i) for i in cross_population_sample]
        #print(f'Итерация {num_iteration} - Результат:')
        df = make_df(cross_population_sample_indexes, cross_population_sample, cross_population_sample_total_dist)
        #print(df.head(3))
        total_df = pd.concat([total_df, df])

    return total_df

total_df = get_inference(100)

print(total_df.sort_values(by='total_dist', ascending=True))

#print(get_sample_mutation(total_df['indexes'][0].to_list()[0], 2)
