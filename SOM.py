import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

K = 100
NUM_OF_SAMPLES = 20000


def retrieve_data(num_of_samples, p=0.95, f=4, sigma_x=0.1):
    x_1 = np.random.uniform(-1, 1, num_of_samples)
    x_2 = np.random.uniform(0, 1, num_of_samples)
    temp = [random.uniform(-1, 1) if x_2[i] > p
            else np.sin(f * x_1[i] + get_epsilon(sigma_x))
            for i in range(num_of_samples)]
    samples = np.zeros((num_of_samples, 2))
    samples[:, 0], samples[:, 1] = x_1, temp

    return samples


def get_prototypes(prototypes_num):
    return np.random.uniform(-1, 1, (prototypes_num, 2))


def get_epsilon(sigma_x):
    epsilon = np.random.normal(0, sigma_x ** 2)
    return epsilon


def find_nearest_prototype_index(sample, prototypes):
    arr = [euclid_distance(sample, prototype) for prototype in prototypes]
    return np.argmin(np.array(arr))


def euclid_distance(first, second):
    return sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)


def get_counted(nearest_index):
    prototype_indexes = np.arange(K)
    return np.power(prototype_indexes - nearest_index, 2)


def get_pi_func(nearest_index, sigma):
    counted = get_counted(nearest_index)
    denoted = 2 * (sigma ** 2)
    result = np.exp((-1) * counted / denoted)
    return result / np.sum(result)


def update_prototypes(nearest_index, prototypes, sample, eta=1, sigma=4):
    pi_func = get_pi_func(nearest_index, sigma).reshape((K, 1))
    samples_vec = np.array([sample for _ in range(K)])
    prototypes = prototypes + (eta * pi_func * (samples_vec - prototypes))
    return prototypes


def plot_SOM(samples, prototypes, step):
    plt.scatter(samples[:, 0], samples[:, 1], c='c', label='samples', alpha=0.3)
    plt.scatter(prototypes[:, 0], prototypes[:, 1], c='b', alpha=0.6, label='prototypes')
    plt.plot(prototypes[:, 0], prototypes[:, 1], c='b', alpha=0.7)
    plt.title(f'Prototypes at {step}')
    plt.show()


def main():
    data = retrieve_data(NUM_OF_SAMPLES)
    prototypes = get_prototypes(K)
    plot_SOM(data, prototypes, 'Initialization')
    for sample in data:
        nearest_prototype_index = find_nearest_prototype_index(sample, prototypes)
        prototypes = update_prototypes(nearest_prototype_index, prototypes, sample)
    plot_SOM(data, prototypes, 'End')


if __name__ == '__main__':
    main()

    
    
