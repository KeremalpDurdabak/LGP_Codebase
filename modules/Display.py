import matplotlib.pyplot as plt
import numpy as np

from modules.Dataset import Dataset

class Display:
    best_individuals = []
    mean_individuals = []
    worst_individuals = []

    @classmethod
    def report_best_individual(cls, generation, population):
        best_fitness_raw = np.max(population.fitness)
        total_instances = len(Dataset.X)
        best_fitness_percentage = (best_fitness_raw / total_instances) * 100
        print(f"Generation {generation} Best Individual: {best_fitness_percentage:.2f}%")
        cls.best_individuals.append(best_fitness_percentage)

    @classmethod
    def report_all_individuals(cls, generation, population):
        mean_fitness_raw = np.mean(population.fitness)
        worst_fitness_raw = np.min(population.fitness)
        total_instances = len(Dataset.X)
        
        mean_fitness_percentage = (mean_fitness_raw / total_instances) * 100
        worst_fitness_percentage = (worst_fitness_raw / total_instances) * 100

        cls.mean_individuals.append(mean_fitness_percentage)
        cls.worst_individuals.append(worst_fitness_percentage)


    @classmethod
    def report_overall_performance(cls):
        pass