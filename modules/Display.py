import matplotlib.pyplot as plt
import numpy as np

from modules.Dataset import Dataset

class Display:
    best_individuals = []
    mean_individuals = []
    worst_individuals = []
    best_class_accuracies = []
    best_instruction_counts = []


    @classmethod
    def report_best_individual(cls, generation, population):
        best_fitness_raw = np.max(population.fitness)
        mean_fitness_raw = np.mean(population.fitness)
        worst_fitness_raw = np.min(population.fitness)
        
        total_instances = len(Dataset.X_train)
        
        best_fitness_percentage = (best_fitness_raw / total_instances) * 100
        mean_fitness_percentage = (mean_fitness_raw / total_instances) * 100
        worst_fitness_percentage = (worst_fitness_raw / total_instances) * 100
        
        print(f"Generation {generation} Best Individual: {best_fitness_percentage:.2f}%")
        
        best_class_accuracies = population.get_best_class_accuracies()
        formatted_accuracies = [f"{x:.2f}" if x is not None else "None" for x in best_class_accuracies]
        
        unique_labels = Dataset.unique_labels  # Use this line to get unique labels
        print(f"Class-wise accuracies: {dict(zip(unique_labels, formatted_accuracies))}")
                
        cls.best_individuals.append(best_fitness_percentage)
        cls.mean_individuals.append(mean_fitness_percentage)
        cls.worst_individuals.append(worst_fitness_percentage)


    @classmethod
    def report_all_individuals(cls, generation, population):
        total_instances = len(Dataset.X_train)
        fitness_percentages = (population.fitness / total_instances) * 100
        sorted_fitness_percentages = np.sort(fitness_percentages)[::-1]  # Sort in descending order
        
        print(f"All Fitness Scores (Sorted): {['{:.2f}%'.format(x) for x in sorted_fitness_percentages]}")


    @classmethod
    def report_overall_performance(cls):
        plt.figure(figsize=(8,6))

        # Graph 1
        plt.subplot(3, 1, 1)
        plt.plot(cls.best_individuals, color='#3498DB', label='Best')
        plt.plot(cls.mean_individuals, color='#F39C12', label='Mean')
        plt.plot(cls.worst_individuals, color='#E74C3C', label='Worst')
        plt.title('Fitness Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()

        # Graph 2
        plt.subplot(3, 1, 2)
        class_accuracies = np.array(cls.best_class_accuracies)
        unique_labels = Dataset.unique_labels  # Use this line to get unique labels
        for i, label in enumerate(unique_labels):
            plt.plot(class_accuracies[:, i], label=f'{label}')  # Use label instead of Class {i}
        plt.title('Class Accuracies of Best Individual')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend()

        # Graph 3
        plt.subplot(3, 1, 3)
        plt.plot(cls.best_instruction_counts)
        plt.title('Instruction Count of Best Individual')
        plt.xlabel('Generation')
        plt.ylabel('Instruction Count')

        plt.tight_layout()
        plt.show()
