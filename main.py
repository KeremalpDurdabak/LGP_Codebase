from modules.Parameter import Parameter
from modules.Dataset import Dataset
from modules.Population import Population
from modules.Display import Display


def main():
    dataset = Dataset()
    dataset.set_iris()

    population = Population()
    population.initialize()

    for generation in range(1, Parameter.generations + 1):

        population.compute_generation_fitness()

        # Report the best individual for the current generation
        Display.report_best_individual(generation, population)
        Display.report_all_individuals(generation, population)
         # Collect data for new graphs
        best_instruction_count = population.get_best_instruction_count()
        Display.best_instruction_counts.append(best_instruction_count)

        # Collect data for new graphs
        best_class_accuracies = population.get_best_class_accuracies()
        Display.best_class_accuracies.append(best_class_accuracies)


        population.generate_next_gen()
    Display.report_overall_performance()


if __name__ == "__main__":
    main()