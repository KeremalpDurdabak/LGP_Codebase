from modules.Parameter import Parameter
from modules.Dataset import Dataset
from modules.Population import Population
from modules.Display import Display


def main():
    dataset = Dataset()
    dataset.set_thyroiddisease()
    #Dataset.resample_data(heuristic=1, tau=200)

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

        #if generation % 10 == 0:
            #Dataset.resample_data(heuristic=2, tau=200)  # You can switch the heurist

        # Parent Selection -> Variation Operators (Crossover, Mutation) -> Child Replacement
        population.generate_next_gen()

    #Compute best individual's score on the test dataset
    population.compute_best_individual_test_fitness()
    
    # Display overall metrics as graphs    
    Display.report_overall_performance()

if __name__ == "__main__":
    main()