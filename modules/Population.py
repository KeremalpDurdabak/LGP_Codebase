import numpy as np
from modules.Dataset import Dataset
from modules.Parameter import Parameter

class Population:
    def __init__(self):
        self.target_index = np.random.randint(0, Parameter.target_index, (Parameter.max_instruction, Parameter.population_count))
        self.source_index = np.random.randint(0, Parameter.source_index, (Parameter.max_instruction, Parameter.population_count))
        self.operator_select = np.random.randint(0, Parameter.operator_select, (Parameter.max_instruction, Parameter.population_count))
        self.source_select = np.random.randint(0, Parameter.source_select, (Parameter.max_instruction, Parameter.population_count))
        self.fitness = np.zeros(Parameter.population_count)

    def initialize(self):
        for col in range(Parameter.population_count):
            instruction_count = np.random.randint(1, Parameter.max_instruction)
            self.target_index[instruction_count, col] = -1
            self.source_index[instruction_count, col] = -1
            self.operator_select[instruction_count, col] = -1
            self.source_select[instruction_count, col] = -1

    def compute_generation_fitness(self):
        true_labels = np.argmax(Dataset.y_train, axis=1)
        self.fitness = np.zeros(Parameter.population_count)

        for col in range(Parameter.population_count):
            # Note: No need to copy Dataset.X into registers here, as we use modulus to index Dataset.X when needed
            registers = np.zeros((Dataset.X_train.shape[0], Parameter.register_count))

            for ins_idx in range(Parameter.max_instruction):
                if self.target_index[ins_idx, col] == -1:
                    break

                target = self.target_index[ins_idx, col] #% Parameter.target_index
                operator = self.operator_select[ins_idx, col] #% Parameter.operator_select
                source_index = self.source_index[ins_idx, col] #% Parameter.source_index
                source_selector = self.source_select[ins_idx, col] #% Parameter.source_select

                if source_selector == 0:
                    source_value = registers[:, source_index % Parameter.register_count]
                elif source_selector == 1:
                    source_value = Dataset.X_train[:, source_index] #% Dataset.X.shape[1]]

                self.apply_operator(target, operator, source_value, registers)

            predicted_labels = np.argmax(registers[:, :Dataset.y_train.shape[1]], axis=1)
            self.fitness[col] = np.sum(predicted_labels == true_labels)


    def generate_next_gen(self):
        # Step 1: Selection
        sorted_fitness_indices = np.argsort(self.fitness)
        best_indices = sorted_fitness_indices[-int(Parameter.population_count * (1 - Parameter.gap_percentage)):]  # Best performers
        worst_indices = sorted_fitness_indices[:int(Parameter.population_count * Parameter.gap_percentage)]  # Worst performers

        new_children_count = len(worst_indices)

        # Initialize children arrays
        children_target_index = np.zeros((Parameter.max_instruction, new_children_count), dtype=int)
        children_source_index = np.zeros((Parameter.max_instruction, new_children_count), dtype=int)
        children_operator_select = np.zeros((Parameter.max_instruction, new_children_count), dtype=int)
        children_source_select = np.zeros((Parameter.max_instruction, new_children_count), dtype=int)

        # Step 2: Crossover
        for i in range(new_children_count):
            parent1_idx = np.random.choice(best_indices)
            parent2_idx = np.random.choice(best_indices)

            if np.random.rand() < 0.5:  # Single-point crossover
                crossover_point = np.random.randint(0, Parameter.max_instruction)
                children_target_index[:crossover_point, i] = self.target_index[:crossover_point, parent1_idx]
                children_target_index[crossover_point:, i] = self.target_index[crossover_point:, parent2_idx]

                # Repeat for other attributes
                children_source_index[:crossover_point, i] = self.source_index[:crossover_point, parent1_idx]
                children_source_index[crossover_point:, i] = self.source_index[crossover_point:, parent2_idx]
                
                children_operator_select[:crossover_point, i] = self.operator_select[:crossover_point, parent1_idx]
                children_operator_select[crossover_point:, i] = self.operator_select[crossover_point:, parent2_idx]
                
                children_source_select[:crossover_point, i] = self.source_select[:crossover_point, parent1_idx]
                children_source_select[crossover_point:, i] = self.source_select[crossover_point:, parent2_idx]

            else:  # Double-point crossover
                point1, point2 = np.sort(np.random.randint(0, Parameter.max_instruction, 2))

                children_target_index[:point1, i] = self.target_index[:point1, parent1_idx]
                children_target_index[point1:point2, i] = self.target_index[point1:point2, parent2_idx]
                children_target_index[point2:, i] = self.target_index[point2:, parent1_idx]

                # Repeat for other attributes
                children_source_index[:point1, i] = self.source_index[:point1, parent1_idx]
                children_source_index[point1:point2, i] = self.source_index[point1:point2, parent2_idx]
                children_source_index[point2:, i] = self.source_index[point2:, parent1_idx]
                
                children_operator_select[:point1, i] = self.operator_select[:point1, parent1_idx]
                children_operator_select[point1:point2, i] = self.operator_select[point1:point2, parent2_idx]
                children_operator_select[point2:, i] = self.operator_select[point2:, parent1_idx]
                
                children_source_select[:point1, i] = self.source_select[:point1, parent1_idx]
                children_source_select[point1:point2, i] = self.source_select[point1:point2, parent2_idx]
                children_source_select[point2:, i] = self.source_select[point2:, parent1_idx]

        # Step 3: Mutation
        mutation_mask = np.random.rand(Parameter.max_instruction, new_children_count) < Parameter.mutation_prob
        children_target_index[mutation_mask] = np.random.randint(0, Parameter.target_index, np.sum(mutation_mask))

        # Repeat mutation for other attributes
        children_source_index[mutation_mask] = np.random.randint(0, Parameter.source_index, np.sum(mutation_mask))
        children_operator_select[mutation_mask] = np.random.randint(0, Parameter.operator_select, np.sum(mutation_mask))
        children_source_select[mutation_mask] = np.random.randint(0, Parameter.source_select, np.sum(mutation_mask))

       # Compute only the new children's fitness
        new_fitness = np.zeros(new_children_count)
        
        true_labels = np.argmax(Dataset.y_train, axis=1)
        
        for i, col in enumerate(worst_indices):
            registers = np.zeros((Dataset.X_train.shape[0], Parameter.register_count))

            for ins_idx in range(Parameter.max_instruction):
                if self.target_index[ins_idx, col] == -1:
                    break

                target = self.target_index[ins_idx, col] % Parameter.target_index
                operator = self.operator_select[ins_idx, col] % Parameter.operator_select
                source_index = self.source_index[ins_idx, col] % Parameter.source_index  # Keep this line as a reference if you want
                source_selector = self.source_select[ins_idx, col] % Parameter.source_select

                if source_selector == 0:
                    source_value = registers[:, self.source_index[ins_idx, col] % Parameter.register_count]
                elif source_selector == 1:
                    source_value = Dataset.X_train[:, self.source_index[ins_idx, col] % Dataset.X_train.shape[1]]


                self.apply_operator(target, operator, source_value, registers)

            predicted_labels = np.argmax(registers[:, :Dataset.y_train.shape[1]], axis=1)
            new_fitness[i] = np.sum(predicted_labels == true_labels)
        
        # Step 4: Replace the worst individuals with new children
        self.target_index[:, worst_indices] = children_target_index
        self.source_index[:, worst_indices] = children_source_index
        self.operator_select[:, worst_indices] = children_operator_select
        self.source_select[:, worst_indices] = children_source_select
        self.fitness[worst_indices] = new_fitness
        
        # Sort the entire population by fitness
        sorted_indices = np.argsort(self.fitness)
        self.target_index = self.target_index[:, sorted_indices]
        self.source_index = self.source_index[:, sorted_indices]
        self.operator_select = self.operator_select[:, sorted_indices]
        self.source_select = self.source_select[:, sorted_indices]
        self.fitness = self.fitness[sorted_indices]


    # Additional function to handle operator logic
    def apply_operator(self, target, operator, source_value, registers):
        if operator == 0:
            registers[:, target] += source_value
        elif operator == 1:
            registers[:, target] -= source_value
        elif operator == 2:
            registers[:, target] *= 2
        elif operator == 3:
            registers[:, target] = registers[:, target] / 2 if np.all(registers[:, target] != 0) else 0

    def get_best_instruction_count(self):
        best_idx = np.argmax(self.fitness)
        return np.count_nonzero(self.target_index[:, best_idx] != -1)

    def get_best_class_accuracies(self):
        best_idx = np.argmax(self.fitness)
        true_labels = np.argmax(Dataset.y_train, axis=1)
        predicted_labels = self.get_predicted_labels(best_idx)
        
        unique_labels = np.unique(true_labels)
        class_accuracies = []
        
        for label in unique_labels:
            correct_preds = np.sum((predicted_labels == label) & (true_labels == label))
            total_instances = np.sum(true_labels == label)
            
            if total_instances > 0:
                accuracy = (correct_preds / total_instances) * 100
                class_accuracies.append(accuracy)
            else:
                class_accuracies.append(None)
        
        return class_accuracies

    def get_predicted_labels(self, individual_idx):
        true_labels = np.argmax(Dataset.y_train, axis=1)
        registers = np.zeros((Dataset.X_train.shape[0], Parameter.register_count))

        for ins_idx in range(Parameter.max_instruction):
            if self.target_index[ins_idx, individual_idx] == -1:
                break

            target = self.target_index[ins_idx, individual_idx]
            operator = self.operator_select[ins_idx, individual_idx]
            source_index = self.source_index[ins_idx, individual_idx]
            source_selector = self.source_select[ins_idx, individual_idx]

            if source_selector == 0:
                source_value = registers[:, source_index % Parameter.register_count]
            elif source_selector == 1:
                source_value = Dataset.X_train[:, source_index % Dataset.X_train.shape[1]]

            self.apply_operator(target, operator, source_value, registers)

        predicted_labels = np.argmax(registers[:, :Dataset.y_train.shape[1]], axis=1)
        return predicted_labels

    def compute_best_individual_test_fitness(self):
        best_idx = np.argmax(self.fitness)
        true_labels_test = np.argmax(Dataset.y_test, axis=1)  # Reverse one-hot encoding
        unique_labels = np.unique(true_labels_test)
        registers = np.zeros((Dataset.X_test.shape[0], Parameter.register_count))

        for ins_idx in range(Parameter.max_instruction):
            if self.target_index[ins_idx, best_idx] == -1:
                break

            target = self.target_index[ins_idx, best_idx]
            operator = self.operator_select[ins_idx, best_idx]
            source_index = self.source_index[ins_idx, best_idx]
            source_selector = self.source_select[ins_idx, best_idx]

            if source_selector == 0:
                source_value = registers[:, source_index % Parameter.register_count]
            elif source_selector == 1:
                source_value = Dataset.X_test[:, source_index % Dataset.X_test.shape[1]]

            self.apply_operator(target, operator, source_value, registers)

        predicted_labels_test = np.argmax(registers[:, :Dataset.y_test.shape[1]], axis=1)
        overall_test_fitness = np.sum(predicted_labels_test == true_labels_test) / len(true_labels_test) * 100

        print(f"Best Individual Test Fitness: {overall_test_fitness:.2f}%")

        # Calculate class-wise accuracy
        for label in unique_labels:
            mask = (true_labels_test == label)
            class_count = np.sum(mask)
            correct_class_count = np.sum(predicted_labels_test[mask] == true_labels_test[mask])
            class_accuracy = (correct_class_count / class_count) * 100
            print(f"Accuracy for class {label}: {class_accuracy:.2f}%")
