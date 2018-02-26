import random
import classifiers

# Function to get the ga individual
# It consists in and array of all tunable parameter values corresponding to the classifiers
def generate_individual():
    # --- Random Forest parameters ---
    # Estimators between 5 and 20 (since the default value is 10)
    n_estimators_rf = random.randint(5, 20)
    # bootstrap could be 0 or 1
    bootstrap = random.randint(0, 1)
    # Seed between 3 and 4 digits
    random_state_rf = random.randint(100, 9999)

    # --- Ada Boost parameters ---
    # Estimators between 25 and 100 (since the default value is 50)
    n_estimators_ab = random.randint(25, 100)
    # Learning rate between 0 and 1
    learning_rate_ab = random.uniform(0, 1)
    # Seed between 3 and 4 digits
    random_state_ab = random.randint(100, 9999)

    # --- Gradient Tree Boosting parameters ---
    # Loss could be 0 or 1
    loss = random.randint(0, 1)
    # Learning rate between 0 and 1
    learning_rate_gtb = random.uniform(0, 1)
    # Estimators between 70 and 300 (since the default value is 100 and large values are preferred)
    n_estimators_gtb = random.randint(70, 300)
    # Between 2 and 10 (since the default value is 3)
    max_depth = random.randint(2, 10)
    # Seed between 3 and 4 digits
    random_state_gtb = random.randint(100, 9999)

    # Return array containing every parameter value
    return [
        # Random Forest parameters
        n_estimators_rf,
        bootstrap,
        random_state_rf,
        # Ada Boost parameters
        n_estimators_ab,
        learning_rate_ab,
        random_state_ab,
        # Gradient Tree Boosting parameters
        loss,
        learning_rate_gtb,
        n_estimators_gtb,
        max_depth,
        random_state_gtb,
    ]

# Function to generate population given a number of individuals
def generate_population(n):
    population = []
    i = 0
    while i < n:
        population.append(
            generate_individual()
        )
        i += 1
    return population

# Fitness function. Our fitness is the ROC value provided by the generated model
def fitness(
        individual,
        X_train,
        y_train,
        X_test,
        y_test,
):
    roc, clf = classifiers.run(
        X_train,
        y_train,
        X_test,
        y_test,
        individual[0],
        individual[1],
        individual[2],
        individual[3],
        individual[4],
        individual[5],
        individual[6],
        individual[7],
        individual[8],
        individual[9],
        individual[10],
    )
    return roc

# Selection function
def selection(population, fitness_results, fitness_mean):
    # The selected individuals are the ones with a ROC value above average
    selected = []
    for i in range(0, len(population)):
        if fitness_results[i] > fitness_mean:
            selected.append(population[i])
    return selected

# Crossover function
def crossover(individual1, individual2):
    child = []
    for i in range(len(individual1)):
        # Generate a random value to determinate if a certain value will be taken from individual 1 or 2
        rand = random.randint(1, 2)
        if rand == 1:
            child.append(individual1[i])
        else:
            child.append(individual2[i])
    return child

# Reproductive function
def reproduce(n, selected):
    new_population = []
    i = 0
    while i < n:
        # Select two random individuals from the selected ones to reproduce
        pos1 = random.randint(0, len(selected)-1)
        pos2 = random.randint(0, len(selected)-1)
        # Make sure the positions are different (only if there is more than one selected)
        while len(selected) > 1 and pos1 == pos2:
            pos2 = random.randint(0, len(selected)-1)
        result = crossover(
            selected[pos1],
            selected[pos2]
        )
        new_population.append(result)
        i += 1
    return new_population

# Mutation function
def mutation(individual, mutation_prob):
    mutated = individual
    # Only mutate the individual if the random number generated is smaller than the mutation probability provided
    prob = random.randrange(100)
    if prob <= mutation_prob:
        # We'll exchange similar values
        # Values at position 1 (bootstrap) and 6 (loss) are 0 or 1
        mutated[1] = individual[6]
        mutated[6] = individual[1]
        # Values at position 2, 5 and 10 correspond to random state
        mutated[2] = individual[5]
        mutated[5] = individual[10]
        mutated[10] = individual[2]
        # Values at position 4 and 7 correspond to learning_rate (between 0 and 1)
        # Those will be recalculated
        mutated[4] = random.uniform(0, 1)
        mutated[7] = random.uniform(0, 1)
    return mutated

def run(
        X_train,
        y_train,
        X_test,
        y_test,
        n,
        max_iter,
        mutation_prob
):
    # Generate population with the provided size
    population = generate_population(n)
    for i in range(0, max_iter):
        print("(Population #{})=======================================".format(i + 1))
        # print('Population size:', len(population))
        # Calculate fitness for every individual
        fitness_results = []
        for individual in population:
            print('Calculating fitness for individual: ', individual)
            fitness_results.append(fitness(individual, X_train, y_train, X_test, y_test))
        # Calculate fitness average
        fitness_total = 0
        for j in range(0, len(population)):
            fitness_total += fitness_results[j]
        fitness_mean = fitness_total / len(fitness_results)
        print("Fitness mean: {}".format(fitness_mean))

        # Select most optimum individuals
        selected = selection(population, fitness_results, fitness_mean)
        print('Number of selected individuals:', len(selected))

        # Finish the loop if there is no individual above average
        if not selected:
            break

        # Reproduce the selected individuals
        print('Generating new population...')
        new_population = reproduce(n, selected)

        # Mutate (if possible) every individual
        print('Applying necessary mutations...')
        mutated_population = []
        for individual in new_population:
            mutated = mutation(individual, mutation_prob)
            mutated_population.append(mutated)
        population = mutated_population

    print("=======================================")
    print('FINAL FITNESS MEAN:', fitness_mean)
    print('OPTIMUM VALUES FROM FINAL POPULATION:', population[0])
    # return the first individual of the optimum population
    return population[0]
