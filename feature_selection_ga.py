from sklearn.model_selection import train_test_split
import random
import classifiers

# Function to get the ga individual
# It consists in and array of 0's and 1's corresponding to every feature column
def generate_individual(df_train):
    columns = []
    for col in df_train.columns.values:
        # Generate random value to determinate if a feature should be included
        columns.append(random.randint(0, 1))
    # Return the array of 0's and 1's
    return columns

# Function to generate population given a number of individuals
def generate_population(n, df_train):
    population = []
    i = 0
    while i < n:
        population.append(generate_individual(df_train))
        i += 1
    return population

# Fitness function. Our fitness is the ROC value provided by the generated model
def fitness(
        individual,
        df_train,
        y,
):
    excluded = []
    for index, col in enumerate(df_train.columns.values):
        # Only exclude the columns which index correspond to a negative value in the individual
        if individual[index] == 0:
            excluded.append(col)
    # Consider every column except for the excluded ones as features
    X = df_train.drop(excluded, axis = 1)
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    rows, cols = X_test.shape
    print("Calculating fitness for {} features".format(cols))
    roc, clf = classifiers.run(
        X_train,
        y_train,
        X_test,
        y_test,
        # Random Forest parameters
        15,
        1,
        1074,
        # Ada Boost parameters
        72,
        0.6746530840498172,
        2940,
        # Gradient Tree Boosting parameters
        1,
        0.18546035948574746,
        139,
        4,
        9090,
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
        # Randomly find start and end index to flip the bit
        rand1 = random.randint(0, len(individual))
        rand2 = random.randint(0, len(individual))
        while rand1 == rand2:
            rand2 = random.randint(3, 5)
        randoms = [rand1, rand2]
        start = min(randoms)
        end = max(randoms)

        # Flip bits in the given range
        for i in range(start, end):
            if individual[i] == 0:
                individual[i] = 1
            else:
                individual[i] = 0
    return mutated

def run(
        df_train,
        y,
        n,
        max_iter,
        mutation_prob
):
    # Generate population with the provided size
    population = generate_population(n, df_train)
    for i in range(0, max_iter):
        print("(Population #{})=======================================".format(i + 1))
        # print('Population size:', len(population))
        # Calculate fitness for every individual
        fitness_results = []
        for individual in population:
            fitness_results.append(
                fitness(individual, df_train, y)
            )
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
