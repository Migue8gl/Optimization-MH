#include "optimizers.h"
#include "ml_tools.h"
#include "tools_helper.h"

std::vector<double> Optimizers::knn(const Data &data, std::vector<double> &weights, std::map<std::string, std::string> &hyperParams)
{
    // Weigth vector to one, knn does not modify weights
    return std::vector<double>(data.parameterSize(), 1.0);
}

std::vector<double> Optimizers::localSearch(const Data &data, std::vector<double> &weights, std::map<std::string, std::string> &hyperParams)
{
    // Default values for parameters
    double variance = 0.3;
    double alpha = 0.5;
    double mean = 0.0;
    unsigned int maxIter = 15000;
    unsigned int maxNeighbour = 0;

    if (!hyperParams.empty())
    {
        if (hyperParams.size() != 5 && hyperParams.empty())
        {
            throw std::invalid_argument("Error: hyperParams must strictly contain 5 parameters.");
        }
        // Update parameters from hyperParams
        if (hyperParams.find("variance") != hyperParams.end())
        {
            variance = std::stod(hyperParams["variance"]);
        }

        if (hyperParams.find("alpha") != hyperParams.end())
        {
            alpha = std::stod(hyperParams["alpha"]);
        }

        if (hyperParams.find("mean") != hyperParams.end())
        {
            mean = std::stod(hyperParams["mean"]);
        }

        if (hyperParams.find("maxIter") != hyperParams.end())
        {
            maxIter = std::stoi(hyperParams["maxIter"]);
        }

        if (hyperParams.find("maxNeighbour") != hyperParams.end())
        {
            maxNeighbour = std::stoi(hyperParams["maxNeighbour"]);
        }
    }

    if (maxNeighbour == 0)
    {
        maxNeighbour = data.parameterSize() * 2;
    }

    unsigned int counter = 0;
    unsigned int neighbourCount = 0;
    double maxFunctionValue = -std::numeric_limits<double>::infinity();
    std::vector<double> w = weights;

    double wAux;
    double objetiveFunction = MLTools::computeFitness(data, w, alpha);

    while (neighbourCount < maxNeighbour && counter < maxIter)
    {
        unsigned int randIndex = ToolsHelper::generateUniformRandomNumberInteger(0, w.size() - 1, Seed::getInstance().getSeed());

        // Store the original value of w[randIndex]
        wAux = w[randIndex];

        // Mutation
        w[randIndex] += ToolsHelper::generateNormalRandomNumber(mean, std::sqrt(variance), Seed::getInstance().getSeed());

        // Ensure w[i] is within the bounds [0, 1]
        w[randIndex] = std::max(0.0, std::min(1.0, w[randIndex]));

        objetiveFunction = MLTools::computeFitness(data, w, alpha);
        if (objetiveFunction > maxFunctionValue)
        {
            maxFunctionValue = objetiveFunction;
            neighbourCount = 0;
        }
        else
        {
            w[randIndex] = wAux;
            neighbourCount++;
        }

        counter++;
    }

    return w;
}

std::vector<double> Optimizers::localSearchStrong(const Data &data, std::vector<double> &weights, std::map<std::string, std::string> &hyperParams)
{
    // Default values for parameters
    double variance = 0.4; // Increased variance for LocalSearchStrong
    double alpha = 0.5;
    double mean = 0.0;
    unsigned int maxIter = 30000;
    unsigned int maxNeighbour = 0;
    unsigned int maxNumAdjust = 5;

    if (!hyperParams.empty())
    {
        if (hyperParams.size() != 5 && hyperParams.empty())
        {
            throw std::invalid_argument("Error: hyperParams must strictly contain 5 parameters.");
        }
        // Update parameters from hyperParams
        if (hyperParams.find("variance") != hyperParams.end())
        {
            variance = std::stod(hyperParams["variance"]);
        }

        if (hyperParams.find("alpha") != hyperParams.end())
        {
            alpha = std::stod(hyperParams["alpha"]);
        }

        if (hyperParams.find("mean") != hyperParams.end())
        {
            mean = std::stod(hyperParams["mean"]);
        }

        if (hyperParams.find("maxIter") != hyperParams.end())
        {
            maxIter = std::stoi(hyperParams["maxIter"]);
        }

        if (hyperParams.find("maxNeighbour") != hyperParams.end())
        {
            maxNeighbour = std::stoi(hyperParams["maxNeighbour"]);
        }
    }

    if (maxNeighbour == 0)
    {
        maxNeighbour = data.parameterSize() * 10;
    }

    unsigned int counter = 0;
    unsigned int neighbourCount = 0;
    double maxFunctionValue = -std::numeric_limits<double>::infinity();
    std::vector<double> w = weights;

    double objetiveFunction = MLTools::computeFitness(data, w, alpha);
    std::vector<std::pair<unsigned int, double>> originalValues;

    while (neighbourCount < maxNeighbour && counter < maxIter)
    {
        unsigned int numAdjustments = ToolsHelper::generateUniformRandomNumberInteger(1, maxNumAdjust, Seed::getInstance().getSeed());

        for (unsigned int i = 0; i < numAdjustments; ++i)
        {
            unsigned int randIndex = ToolsHelper::generateUniformRandomNumberInteger(0, w.size() - 1, Seed::getInstance().getSeed());
            originalValues.emplace_back(randIndex, w[randIndex]);
            // Mutation
            w[randIndex] += ToolsHelper::generateNormalRandomNumber(mean, std::sqrt(variance), Seed::getInstance().getSeed());

            // Ensure w[i] is within the bounds [0, 1]
            w[randIndex] = std::max(0.0, std::min(1.0, w[randIndex]));
        }

        objetiveFunction = MLTools::computeFitness(data, w, alpha);
        if (objetiveFunction > maxFunctionValue)
        {
            maxFunctionValue = objetiveFunction;
            neighbourCount = 0;
        }
        else
        {
            for (const auto &pair : originalValues)
            {
                w[pair.first] = pair.second;
            }
            neighbourCount += numAdjustments;
        }

        counter++;
    }

    return w;
}

std::vector<double> Optimizers::levyFlight(const std::vector<double> &butterfly, double alpha)
{
    // Parameters for Levy flight distribution
    const double scale = 0.01;

    std::mt19937 eng(Seed::getInstance().getSeed());

    // Levy flight formula
    std::gamma_distribution<double> dist(alpha, scale);

    double stepSize = dist(eng);
    double direction = ToolsHelper::generateUniformRandomNumberDouble(0.0, 2.0 * M_PI, Seed::getInstance().getSeed());

    // Calculate the new position based on the butterfly and the step size
    std::vector<double> newPosition(butterfly.size());

    std::vector<double>::const_iterator butterflyIter = butterfly.begin();
    std::vector<double>::iterator newPositionIter = newPosition.begin();

    while (butterflyIter != butterfly.end())
    {
        *newPositionIter = *butterflyIter + stepSize * tan(direction);
        ++butterflyIter;
        ++newPositionIter;
    }

    return newPosition;
}

void Optimizers::migration(std::vector<std::vector<double>> &subpob1, std::vector<std::vector<double>> &subpob2, double period, double p)
{
    for (std::vector<std::vector<double>>::iterator it1 = subpob1.begin(); it1 != subpob1.end(); ++it1)
    {
        for (std::vector<double>::iterator it2 = it1->begin(); it2 != it1->end(); ++it2)
        {
            double n = ToolsHelper::generateUniformRandomNumberDouble(0, 1, Seed::getInstance().getSeed()) * period;
            if (n <= p)
            {
                int index = ToolsHelper::generateUniformRandomNumberInteger(0, subpob1.size() - 1, Seed::getInstance().getSeed());
                *it2 = subpob1[index][it2 - it1->begin()];
            }
            else
            {
                int index = ToolsHelper::generateUniformRandomNumberInteger(0, subpob2.size() - 1, Seed::getInstance().getSeed());
                *it2 = subpob2[index][it2 - it1->begin()];
            }
        }
    }
}

void Optimizers::adjust(std::vector<std::vector<double>> &subpob2, const std::vector<double> &bestButterfly, double p, double BAR, double alpha)
{
    for (std::vector<std::vector<double>>::iterator it1 = subpob2.begin(); it1 != subpob2.end(); ++it1)
    {
        for (std::vector<double>::iterator it2 = it1->begin(); it2 != it1->end(); ++it2)
        {
            double n = ToolsHelper::generateUniformRandomNumberDouble(0, 1, Seed::getInstance().getSeed());

            if (n <= p)
            {
                *it2 = bestButterfly[it2 - it1->begin()];
            }
            else
            {
                int index = ToolsHelper::generateUniformRandomNumberInteger(0, subpob2.size() - 1, Seed::getInstance().getSeed());
                *it2 = subpob2[index][it2 - it1->begin()];

                if (n > BAR)
                {
                    std::vector<double> dx = levyFlight(*it1, alpha);
                    *it2 += alpha * (dx[it2 - it1->begin()] - 0.5);
                    // *it2 = std::max(0.0, std::min(1.0, *it2));
                }
            }
        }
    }
}

std::vector<std::vector<double>> Optimizers::elitism(const Data &data, std::vector<std::vector<double>> &np, unsigned int numElite)
{
    std::vector<double> fitnessPopulation = MLTools::computePopulationFitness(data, np, 0.5);
    std::vector<std::vector<double>> eliteButterflies;

    // Find the indices of elite butterflies
    std::vector<unsigned int> eliteIndices;
    for (unsigned int i = 0; i < numElite; ++i)
    {
        std::vector<double>::iterator bestSolutionIter = std::max_element(fitnessPopulation.begin(), fitnessPopulation.end());
        unsigned int bestSolutionIndex = bestSolutionIter - fitnessPopulation.begin();
        eliteIndices.push_back(bestSolutionIndex);
    }

    // Copy the elite butterflies
    for (unsigned int index : eliteIndices)
    {
        eliteButterflies.push_back(np[index]);
    }

    // Replace the worst butterflies with the elite butterflies
    for (unsigned int i = 0; i < numElite; ++i)
    {
        std::vector<double>::iterator worstSolutionIter = std::min_element(fitnessPopulation.begin(), fitnessPopulation.end());
        unsigned int worstIndex = worstSolutionIter - fitnessPopulation.begin();
        np[worstIndex] = eliteButterflies[i];
    }

    return np;
}

std::vector<double> Optimizers::mbo(const Data &data, std::vector<double> &weights, std::map<std::string, std::string> &hyperParams)
{
    // Initialize population of butterflies NP
    std::vector<std::vector<double>> np(data.size(), std::vector<double>(data.parameterSize()));

    for (std::vector<double> &butterfly : np)
    {
        std::generate(butterfly.begin(), butterfly.end(), [&]()
                      { return ToolsHelper::generateUniformRandomNumberDouble(0.0, 1.0, Seed::getInstance().getSeed()); });
    }

    int t = 1;
    double delta = 0.05;
    double error = 1.0;
    int optimizerOpt = 0;
    const int maxGen = 10;
    const double BAR = 5.0 / 12.0;
    const double p = 5.0 / 12.0;
    const double periodo = 1.2;
    const double smax = 1.0;
    double alpha = 0.0;
    int np1Size = static_cast<int>(p * np.size());
    std::vector<double> fitnessPopulation(data.parameterSize(), 0);
    std::map<std::string, std::string> hyperLSParams;

    fitnessPopulation = MLTools::computePopulationFitness(data, np, 0.5);

    while (t < maxGen && error > delta)
    {
        double bestFitness = fitnessPopulation[0];
        unsigned int bestSolutionIndex = 0;

        // Find the best butterfly in the np population
        std::vector<double>::iterator bestSolutionIter = fitnessPopulation.begin();

        for (bestSolutionIter = fitnessPopulation.begin(); bestSolutionIter != fitnessPopulation.end(); ++bestSolutionIter)
        {
            if (*bestSolutionIter > bestFitness)
            {
                bestFitness = *bestSolutionIter;
                error = 1 - bestFitness;
                bestSolutionIndex = bestSolutionIter - fitnessPopulation.begin();
            }
        }

        std::vector<double> bestButterfly;

        switch (optimizerOpt)
        {
        case 0:
            // Keep the best butterfly
            bestButterfly = np[bestSolutionIndex];
            break;
        case 1:
            bestButterfly = Optimizers::localSearch(data, np[bestSolutionIndex], hyperLSParams);
            break;
        case 2:
            bestButterfly = Optimizers::simulatedAnnealing(data, np[bestSolutionIndex], hyperLSParams);
            break;
        }

        // Divide the remaining population into two subpopulations
        std::vector<std::vector<double>>
            np1(np.begin(), np.begin() + np1Size);
        std::vector<std::vector<double>> np2(np.begin() + np1Size, np.end());

        alpha = smax / pow(2, t);

        // Start mig_thread asynchronously
        std::future<void> mig_future = std::async(std::launch::async, migration, std::ref(np1), std::ref(np2), periodo, p);

        // Start aju_thread asynchronously
        std::future<void> aju_future = std::async(std::launch::async, adjust, std::ref(np2), bestButterfly, p, BAR, alpha);

        // Wait for both threads to complete
        mig_future.get();
        aju_future.get();

        // Copy the data from np1 and np2 back to the np array
        np = np1;
        np.insert(np.end(), np2.begin(), np2.end());

        int numElite = 3;
        np = elitism(data, np, numElite);

        std::mt19937 gen(Seed::getInstance().getSeed());
        std::shuffle(np.begin(), np.end(), gen);
        fitnessPopulation = MLTools::computePopulationFitness(data, np, 0.5);
        t++;
    }

    double bestEval = fitnessPopulation[0];
    unsigned int bestSolutionIndex = 0;

    std::vector<double>::iterator bestEvalIter = fitnessPopulation.begin();
    std::vector<double>::iterator fitnessIter = fitnessPopulation.begin();

    for (; fitnessIter != fitnessPopulation.end(); ++fitnessIter)
    {
        if (*fitnessIter > bestEval)
        {
            bestEval = *fitnessIter;
            bestSolutionIndex = fitnessIter - bestEvalIter;
        }
    }

    switch (optimizerOpt)
    {
    case 1:
        np[bestSolutionIndex] = Optimizers::localSearch(data, np[bestSolutionIndex], hyperLSParams);
        break;
    case 2:
        np[bestSolutionIndex] = Optimizers::simulatedAnnealing(data, np[bestSolutionIndex], hyperLSParams);
        break;
    }

    return np[bestSolutionIndex];
}

std::vector<double> Optimizers::simulatedAnnealing(const Data &data, std::vector<double> &weights, std::map<std::string, std::string> &hyperParams)
{
    // Initialize weight vector using a random generator
    std::vector<double> currentWeights = weights;
    std::vector<double> bestWeights = weights;
    std::vector<double> newWeights = weights;

    // Initialize parameters
    int maxNeighbours = 5 * currentWeights.size();
    double alpha = 0.5;
    int maxSuccess = 0.1 * maxNeighbours;
    int M = 15000 / maxNeighbours;
    double mu = 0.3;
    double k = 1.0;
    int iter = 0;
    double mean = 0.0;
    int successCount = 1;
    double variance = 0.3;

    double currentEval = MLTools::computeFitness(data, currentWeights, alpha);
    double bestEval = currentEval;

    // Initial temperature
    double initialTemperature = (mu * currentEval) / -log(mu);
    // Final temperature
    double finalTemperature = 0.001;

    // Ensure final temperature is always smaller than initial temperature
    while (finalTemperature > initialTemperature)
    {
        finalTemperature *= 0.001;
    }

    double beta = (initialTemperature - finalTemperature) / (M * finalTemperature * initialTemperature);

    // Current temperature
    double currentTemperature = initialTemperature;

    while (iter <= M && successCount != 0)
    {
        successCount = 0;
        int neighboursCount = 0;

        while (successCount <= maxSuccess && neighboursCount <= maxNeighbours)
        {
            // Start mutations
            int index = ToolsHelper::generateUniformRandomNumberInteger(0, newWeights.size() - 1, Seed::getInstance().getSeed());
            newWeights[index] += ToolsHelper::generateNormalRandomNumber(mean, std::sqrt(variance), Seed::getInstance().getSeed());
            neighboursCount++;

            newWeights[index] = std::max(0.0, std::min(1.0, newWeights[index]));

            // Evaluate the new solution
            double newEval = MLTools::computeFitness(data, newWeights, alpha);
            double diff = newEval - currentEval;

            if (diff > 0.0 || ToolsHelper::generateUniformRandomNumberDouble(0, 1, Seed::getInstance().getSeed()) > exp(-diff / (k * currentTemperature)))
            {
                currentEval = newEval;
                currentWeights[index] = newWeights[index];
                successCount++;

                if (currentEval > bestEval)
                {
                    bestEval = currentEval;
                    bestWeights = currentWeights;
                }
            }
            currentWeights[index] = newWeights[index];
            currentEval = newEval;
        }

        currentTemperature = currentTemperature / (1 + beta * currentTemperature);
        iter++;
    }

    return bestWeights;
}