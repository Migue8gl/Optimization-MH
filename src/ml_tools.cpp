#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include "ml_tools.h"
#include "tools_helper.h"
#include <chrono>
#include <iomanip>
#include <thread>
#include "seed.h"
#include <future>
#include <queue>
#include <functional>

char MLTools::kNNClassifier(const Data &data, const std::vector<double> &element, const std::vector<double> &weigths, const unsigned int &k)
{
    {
        unsigned int dataSize = data.size();
        std::vector<std::vector<double>> dataMatrix = data.getData();
        std::vector<char> dataLabels = data.getLabels();

        if (k <= 0 || k > dataSize)
        {
            throw std::invalid_argument("Invalid value of k");
        }

        std::vector<std::pair<double, char>> distancesAndClasses;

        for (unsigned i = 0; i < dataSize; ++i)
        {
            if (element != dataMatrix[i])
            { // Skip the same element
                double distance = MLTools::computeEuclideanDistance(element, dataMatrix[i], weigths);
                distancesAndClasses.emplace_back(distance, dataLabels[i]);
            }
        }

        // Sort distancesAndClasses by distance (ascending order)
        std::sort(distancesAndClasses.begin(), distancesAndClasses.end(), [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        // Count the occurrences of each class among the k nearest neighbors
        std::unordered_map<char, int> classCounts;
        for (unsigned int i = 0; i < k; ++i)
        {
            char cls = distancesAndClasses[i].second;
            classCounts[cls]++;
        }

        // Find the class with the highest count (mode)
        char predictedClass = '\0';
        int maxCount = -1;
        for (const auto &pair : classCounts)
        {
            if (pair.second > maxCount)
            {
                predictedClass = pair.first;
                maxCount = pair.second;
            }
        }

        return predictedClass;
    }
}

std::vector<double> MLTools::knn(const Data &data, std::vector<double> &weights, std::vector<std::string> &hyperParams)
{
    // Weigth vector to one, knn does not modify weights
    return std::vector<double>(data.getData()[0].size(), 1.0);
}

void MLTools::kCrossValidation(const Data &data, const Optimizer &optimizer, const int numberPartitions, std::vector<std::string> &hyperParams)
{
    const double alpha = 0.5;
    double TS_average = 0, TR_average = 0, A_average = 0;
    bool showPartitions = true;

    auto overallStartTime = std::chrono::high_resolution_clock::now();

    std::cout << "\n***** (CROSS VALIDATION K = " << numberPartitions << ") *****\n"
              << std::endl;

    std::vector<Data> partitions = data.createPartitions(numberPartitions);

    for (unsigned int partitionIndex = 0; partitionIndex < partitions.size(); partitionIndex++)
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        const Data &trainingData = partitions[partitionIndex];
        Data testData;
        unsigned int reductionCount = 0;

        for (unsigned int i = 0; i < partitions.size(); i++)
        {
            if (i != partitionIndex)
            {
                const std::vector<std::vector<double>> &otherData = partitions[i].getData();
                const std::vector<char> &otherLabels = partitions[i].getLabels();
                for (unsigned int j = 0; j < otherData.size(); j++)
                {
                    testData.addDataPoint(otherData[j], otherLabels[j]);
                }
            }
        }

        // Test
        std::vector<double> w(data.getData()[0].size());
        std::generate(w.begin(), w.end(), [&]()
                      { return ToolsHelper::generateUniformRandomNumberDouble(0.0, 1.0, Seed::getInstance().getSeed()); });

        std::vector<double> weights = optimizer(trainingData, w, hyperParams);

        for (double &wi : weights)
        {
            if (wi < 0.1)
            {
                reductionCount++;
                wi = 0.0;
            }
        }

        double classificationAccuracy = MLTools::computeAccuracy(testData, weights);
        double reductionRate = static_cast<double>(reductionCount) / static_cast<double>(weights.size());
        double fitness = alpha * classificationAccuracy + (1.0 - alpha) * reductionRate;

        TS_average += classificationAccuracy;
        TR_average += reductionRate;
        A_average += fitness;

        // Calculate progress
        float progress = static_cast<float>(partitionIndex + 1) / partitions.size();
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> executionTime = endTime - startTime;

        // Update the progress bar
        ToolsHelper::progress_bar(progress);
        if (showPartitions)
        {
            std::cout << "\n\n[PART " << partitionIndex + 1 << "] | Classification Rate: " << classificationAccuracy << std::endl;
            std::cout << "[PART " << partitionIndex + 1 << "] | Reduction Rate: " << reductionRate << std::endl;
            std::cout << "[PART " << partitionIndex + 1 << "] | Fitness: " << fitness << std::endl;
            std::cout << "[PART " << partitionIndex + 1 << "] | Execution Time: " << std::fixed << std::setprecision(2) << executionTime.count() << " ms\n\n";
            std::cout << "--------------------------------------\n"
                      << std::endl;
        }
    }

    auto overallEndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> totalTime = overallEndTime - overallStartTime;

    if (!showPartitions)
        std::cout << "\n\n";
    std::cout << "***** (FINAL RESULTS) *****\n"
              << std::endl;
    std::cout << "Average Classification Rate: " << TS_average / partitions.size() << std::endl;
    std::cout << "Average Reduction Rate: " << TR_average / partitions.size() << std::endl;
    std::cout << "Average Fitness: " << A_average / partitions.size() << std::endl;
    std::cout << "Total Execution Time: " << std::fixed << std::setprecision(2) << totalTime.count() << " ms\n\n";
}

std::vector<double> MLTools::localSearch(const Data &data, std::vector<double> &weights, std::vector<std::string> &hyperParams)
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
        variance = std::stod(hyperParams[0]);
        alpha = std::stod(hyperParams[1]);
        mean = std::stod(hyperParams[2]);
        maxIter = std::stoi(hyperParams[3]);
        maxNeighbour = std::stoi(hyperParams[4]);
    }

    if (maxNeighbour == 0)
    {
        maxNeighbour = data.getData()[0].size() * 2;
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

std::vector<double> MLTools::localSearchStrong(const Data &data, std::vector<double> &weights, std::vector<std::string> &hyperParams)
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
        variance = std::stod(hyperParams[0]);
        alpha = std::stod(hyperParams[1]);
        mean = std::stod(hyperParams[2]);
        maxIter = std::stoi(hyperParams[3]);
        maxNeighbour = std::stoi(hyperParams[4]);
    }

    if (maxNeighbour == 0)
    {
        maxNeighbour = data.getData()[0].size() * 10;
    }

    unsigned int counter = 0;
    unsigned int neighbourCount = 0;
    double maxFunctionValue = -std::numeric_limits<double>::infinity();
    std::vector<double> w = weights;

    std::queue<double> wAux;
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

double MLTools::computeFitness(const Data &data, std::vector<double> &weights, const double &alpha)
{
    double classificationRate = 0.0;
    double reductionRate = 0.0;
    double reductionCount = 0.0;
    double fitness;

    for (unsigned int i = 0; i < weights.size(); ++i)
    {
        if (weights[i] < 0.1)
        {
            reductionCount += 1.0;
            weights[i] = 0.0;
        }
    }
    classificationRate = MLTools::computeAccuracy(data, weights);
    reductionRate = reductionCount / static_cast<double>(weights.size());
    fitness = reductionRate * alpha + classificationRate * (1 - alpha);

    return fitness;
}

std::vector<double> MLTools::computePopulationFitness(const Data &data, std::vector<std::vector<double>> &populationWeights, const double &alpha)
{
    std::vector<double> fitness(populationWeights.size(), 0.0);

    for (unsigned int i = 0; i < populationWeights.size(); ++i)
    {
        double classificationRate = 0.0;
        double reductionRate = 0.0;
        double reductionCount = 0.0;

        for (unsigned int j = 0; j < populationWeights[i].size(); ++j)
        {
            if (populationWeights[i][j] < 0.1)
            {
                reductionCount += 1.0;
                populationWeights[i][j] = 0.0;
            }
        }

        classificationRate = MLTools::computeAccuracy(data, populationWeights[i]);
        reductionRate = reductionCount / static_cast<double>(populationWeights[i].size());

        fitness[i] = reductionRate * alpha + classificationRate * (1 - alpha);
    }

    return fitness;
}

double MLTools::computeEuclideanDistance(const std::vector<double> &point1, const std::vector<double> &point2, const std::vector<double> &weights)
{
    if (point1.size() != point2.size() || point1.size() != weights.size())
    {
        throw std::invalid_argument("Vector dimensions and weights must match.");
    }

    double sum = 0.0;
    for (unsigned int i = 0; i < point1.size(); ++i)
    {
        double diff = point1[i] - point2[i];
        sum += weights[i] * (diff * diff);
    }
    return sqrt(sum);
}

double MLTools::computeAccuracy(const Data &sample, const std::vector<double> &weights)
{
    double correctlyClassifiedInstances = 0.0;
    const std::vector<std::vector<double>> &samples = sample.getData();
    const std::vector<char> &classes = sample.getLabels();
    unsigned int totalInstances = sample.size();

    for (unsigned int i = 0; i < totalInstances; ++i)
    {
        char predictedClass = MLTools::kNNClassifier(sample, samples[i], weights);

        if (predictedClass == classes[i])
            correctlyClassifiedInstances += 1.0;
    }

    return correctlyClassifiedInstances / static_cast<double>(totalInstances);
}

std::vector<double> MLTools::levyFlight(const std::vector<double> &butterfly, double alpha)
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
    for (size_t i = 0; i < butterfly.size(); ++i)
    {
        newPosition[i] = butterfly[i] + stepSize * tan(direction);
    }

    return newPosition;
}

void MLTools::migration(std::vector<std::vector<double>> &subpob1, std::vector<std::vector<double>> &subpob2, double period, double p)
{
    for (unsigned int i = 0; i < subpob1.size(); ++i)
    {
        for (unsigned int kIndex = 0; kIndex < subpob1[i].size(); ++kIndex)
        {
            double n = ToolsHelper::generateUniformRandomNumberDouble(0, 1, Seed::getInstance().getSeed()) * period;
            if (n <= p)
            {
                int index = ToolsHelper::generateUniformRandomNumberInteger(0, subpob1.size() - 1, Seed::getInstance().getSeed());
                subpob1[i][kIndex] = subpob1[index][kIndex];
            }
            else
            {
                int index = ToolsHelper::generateUniformRandomNumberInteger(0, subpob2.size() - 1, Seed::getInstance().getSeed());
                subpob1[i][kIndex] = subpob2[index][kIndex];
            }
        }
    }
}

void MLTools::adjust(std::vector<std::vector<double>> &subpob2, const std::vector<double> &bestButterfly, double p, double BAR, double alpha)
{
    for (unsigned int i = 0; i < subpob2.size(); ++i)
    {
        for (unsigned int kIndex = 0; kIndex < subpob2[i].size(); ++kIndex)
        {
            double n = ToolsHelper::generateUniformRandomNumberDouble(0, 1, Seed::getInstance().getSeed());

            if (n <= p)
            {
                subpob2[i][kIndex] = bestButterfly[kIndex];
            }
            else
            {
                int index = ToolsHelper::generateUniformRandomNumberInteger(0, subpob2.size() - 1, Seed::getInstance().getSeed());
                subpob2[i][kIndex] = subpob2[index][kIndex];

                if (n > BAR)
                {
                    std::vector<double> dx = levyFlight(subpob2[i], alpha);
                    subpob2[i][kIndex] += alpha * (dx[kIndex] - 0.5);
                    subpob2[i][kIndex] = std::max(0.0, std::min(1.0, subpob2[i][kIndex]));
                }
            }
        }
    }
}

std::vector<std::vector<double>> MLTools::elitism(const Data &data, std::vector<std::vector<double>> &np, unsigned int numElite)
{
    std::vector<double> fitnessPopulation = MLTools::computePopulationFitness(data, np, 0.5);
    std::vector<std::vector<double>> eliteButterflies;

    // Encuentra los índices de las mariposas élite
    std::vector<unsigned int> eliteIndices;
    for (unsigned int i = 0; i < numElite; ++i)
    {
        unsigned int bestSolutionIndex = std::distance(fitnessPopulation.begin(), std::max_element(fitnessPopulation.begin(), fitnessPopulation.end()));
        eliteIndices.push_back(bestSolutionIndex);
        fitnessPopulation[bestSolutionIndex] = -std::numeric_limits<double>::max();
    }

    // Copia las mariposas élite
    for (unsigned int index : eliteIndices)
    {
        eliteButterflies.push_back(np[index]);
    }

    // Reemplaza las peores mariposas por las mariposas élite
    for (unsigned int i = 0; i < numElite; ++i)
    {
        unsigned int worstIndex = std::distance(fitnessPopulation.begin(), std::min_element(fitnessPopulation.begin(), fitnessPopulation.end()));
        np[worstIndex] = eliteButterflies[i];
        fitnessPopulation[worstIndex] = -std::numeric_limits<double>::max();
    }

    return np;
}

std::vector<double> MLTools::simulatedAnnealing(const Data &data, std::vector<double> &weights, std::vector<std::string> &hyperParams)
{
    // Initialize weight vector using a random generator
    std::vector<double> currentWeights = weights;
    std::vector<double> bestWeights(data.getData()[0].size());

    // Initialize parameters
    int maxNeighbours = 10 * currentWeights.size();
    double alpha = 0.5;
    int maxSuccess = 0.1 * maxNeighbours;
    int M = 15000 / maxNeighbours;
    double mu = 0.3;
    double k = 1.0;
    int iter = 0;
    double mean = 0.0;
    double variance = 0.3;

    double currentEval = MLTools::computeFitness(data, weights, alpha);
    double bestEval = currentEval;
    bestWeights.swap(currentWeights);

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

    while (iter < M && currentTemperature > finalTemperature)
    {
        int successCount = 0;
        int neighboursCount = 0;

        while (successCount < maxSuccess && neighboursCount < maxNeighbours)
        {
            std::vector<double> newWeights = currentWeights;

            // Start mutations
            int index = ToolsHelper::generateUniformRandomNumberInteger(0, newWeights.size() - 1, Seed::getInstance().getSeed());
            newWeights[index] += ToolsHelper::generateNormalRandomNumber(mean, std::sqrt(variance), Seed::getInstance().getSeed());

            newWeights[index] = std::max(0.0, std::min(1.0, newWeights[index]));

            // Evaluate the new solution
            double newEval = MLTools::computeFitness(data, newWeights, alpha);
            double diff = newEval - currentEval;

            if (diff > 0.0 || ToolsHelper::generateUniformRandomNumberDouble(0, 1, Seed::getInstance().getSeed()) <= (exp(diff) / (k * currentTemperature)))
            {
                currentEval = newEval;
                currentWeights.swap(newWeights);
                successCount += 1;

                if (currentEval > bestEval)
                {
                    bestEval = currentEval;
                    bestWeights = currentWeights;
                }
            }

            neighboursCount += 1;
        }

        currentTemperature = currentTemperature / (1 + (beta * currentTemperature));
        iter += 1;
    }

    return bestWeights;
}

std::vector<double> MLTools::mbo(const Data &data, std::vector<double> &weights, std::vector<std::string> &hyperParams)
{
    // Initialize population of butterflies NP
    std::vector<std::vector<double>> np(data.size(), std::vector<double>(data.getData()[0].size()));

    for (std::vector<double> &butterfly : np)
    {
        std::generate(butterfly.begin(), butterfly.end(), [&]()
                      { return ToolsHelper::generateUniformRandomNumberDouble(0.0, 1.0, Seed::getInstance().getSeed()); });
    }

    int t = 1;
    double delta = 0.08;
    double error = 1.0;
    const int maxGen = 10;
    const double BAR = 5.0 / 12.0;
    const double p = 5.0 / 12.0;
    const double periodo = 1.2;
    const double smax = 1.0;
    double alpha = 0.0;
    int np1Size = static_cast<int>(p * np.size());
    std::vector<double> fitnessPopulation;
    std::vector<std::string> hyperLSParams;

    fitnessPopulation = MLTools::computePopulationFitness(data, np, 0.5);

    while (t < maxGen && error >= delta)
    {
        double bestFitness = fitnessPopulation[0];
        unsigned int bestSolutionIndex = 0;

        // Find the best butterfly in the np population
        for (unsigned int i = 0; i < fitnessPopulation.size(); ++i)
        {
            if (fitnessPopulation[i] > bestFitness)
            {
                bestFitness = fitnessPopulation[i];
                error = 1 - bestFitness;
                bestSolutionIndex = i;
            }
        }

        // Keep the best butterfly
        std::vector<double> bestButterfly = np[bestSolutionIndex];

        // Divide the remaining population into two subpopulations
        std::vector<std::vector<double>>
            np1(np.begin(), np.begin() + np1Size);
        std::vector<std::vector<double>> np2(np.begin() + np1Size, np.end());

        alpha = smax / (t * t);

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
    for (unsigned int i = 0; i < fitnessPopulation.size(); ++i)
    {
        if (fitnessPopulation[i] > bestEval)
        {
            bestEval = fitnessPopulation[i];
            bestSolutionIndex = i;
        }
    }

    np[bestSolutionIndex] = MLTools::localSearch(data, np[bestSolutionIndex], hyperLSParams);

    return np[bestSolutionIndex];
}
