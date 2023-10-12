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

char MLTools::KNNClassifier(const Data &data, const std::vector<double> &element, const std::vector<double> &weigths, const unsigned int &k)
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

std::vector<double> MLTools::KNN(const Data &data, std::vector<double> &weights)
{
    // Weigth vector to one, knn does not modify weights
    return std::vector<double>(data.getData()[0].size(), 1.0);
}

void MLTools::kCrossValidation(const Data &data, const MLTools::Optimizer &optimizer, const int numberPartitions, const std::string &option)
{
    const double alpha = 0.5;
    double TS_average = 0, TR_average = 0, A_average = 0;

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
                      { return ToolsHelper::generateNormalRandomNumber(0.0, std::sqrt(0.3), Seed::getInstance().getSeed()); });
        std::vector<double> weights = optimizer(trainingData, w);

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

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> executionTime = endTime - startTime;

        std::cout << "[PART " << partitionIndex + 1 << "] | Classification Rate: " << classificationAccuracy << std::endl;
        std::cout << "[PART " << partitionIndex + 1 << "] | Reduction Rate: " << reductionRate << std::endl;
        std::cout << "[PART " << partitionIndex + 1 << "] | Fitness: " << fitness << std::endl;
        std::cout << "[PART " << partitionIndex + 1 << "] | Execution Time: " << std::fixed << std::setprecision(2) << executionTime.count() << " ms\n\n";
        std::cout << "--------------------------------------\n"
                  << std::endl;
    }

    auto overallEndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> totalTime = overallEndTime - overallStartTime;

    std::cout << "***** (FINAL RESULTS) *****\n"
              << std::endl;
    std::cout << "Average Classification Rate: " << TS_average / partitions.size() << std::endl;
    std::cout << "Average Reduction Rate: " << TR_average / partitions.size() << std::endl;
    std::cout << "Average Fitness: " << A_average / partitions.size() << std::endl;
    std::cout << "Total Execution Time: " << std::fixed << std::setprecision(2) << totalTime.count() << " ms\n\n";
}

std::vector<double> MLTools::localSearch(const Data &data, std::vector<double> &weights)
{
    // Variables used
    const double variance = 0.3;
    const double alpha = 0.5;
    const double mean = 0.0;
    int maxIter = 15000;
    int maxNeighbour = 0;

    if (maxNeighbour == 0)
    {
        maxNeighbour = data.getData()[0].size() * 2;
    }

    int counter = 0, neighbourCount = 0;
    double maxFunctionValue = -std::numeric_limits<double>::infinity();
    /*std::vector<double> w(data.getData()[0].size());

    // Initialize w with random normal values in one line
    std::generate(w.begin(), w.end(), [&]()
                  { return ToolsHelper::generateNormalRandomNumber(mean, std::sqrt(variance), Seed::getInstance().getSeed()); });*/
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
        if (w[randIndex] < 0.1 && w[randIndex] > -0.1)
        {
            w[randIndex] = 0.0; // Modify the weights directly in the input vector
        }

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
            // Modify the weights directly in the input vector
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
        std::vector<double> weights = populationWeights[i];

        for (unsigned int j = 0; j < weights.size(); ++j)
        {
            if (weights[j] < 0.1)
            {
                reductionCount += 1.0;
                // Optionally modify the weights directly in the input vector
                weights[j] = 0.0;
            }
        }

        classificationRate = MLTools::computeAccuracy(data, weights);
        reductionRate = reductionCount / static_cast<double>(weights.size());

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
        char predictedClass = MLTools::KNNClassifier(sample, samples[i], weights);

        if (predictedClass == classes[i])
            correctlyClassifiedInstances += 1.0;
    }

    return correctlyClassifiedInstances / static_cast<double>(totalInstances);
}

double MLTools::levyFlight()
{
    // Parameters for Levy flight distribution
    const double alpha = 1.5; // You can adjust this value to control the Levy flight characteristics
    const double scale = 0.01;

    std::mt19937 eng(Seed::getInstance().getSeed());

    // Levy flight formula
    std::gamma_distribution<double> dist(alpha, scale);

    return dist(eng);
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
                    double dx = levyFlight(); // Implement this function
                    subpob2[i][kIndex] += alpha * (dx - 0.5);
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

std::vector<double> MLTools::mbo(const Data &data, std::vector<double> &weights)
{
    double variance = 0.3, mean = 0.0;

    // Initialize population of butterflies NP
    std::vector<std::vector<double>> np(data.size(), std::vector<double>(data.getData()[0].size()));

    for (std::vector<double> &butterfly : np)
    {
        std::generate(butterfly.begin(), butterfly.end(), [mean, variance]()
                      { return ToolsHelper::generateNormalRandomNumber(mean, std::sqrt(variance), Seed::getInstance().getSeed()); });
    }

    int t = 1;
    double delta = 0.1;
    double error = 1.0;
    const int maxGen = 30;
    const double BAR = 5.0 / 12.0;
    const double p = 5.0 / 12.0;
    const double periodo = 1.2;
    const double smax = 1.0;
    double alpha = 0.0;
    int np1Size = static_cast<int>(p * np.size());
    std::vector<double> fitnessPopulation;

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
        std::vector<double> bestButterfly = MLTools::localSearch(data, np[bestSolutionIndex]); // Improvement of best butterfly

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

    np[bestSolutionIndex] = MLTools::localSearch(data, np[bestSolutionIndex]); // Implement localSearch function

    return np[bestSolutionIndex];
}
