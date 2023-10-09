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
#include "seed.h"

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

        for (size_t i = 0; i < dataSize; ++i)
        {
            if (element != dataMatrix[i])
            { // Skip the same element
                double distance = ToolsHelper::computeEuclideanDistance(element, dataMatrix[i], weigths);
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

std::vector<double> MLTools::KNN(const Data &data, const std::string &opt)
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
                for (size_t j = 0; j < otherData.size(); j++)
                {
                    testData.addDataPoint(otherData[j], otherLabels[j]);
                }
            }
        }

        std::vector<double> weights = optimizer(trainingData, option);

        for (double &wi : weights)
        {
            if (wi < 0.1)
            {
                reductionCount++;
                wi = 0.0;
            }
        }

        double classificationAccuracy = ToolsHelper::computeAccuracy(testData, weights);
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

std::vector<double> MLTools::localSearch(const Data &data, const std::string &opt)
{
    const double variance = 0.3;
    const double alpha = 0.5;
    const double mean = 0.0;
    int reductionCount = 0;
    int maxIter = 15000;
    int maxNeighbour = 0;

    if (maxNeighbour == 0)
    {
        maxNeighbour = data.getData()[0].size() * 2;
    }

    int counter = 0, neighbourCount = 0;
    double maxFunctionValue = -std::numeric_limits<double>::infinity();
    std::vector<double> w(data.getData()[0].size());
    std::default_random_engine eng(Seed::getInstance().getSeed());
    std::normal_distribution<double> normalDist(mean, std::sqrt(variance));

    // Initialize w with random normal values in one line
    std::generate(w.begin(), w.end(), [&]()
                  { return normalDist(eng); });

    double wAux;
    double objetiveFunction = MLTools::computeFitness(data, w, alpha);

    while (neighbourCount < maxNeighbour && counter < maxIter)
    {
        std::vector<double> z(w.size());
        std::generate(z.begin(), z.end(), [&]()
                      { return normalDist(eng); });

        std::vector<double> originalW = w;

        for (size_t i = 0; i < w.size(); ++i)
        {
            // Store the original value of w[i]
            wAux = w[i];

            // Mutation
            w[i] += z[i];

            // Ensure w[i] is within the bounds [0, 1]
            w[i] = std::max(0.0, std::min(1.0, w[i]));

            if (w[i] < 0.1)
            {
                w[i] = 0;
                reductionCount += 1;
            }

            objetiveFunction = MLTools::computeFitness(data, w, alpha);
            if (objetiveFunction > maxFunctionValue)
            {
                maxFunctionValue = objetiveFunction;
                neighbourCount = 0;
            }
            else
            {
                w[i] = wAux;
                neighbourCount++;
            }
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

    for (size_t i = 0; i < weights.size(); ++i)
    {
        if (weights[i] < 0.1)
        {
            reductionCount += 1.0;
            // Modify the weights directly in the input vector
            weights[i] = 0.0;
        }
    }

    classificationRate = ToolsHelper::computeAccuracy(data, weights);
    reductionRate = reductionCount / static_cast<double>(weights.size());
    fitness = reductionRate * alpha + classificationRate * (1 - alpha);

    return fitness;
}
