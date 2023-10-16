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
#include "math_tools.h"
#include <functional>

char MLTools::kNNClassifier(const Data &data, const std::vector<double> &element, const std::vector<double> &weights, const unsigned int &k)
{
    unsigned int dataSize = data.size();
    const std::vector<std::vector<double>> &dataMatrix = data.getData();
    const std::vector<char> &dataLabels = data.getLabels();

    if (k <= 0 || k > dataSize)
    {
        throw std::invalid_argument("Invalid value of k");
    }

    std::vector<std::pair<double, char>> distancesAndClasses;

    for (std::vector<std::vector<double>>::const_iterator dataIter = dataMatrix.begin(); dataIter != dataMatrix.end(); ++dataIter)
    {
        std::vector<char>::const_iterator labelIter = dataLabels.begin() + (dataIter - dataMatrix.begin());
        if (element != *dataIter)
        {
            double distance = MathTools::computeEuclideanDistance(element, *dataIter, weights);
            distancesAndClasses.emplace_back(distance, *labelIter);
        }
    }

    // Sort distancesAndClasses by distance (ascending order)
    std::sort(distancesAndClasses.begin(), distancesAndClasses.end(), [](const auto &a, const auto &b)
              { return a.first < b.first; });

    // Count the occurrences of each class among the k nearest neighbors
    std::unordered_map<char, int> classCounts;
    for (unsigned int i = 0; i < k && i < distancesAndClasses.size(); ++i)
    {
        char cls = distancesAndClasses[i].second;
        classCounts[cls]++;
    }

    // Find the class with the highest count (mode)
    char predictedClass = '\0';
    int maxCount = -1;
    for (std::unordered_map<char, int>::iterator it = classCounts.begin(); it != classCounts.end(); ++it)
    {
        if (it->second > maxCount)
        {
            predictedClass = it->first;
            maxCount = it->second;
        }
    }

    return predictedClass;
}

void MLTools::kCrossValidation(const Data &data, const Optimizers::Optimizer &optimizer, const int numberPartitions, std::map<std::string, std::string> &hyperParams)
{
    const double alpha = 0.5;
    double TS_average = 0, TR_average = 0, A_average = 0;
    bool showPartitions = true;

    auto overallStartTime = std::chrono::high_resolution_clock::now();

    std::cout << "\n***** (CROSS VALIDATION K = " << numberPartitions << ") *****\n"
              << std::endl;

    std::vector<Data> partitions = data.createPartitions(numberPartitions);

    for (std::vector<Data>::iterator partitionIt = partitions.begin(); partitionIt != partitions.end(); ++partitionIt)
    {
        auto startTime = std::chrono::high_resolution_clock::now();
        const Data &testData = *partitionIt;
        Data trainingData;
        unsigned int reductionCount = 0;

        for (std::vector<Data>::iterator trainDataIt = partitions.begin(); trainDataIt != partitions.end(); ++trainDataIt)
        {
            if (trainDataIt != partitionIt)
            {
                trainingData.mergeData(*trainDataIt);
            }
        }

        std::vector<double> initialWeights = ToolsHelper::initializeWeights(data.parameterSize());
        std::vector<double> weights = optimizer(trainingData, initialWeights, hyperParams);

        for (std::vector<double>::iterator wi = weights.begin(); wi != weights.end(); ++wi)
        {
            if (*wi < 0.1)
            {
                reductionCount++;
                *wi = 0.0;
            }
        }

        double classificationAccuracy = MLTools::computeAccuracy(testData, weights);
        double reductionRate = static_cast<double>(reductionCount) / static_cast<double>(weights.size());
        double fitness = alpha * classificationAccuracy + (1.0 - alpha) * reductionRate;

        TS_average += classificationAccuracy;
        TR_average += reductionRate;
        A_average += fitness;

        // Calculate progress
        float progress = static_cast<float>(partitionIt - partitions.begin() + 1) / partitions.size();
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> executionTime = endTime - startTime;

        // Calculate minutes, seconds, and milliseconds
        int minutes = static_cast<int>(executionTime.count()) / 60;
        int seconds = static_cast<int>(executionTime.count()) % 60;
        int milliseconds = static_cast<int>((executionTime.count() - static_cast<int>(executionTime.count())) * 1000);

        // Update the progress bar
        ToolsHelper::progress_bar(progress);
        if (showPartitions)
        {
            size_t partNumber = partitionIt - partitions.begin();
            std::cout << "\n\n[PART " << partNumber + 1 << "] | Classification Rate: " << classificationAccuracy << std::endl;
            std::cout << "[PART " << partNumber + 1 << "] | Reduction Rate: " << reductionRate << std::endl;
            std::cout << "[PART " << partNumber + 1 << "] | Fitness: " << fitness << std::endl;
            std::cout << "[PART " << partNumber + 1 << "] | Execution Time: " << minutes << " min " << seconds << " sec " << milliseconds << " ms\n\n";
            std::cout << "--------------------------------------\n"
                      << std::endl;
        }
    }

    auto overallEndTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> totalTime = overallEndTime - overallStartTime;

    // Calculate total minutes, seconds, and milliseconds
    int totalMinutes = static_cast<int>(totalTime.count()) / 60;
    int totalSeconds = static_cast<int>(totalTime.count()) % 60;
    int totalMilliseconds = static_cast<int>((totalTime.count() - static_cast<int>(totalTime.count())) * 1000);

    if (!showPartitions)
        std::cout << "\n\n";
    std::cout << "***** (FINAL RESULTS) *****\n"
              << std::endl;
    std::cout << "Average Classification Rate: " << TS_average / partitions.size() << std::endl;
    std::cout << "Average Reduction Rate: " << TR_average / partitions.size() << std::endl;
    std::cout << "Average Fitness: " << A_average / partitions.size() << std::endl;
    std::cout << "Total Execution Time: " << totalMinutes << " min " << totalSeconds << " sec " << totalMilliseconds << " ms\n\n";
}

double MLTools::computeFitness(const Data &data, std::vector<double> &weights, const double &alpha)
{
    double classificationRate = 0.0;
    double reductionRate = 0.0;
    double reductionCount = 0.0;
    double fitness;

    for (std::vector<double>::iterator it = weights.begin(); it != weights.end(); ++it)
    {
        if (*it < 0.1)
        {
            reductionCount += 1.0;
            *it = 0.0;
        }
    }
    classificationRate = MLTools::computeAccuracy(data, weights);
    reductionRate = reductionCount / static_cast<double>(weights.size());
    fitness = reductionRate * alpha + classificationRate * (1 - alpha);

    return fitness;
}

std::vector<double> MLTools::computePopulationFitness(const Data &data, std::vector<std::vector<double>> &populationWeights, const double &alpha)
{
    std::vector<double> fitness(populationWeights.size());

    for (std::vector<std::vector<double>>::iterator popWeightIt = populationWeights.begin(); popWeightIt != populationWeights.end(); ++popWeightIt)
    {
        double classificationRate = 0.0;
        double reductionRate = 0.0;
        double reductionCount = 0.0;

        for (std::vector<double>::iterator weightValueIt = popWeightIt->begin(); weightValueIt != popWeightIt->end(); ++weightValueIt)
        {
            if (*weightValueIt < 0.1)
            {
                reductionCount += 1.0;
                *weightValueIt = 0.0;
            }
        }

        classificationRate = MLTools::computeAccuracy(data, *popWeightIt);
        reductionRate = reductionCount / static_cast<double>(popWeightIt->size());

        fitness[popWeightIt - populationWeights.begin()] = reductionRate * alpha + classificationRate * (1 - alpha);
    }

    return fitness;
}

double MLTools::computeAccuracy(const Data &sample, const std::vector<double> &weights)
{
    double correctlyClassifiedInstances = 0.0;
    const std::vector<std::vector<double>> &samples = sample.getData();
    const std::vector<char> &classes = sample.getLabels();
    unsigned int totalInstances = sample.size();

    auto sampleIterator = samples.begin();
    auto classIterator = classes.begin();

    for (; sampleIterator != samples.end(); ++sampleIterator, ++classIterator)
    {
        char predictedClass = MLTools::kNNClassifier(sample, *sampleIterator, weights);

        if (predictedClass == *classIterator)
            correctlyClassifiedInstances += 1.0;
    }

    return correctlyClassifiedInstances / static_cast<double>(totalInstances);
}
