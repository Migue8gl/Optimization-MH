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
            double distance = MLTools::computeEuclideanDistance(element, *dataIter, weights);
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

        // Test
        std::vector<double> w(data.getData()[0].size());
        std::generate(w.begin(), w.end(), [&]()
                      { return ToolsHelper::generateNormalRandomNumber(0.0, 1.0, Seed::getInstance().getSeed()); });

        std::vector<double> weights = optimizer(trainingData, w, hyperParams);

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

double MLTools::computeEuclideanDistance(const std::vector<double> &point1, const std::vector<double> &point2, const std::vector<double> &weights)
{
    if (point1.size() != point2.size() || point1.size() != weights.size())
    {
        throw std::invalid_argument("Vector dimensions and weights must match.");
    }

    double sum = 0.0;

    for (std::vector<double>::const_iterator it1 = point1.begin(), it2 = point2.begin(), itWeights = weights.begin(); it1 != point1.end(); ++it1, ++it2, ++itWeights)
    {
        double diff = *it1 - *it2;
        sum += *itWeights * (diff * diff);
    }

    return sqrt(sum);
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

void MLTools::migration(std::vector<std::vector<double>> &subpob1, std::vector<std::vector<double>> &subpob2, double period, double p)
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

void MLTools::adjust(std::vector<std::vector<double>> &subpob2, const std::vector<double> &bestButterfly, double p, double BAR, double alpha)
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

std::vector<std::vector<double>> MLTools::elitism(const Data &data, std::vector<std::vector<double>> &np, unsigned int numElite)
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
    std::vector<double> fitnessPopulation(data.getData()[0].size(), 0);
    std::vector<std::string> hyperLSParams;

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
            bestButterfly = MLTools::localSearch(data, np[bestSolutionIndex], hyperLSParams);
            break;
        case 2:
            bestButterfly = MLTools::simulatedAnnealing(data, np[bestSolutionIndex], hyperLSParams);
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
        np[bestSolutionIndex] = MLTools::localSearch(data, np[bestSolutionIndex], hyperLSParams);
        break;
    case 2:
        np[bestSolutionIndex] = MLTools::simulatedAnnealing(data, np[bestSolutionIndex], hyperLSParams);
        break;
    }

    return np[bestSolutionIndex];
}

std::vector<double> MLTools::simulatedAnnealing(const Data &data, std::vector<double> &weights, std::vector<std::string> &hyperParams)
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