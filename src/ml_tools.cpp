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
                for (unsigned int j = 0; j < otherData.size(); j++)
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

std::vector<double> MLTools::localSearch(const Data &data, const std::string &opt)
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
    std::vector<double> w(data.getData()[0].size());

    // Initialize w with random normal values in one line
    std::generate(w.begin(), w.end(), [&]()
                  { return ToolsHelper::generateNormalRandomNumber(mean, std::sqrt(variance), Seed::getInstance().getSeed()); });

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
        // w[randIndex] = std::max(0.0, std::min(1.0, w[randIndex]));
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

double MLTools::computeFitness(const Data &data, std::vector<double> weights, const double &alpha)
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

std::vector<double> MLTools::computePopulationFitness(const Data &data, std::vector<std::vector<double>> populationWeights, const double &alpha)
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

    std::random_device rd;
    std::mt19937 eng(rd);

    // Levy flight formula
    std::gamma_distribution<double> dist(alpha, scale);

    return dist(eng);
}

void MLTools::migration(std::vector<std::vector<double>> &subpob1, std::vector<std::vector<double>> &subpob2, double period, double p)
{
    for (std::vector<double> &butterfly : subpob1)
    {
        for (double &value : butterfly)
        {
            double n = ToolsHelper::generateUniformRandomNumberDouble(0, 1, Seed::getInstance().getSeed()) * period;
            if (n <= p)
            {
                int index = ToolsHelper::generateUniformRandomNumberDouble(0, 1, Seed::getInstance().getSeed()) * subpob1.size();
                value = subpob1[index][&value - &butterfly[0]];
            }
            else
            {
                int index = ToolsHelper::generateUniformRandomNumberDouble(0, 1, Seed::getInstance().getSeed()) * subpob2.size();
                value = subpob2[index][&value - &butterfly[0]];
            }
        }
    }
}

void MLTools::adjust(std::vector<std::vector<double>> &subpob1, std::vector<std::vector<double>> &subpob2, const std::vector<int> &indexbest1, const std::vector<int> &indexbest2, double p, const std::vector<double> &fitnessPopulation, double BAR, double alpha)
{
    for (std::vector<double> &butterfly : subpob2)
    {
        for (double &value : butterfly)
        {
            double n = ToolsHelper::generateUniformRandomNumberDouble(0, 1, Seed::getInstance().getSeed());

            if (n <= p)
            {
                int i = &value - &butterfly[0];
                if (fitnessPopulation[indexbest1[i]] > fitnessPopulation[indexbest2[i] + subpob1.size()])
                {
                    value = subpob1[indexbest1[i]][i];
                }
                else
                {
                    value = subpob2[indexbest2[i]][i];
                }
            }
            else
            {
                int index = ToolsHelper::generateUniformRandomNumberDouble(0, 1, Seed::getInstance().getSeed()) * subpob2.size();
                value = subpob2[index][&value - &butterfly[0]];

                if (n > BAR)
                {
                    double dx = levyFlight();
                    value += alpha * (dx - 0.5);
                }
            }
        }
    }
}

std::vector<double> MLTools::mbo(const Data &data, bool ls)
{
    double variance = 0.3, mean = 0.0;
    double alpha = 0.5;

    // Initialize population of mariposas NP
    std::vector<std::vector<double>> np(data.size());
    std::vector<double> w(data.getData()[0].size());
    for (std::vector<double> &bf : np)
    {
        std::generate(bf.begin(), bf.end(), ToolsHelper::generateNormalRandomNumber(mean, std::sqrt(variance), Seed::getInstance().getSeed()));
    }

    int t = 1;
    const int maxGen = 10;
    double BAR = 5.0 / 12.0;
    double p = 5.0 / 12.0;
    double periodo = 1.2;
    double smax = 1.0;
    double alpha = 0;
    int np1Size = static_cast<int>(p * np.size());
    int np2Size = np.size() - np1Size;
    double cont = 0;
    std::vector<double> fitnessPopulation;
    std::vector<int> indices1(np1Size), indices2(np2Size);
    std::iota(indices1.begin(), indices1.end(), 0);
    std::iota(indices2.begin(), indices2.end(), 0);
    fitnessPopulation = MLTools::computePopulationFitness(data, np, alpha); // Replace with your actual evaluation function

    while (t < maxGen)
    {
        // Sort in descending order of fitness
        std::sort(indices1.begin(), indices1.end(), [&fitnessPopulation](int a, int b)
                  { return fitnessPopulation[a] > fitnessPopulation[b]; });
        std::sort(indices2.begin(), indices2.end(), [&fitnessPopulation](int a, int b)
                  { return fitnessPopulation[a] > fitnessPopulation[b]; });

        std::vector<std::vector<double>> np1, np2;
        cont = 0;
        for (const std::vector<double> &bf : np)
        {
            if (cont < np1Size)
            {
                np1.push_back(bf);
            }
            else
            {
                np2.push_back(bf);
            }
            cont++;
        }

        alpha = smax / (t * t);

        std::thread mig_thread(migration, std::ref(np1), std::ref(np2), periodo, p);
        std::thread aju_thread(adjust, std::ref(np1), std::ref(np2), indices1, indices2, p, fitnessPopulation, BAR, alpha);

        mig_thread.join();
        aju_thread.join();

        // Copy the data from np1 and np2 back to the np array
        np = np1;
        np.insert(np.end(), np2.begin(), np2.end());

        std::mt19937 gen(Seed::getInstance().getSeed());
        std::shuffle(np.begin(), np.end(), gen);
        fitnessPopulation = MLTools::computePopulationFitness(data, np, alpha); // Replace with your actual evaluation function
        t++;
        cont = 0;
    }

    double best_eval = fitnessPopulation[0];
    unsigned int best_solution_index = 0;
    for (unsigned int i = 0; i < fitnessPopulation.size(); ++i)
    {
        if (fitnessPopulation[i] > best_eval)
        {
            best_eval = fitnessPopulation[i];
            best_solution_index = i;
        }
    }

    if (ls)
        MLTools::localSearch(data, np[best_solution_index]); // Implement localSearch function

    return np[best_solution_index];
}
