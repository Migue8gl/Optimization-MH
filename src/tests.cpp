#include "tests.h"
#include "tools_helper.h"
#include "ml_tools.h"
#include <map>
#include "seed.h"

void Tests::testReadAndDisplayData(const Data &data)
{
    ToolsHelper::displayDataInfo(data);
}

void Tests::testPartitions(const Data &data, int k)
{
    std::vector<Data> partitions = data.createPartitions(k);

    // Display partition information (for demonstration)
    for (int i = 0; i < k; ++i)
    {
        std::cout << "Partition " << i + 1 << ":\n";

        // Separator line
        std::cout << "----------------------------------\n";

        // Count classes in the current partition
        std::map<char, int> classCounts;

        for (size_t j = 0; j < partitions[i].size(); ++j)
        {
            char currentClass = partitions[i].getLabels()[j];

            // Update class counts for the current partition
            classCounts[currentClass]++;
        }

        // Display class counts for the current partition
        std::cout << "Class Counts in Partition " << i + 1 << ":\n";
        for (const auto &entry : classCounts)
        {
            std::cout << "Class " << entry.first << ": " << entry.second << " instances\n";
        }

        // Display total number of instances in the current partition
        std::cout << "Total Instances in Partition " << i + 1 << ": " << partitions[i].size() << " instances\n";

        // Separator line between partitions
        std::cout << "----------------------------------\n\n";
    }
}

void Tests::testKNNClassifier(const Data &data, int k)
{
    std::cout << "Testing KNNClassifier with k = " << k << std::endl;

    // Test element (you can customize this)
    int seed = Seed::getInstance().getSeed();
    int randomIndex = ToolsHelper::generateRandomNumberInteger(0, data.size() - 1, seed);
    std::vector<double> element = data.getData()[randomIndex];
    std::vector<double> weights(data.getData()[0].size(), 1.0);

    // Compute the predicted class
    char predictedClass = MLTools::KNNClassifier(data, element, weights, k);

    // Get the true class label (assuming it's available in the 'data' object)
    char trueClass = data.getLabels()[randomIndex];

    // Print information
    std::cout << "\nTest Element: [";
    for (int i = 0; i < element.size(); i++)
    {
        std::cout << element[i];
        if (i < element.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "\nWeights: [";
    for (int i = 0; i < weights.size(); i++)
    {
        std::cout << weights[i];
        if (i < weights.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "\nPredicted Class = " << predictedClass << std::endl;
    std::cout << "True Class = " << trueClass << std::endl;

    // Check if the prediction is correct and print the result
    if (predictedClass == trueClass)
    {
        std::cout << "Prediction is correct." << std::endl;
    }
    else
    {
        std::cout << "Prediction is incorrect." << std::endl;
    }
}

void Tests::printDataInfo(const Data &data)
{
    const std::vector<std::vector<double>> &dataMatrix = data.getData();
    const std::vector<char> &labels = data.getLabels();

    std::cout << "---------------------------------" << std::endl;

    std::cout << "\nFirst few data points:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(5, data.size()); ++i)
    {
        std::cout << "Data point " << i + 1 << ": ";
        for (size_t j = 0; j < dataMatrix[i].size(); ++j)
        {
            if (j > 0)
                std::cout << ", ";
            std::cout << dataMatrix[i][j];
        }
        std::cout << " (Label: " << labels[i] << ")\n"
                  << std::endl;
    }
    std::cout << "---------------------------------" << std::endl;
}

void Tests::testShuffledData(const Data &data, int seedValue, int numShuffles)
{
    std::cout << "Data Info Before Shuffling (Seed: " << seedValue << "):" << std::endl;
    Tests::printDataInfo(data);

    for (int i = 1; i <= numShuffles; ++i)
    {
        std::mt19937 rng(seedValue); // Reinitialize the random number generator with the same seed
        std::vector<std::vector<double>> shuffledData = data.getData();
        std::vector<char> shuffledLabels = data.getLabels();

        std::shuffle(shuffledData.begin(), shuffledData.end(), rng);
        std::shuffle(shuffledLabels.begin(), shuffledLabels.end(), rng);

        Data shuffledDataObject(shuffledData, shuffledLabels);

        std::cout << "\nData Info After Shuffling " << i << " Time(s):" << std::endl;
        Tests::printDataInfo(shuffledDataObject);
    }
}