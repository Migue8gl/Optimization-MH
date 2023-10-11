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

        for (unsigned int j = 0; j < partitions[i].size(); ++j)
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

void Tests::testKNNClassifier(const Data &data, const unsigned int &k)
{
    std::cout << "Testing KNNClassifier with k = " << k << std::endl;

    // Test element (you can customize this)
    int seed = Seed::getInstance().getSeed();
    int randomIndex = ToolsHelper::generateUniformRandomNumberInteger(0, data.size() - 1, seed);
    std::vector<double> element = data.getData()[randomIndex];
    std::vector<double> weights(data.getData()[0].size(), 1.0);

    // Compute the predicted class
    char predictedClass = MLTools::KNNClassifier(data, element, weights, k);

    // Get the true class label (assuming it's available in the 'data' object)
    char trueClass = data.getLabels()[randomIndex];

    // Print information
    std::cout << "\nTest Element: [";
    for (unsigned int i = 0; i < element.size(); i++)
    {
        std::cout << element[i];
        if (i < element.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;

    std::cout << "\nWeights: [";
    for (unsigned int i = 0; i < weights.size(); i++)
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
        for (unsigned int j = 0; j < dataMatrix[i].size(); ++j)
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

void Tests::runTests(const Data &data, int dataset)
{
    // Redirect stdout to a text file for testPartitions
    std::ofstream outFile1("./files/tests/test_partitions_" + std::to_string(dataset) + ".txt");
    std::streambuf *coutBuffer1 = std::cout.rdbuf(); // Save cout buffer
    std::cout.rdbuf(outFile1.rdbuf());               // Redirect cout to outFile1

    std::cout << "-------- Test partitions --------\n"
              << std::endl;
    Tests::testPartitions(data);

    // Restore cout
    std::cout.rdbuf(coutBuffer1);

    // Redirect stdout to a text file for testReadAndDisplayData
    std::ofstream outFile2("./files/tests/test_data_reading_" + std::to_string(dataset) + ".txt");
    std::streambuf *coutBuffer2 = std::cout.rdbuf(); // Save cout buffer
    std::cout.rdbuf(outFile2.rdbuf());               // Redirect cout to outFile2

    std::cout << "-------- Test Data Reading --------\n"
              << std::endl;
    Tests::testReadAndDisplayData(data);

    // Restore cout
    std::cout.rdbuf(coutBuffer2);

    // Redirect stdout to a text file for testKNNClassifier
    std::ofstream outFile3("./files/tests/test_knn_classifier_" + std::to_string(dataset) + ".txt");
    std::streambuf *coutBuffer3 = std::cout.rdbuf(); // Save cout buffer
    std::cout.rdbuf(outFile3.rdbuf());               // Redirect cout to outFile3

    std::cout << "-------- Test KNN Classifier --------\n"
              << std::endl;
    Tests::testKNNClassifier(data, 10); // You can customize the k value

    // Restore cout
    std::cout.rdbuf(coutBuffer3);

    // Redirect stdout to a text file for testShuffledData
    std::ofstream outFile4("./files/tests/test_shuffled_data_" + std::to_string(dataset) + ".txt");
    std::streambuf *coutBuffer4 = std::cout.rdbuf(); // Save cout buffer
    std::cout.rdbuf(outFile4.rdbuf());               // Redirect cout to outFile4

    std::cout << "-------- Test Shuffled Data --------\n"
              << std::endl;
    Tests::testShuffledData(data, Seed::getInstance().getSeed(), 4); // You can customize the seed and k value

    // Restore cout
    std::cout.rdbuf(coutBuffer4);
}