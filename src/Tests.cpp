#include "Tests.h"
#include "ToolsHelper.h"
#include "MLAlgorithms.h"
#include <map>

void Tests::testReadAndDisplayData(const Data &data)
{
    ToolsHelper::displayDataInfo(data);
}

void Tests::testPartitions(const Data &data, int k)
{
    std::vector<Data> partitions = ToolsHelper::createPartitions(data, k);

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
    std::vector<double> element = {2.5, 3.5};

    char predictedClass = MLAlgorithms::KNNClassifier(data, element, k);
    std::cout << "Predicted Class = " << predictedClass << std::endl;
}
