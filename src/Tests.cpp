#include <string>
#include <iostream>
#include <vector>
#include "ToolsHelper.cpp"

class Tests
{
public:
    static void testReadAndDisplayData(const std::vector<std::vector<double>> data_matrix,
                                       const std::vector<char> class_vector)
    {
        ToolsHelper::displayDataInfo(data_matrix, class_vector);
    }

    static void testPartitions(const std::vector<std::vector<double>> data_matrix,
                               const std::vector<char> class_vector)
    {
        int k = 10; // Number of partitions

        auto partitions = ToolsHelper::createPartitions(data_matrix, class_vector, k);

        // Display partition information (for demonstration)
        for (int i = 0; i < k; ++i)
        {
            std::cout << "Partition " << i + 1 << ":\n";

            // Separator line
            std::cout << "----------------------------------\n";

            // Count classes in the current partition
            std::map<char, int> classCounts;

            for (size_t j = 0; j < partitions.second[i].size(); ++j)
            {
                char currentClass = partitions.second[i][j];

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
            std::cout << "Total Instances in Partition " << i + 1 << ": " << partitions.second[i].size() << " instances\n";

            // Separator line between partitions
            std::cout << "==================================\n\n";
        }
    }
};