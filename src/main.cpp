#include <iostream>
#include <string>
#include <vector>
#include "ToolsHelper.cpp" // Include the header file for ToolHelper

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "[ERROR] Incorrect number of arguments." << std::endl;
        std::cerr << "Usage: ./main {seed} [1-3]" << std::endl;
        std::cerr << "Where [1-3] corresponds to: 1=spectf-heart, 2=parkinsons, 3=ionosphere" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    double Seed = std::stod(argv[1]);
    int option = std::stoi(argv[2]);

    std::string path;
    ToolsHelper toolHelper; // Create an instance of ToolHelper

    switch (option)
    {
    case 1:
        path = "./data/spectf-heart.arff";
        break;
    case 2:
        path = "./data/parkinsons.arff";
        break;
    case 3:
        path = "./data/ionosphere.arff";
        break;
    default:
        std::cerr << "[ERROR] Unrecognized parameter." << std::endl;
        std::cerr << "You must specify the dataset: 1=spectf-heart, 2=parkinsons, 3=ionosphere." << std::endl;
        std::cerr << "Usage: ./main {seed} [1-3]" << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> data_matrix;
    std::vector<char> class_vector;

    try
    {
        toolHelper.readDataARFF(path, data_matrix, class_vector);

        // Display the data in a fancy way
        toolHelper.displayDataInfo(data_matrix, class_vector);

        int k = 10; // Number of partitions

        auto partitions = toolHelper.createPartitions(data_matrix, class_vector, k);

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
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
