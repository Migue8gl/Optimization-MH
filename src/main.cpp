#include <iostream>
#include <string>
#include <vector>
#include "ToolsHelper.cpp" // Include the header file for ToolHelper
#include "Tests.cpp"

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
        ToolsHelper::readDataARFF(path, data_matrix, class_vector);
        Tests::testPartitions(data_matrix, class_vector);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
};
