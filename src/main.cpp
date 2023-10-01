#include <iostream>
#include <string>
#include <vector>
#include "Tests.cpp"

void runTests(const std::vector<std::vector<double>> data_matrix,
              const std::vector<char> class_vector, int dataset)
{
    // Redirect stdout to a text file for testPartitions
    std::ofstream outFile1("./files/tests/test_partitions_" + std::to_string(dataset) + ".txt");
    std::streambuf *coutBuffer1 = std::cout.rdbuf(); // Save cout buffer
    std::cout.rdbuf(outFile1.rdbuf());               // Redirect cout to outFile1

    std::cout << "-------- Test particiones --------\n"
              << std::endl;
    Tests::testPartitions(data_matrix, class_vector);

    // Restore cout
    std::cout.rdbuf(coutBuffer1);

    // Redirect stdout to a text file for testReadAndDisplayData
    std::ofstream outFile2("./files/tests/test_data_reading_" + std::to_string(dataset) + ".txt");
    std::streambuf *coutBuffer2 = std::cout.rdbuf(); // Save cout buffer
    std::cout.rdbuf(outFile2.rdbuf());               // Redirect cout to outFile2

    std::cout << "-------- Test data reading --------\n"
              << std::endl;
    Tests::testReadAndDisplayData(data_matrix, class_vector);

    // Restore cout
    std::cout.rdbuf(coutBuffer2);
}

int main(int argc, char *argv[])
{
    if (argc > 4 || argc < 2)
    {
        std::cerr << "[ERROR] Incorrect number of arguments." << std::endl;
        std::cerr << "Usage: ./main {seed} [1-3] {run_tests}" << std::endl;
        std::cerr << "Where [1-3] corresponds to: 1=spectf-heart, 2=parkinsons, 3=ionosphere" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    double Seed = std::stod(argv[1]);
    int option = std::stoi(argv[2]);
    std::string path;
    bool run_test = false;

    if (argc >= 4)
    {
        std::string testOption = argv[3];
        if (testOption == "true" || testOption == "TRUE" || testOption == "1")
        {
            run_test = true;
        }
    }

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

        if (run_test)
        {
            runTests(data_matrix, class_vector, option);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
};
