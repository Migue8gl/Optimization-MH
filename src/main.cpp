#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <vector>
#include "tests.h"
#include "tools_helper.h"
#include "seed.h"

/**
 * @brief Test function to debug.
 *
 * This function takes data information and dumps all information
 * for debuggin purposes in a txt file.
 *
 * @param data Instace of Data class, contains information about data labels and data points.
 * @param dataset Representative integer of the dataset
 */
void runTests(const Data &data, int dataset)
{
    // Redirect stdout to a text file for testPartitions
    std::ofstream outFile1("./files/tests/test_partitions_" + std::to_string(dataset) + ".txt");
    std::streambuf *coutBuffer1 = std::cout.rdbuf(); // Save cout buffer
    std::cout.rdbuf(outFile1.rdbuf());               // Redirect cout to outFile1

    std::cout << "-------- Test particiones --------\n"
              << std::endl;
    Tests::testPartitions(data);

    // Restore cout
    std::cout.rdbuf(coutBuffer1);

    // Redirect stdout to a text file for testReadAndDisplayData
    std::ofstream outFile2("./files/tests/test_data_reading_" + std::to_string(dataset) + ".txt");
    std::streambuf *coutBuffer2 = std::cout.rdbuf(); // Save cout buffer
    std::cout.rdbuf(outFile2.rdbuf());               // Redirect cout to outFile2

    std::cout << "-------- Test data reading --------\n"
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
    Seed::getInstance().setSeed(std::stod(argv[1]));
    int option = std::stoi(argv[2]);
    std::string path;
    bool run_test = false;

    if (argc >= 4)
    {
        std::string testOption = argv[3];
        if (ToolsHelper::toUpperCase(testOption) == "TRUE" || testOption == "1")
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
        std::cerr << "Usage: ./main {seed} [1-3] {run_tests}" << std::endl;
        return 1;
    }

    try
    {
        Data data;
        data.readDataARFF(path);
        ToolsHelper::normalizeData(data);
        std::vector<Data> partitions = data.createPartitions(5);
        ToolsHelper::execute(partitions, "1");

        if (run_test)
        {
            runTests(data, option);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
};
