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

int main(int argc, char *argv[])
{
    if (argc > 7)
    {
        std::cerr << "[ERROR] Incorrect number of arguments." << std::endl;
        std::cerr << "Usage: ./main -s {seed} -d {dataset}[1-3] -t {run_tests}" << std::endl;
        std::cerr << "Where [1-3] corresponds to: 1=spectf-heart, 2=parkinsons, 3=ionosphere" << std::endl;
        return 1;
    }

    // Parse command-line arguments
    std::string sarg;
    int option = -1;
    bool run_test = false;
    for (int i = 1; i < argc;)
    {
        sarg = argv[i++];

        if (sarg == "-s")
        {
            Seed::getInstance().setSeed(std::stod(argv[i++]));
        }

        else if (sarg == "-d")
        {
            option = std::stoi(argv[i++]);
        }

        else if (sarg == "-t")
        {
            std::string testOption = argv[i++];
            if (ToolsHelper::toUpperCase(testOption) == "TRUE" || testOption == "1")
            {
                run_test = true;
            }
        }
        else
        {
            std::cerr << "[ERROR] Unrecognized parameters." << std::endl;
            std::cerr << "Usage: ./main -s {seed} -d {dataset}[1-3] -t {run_tests}" << std::endl;
            std::cerr << "Where [1-3] corresponds to: 1=spectf-heart, 2=parkinsons, 3=ionosphere" << std::endl;
            return 1;
        }
    }

    std::vector<std::string> path;
    int cont = 0;
    switch (option)
    {
    case 1:
        path.push_back("./data/spectf-heart.arff");
        break;
    case 2:
        path.push_back("./data/parkinsons.arff");
        break;
    case 3:
        path.push_back("./data/ionosphere.arff");
        break;
    default:
        path.push_back("./data/spectf-heart.arff");
        path.push_back("./data/parkinsons.arff");
        path.push_back("./data/ionosphere.arff");
        option = 0;
        cont = 1;
        break;
    }

    try
    {
        for (const std::string &p : path)
        {
            std::cout << "<------------------ " << ToolsHelper::getDatasetTitle(option + cont) << " ------------------>" << std::endl;

            Data data;
            data.readDataARFF(p);
            ToolsHelper::normalizeData(data);
            MLTools::kCrossValidation(data, MLTools::KNN);

            if (run_test)
            {
                runTests(data, option);
            }

            if (cont != 0)
                cont++;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
};
