#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>
#include <vector>
#include "tests.h"
#include "tools_helper.h"
#include "seed.h"
#include "ml_tools.h"


int main(int argc, char *argv[])
{
    if (argc > 6)
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
            run_test = true;
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
            ToolsHelper::normalizeDataCeroOne(data);
            // Separator for the first function call
            std::cout << "------------------------------------------" << std::endl;
            std::cout << "Calling MLTools::kCrossValidation with KNN" << std::endl;
            std::cout << "------------------------------------------" << std::endl;

            // Call the first function
            MLTools::kCrossValidation(data, MLTools::KNN);

            // Separator for the second function call
            std::cout << "--------------------------------------------------" << std::endl;
            std::cout << "Calling MLTools::kCrossValidation with localSearch" << std::endl;
            std::cout << "--------------------------------------------------" << std::endl;

            // Call the second function
            MLTools::kCrossValidation(data, MLTools::localSearch);

            // Separator for the third function call
            std::cout << "------------------------------------------" << std::endl;
            std::cout << "Calling MLTools::kCrossValidation with MBO" << std::endl;
            std::cout << "------------------------------------------" << std::endl;

            // Call the third function
            MLTools::kCrossValidation(data, MLTools::mbo);

            if (run_test)
            {
                Tests::runTests(data, option);
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
