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
        std::cerr << "Usage: ./main -s {seed} -d {dataset}[1-4] -t {run_tests}" << std::endl;
        std::cerr << "Where [1-4] corresponds to: 1=spectf-heart, 2=parkinsons, 3=ionosphere, 4=diabetes" << std::endl;
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
            std::cerr << "Usage: ./main -s {seed} -d {dataset}[1-4] -t {run_tests}" << std::endl;
            std::cerr << "Where [1-4] corresponds to: 1=spectf-heart, 2=parkinsons, 3=ionosphere, 4=diabetes" << std::endl;
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
    case 4:
        path.push_back("./data/diabetes.arff");
        break;
    default:
        path.push_back("./data/spectf-heart.arff");
        path.push_back("./data/parkinsons.arff");
        path.push_back("./data/ionosphere.arff");
        path.push_back("./data/diabetes.arff");
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
            std::vector<std::string> hyperParams;
            data.readDataARFF(p);
            ToolsHelper::normalizeDataZeroOne(data);
            
            std::cout << "------------------------------------------" << std::endl;
            std::cout << "Calling MLTools::kCrossValidation with KNN" << std::endl;
            std::cout << "------------------------------------------" << std::endl;

            MLTools::kCrossValidation(data, MLTools::knn, 5, hyperParams);

            std::cout << "---------------------------------------------------" << std::endl;
            std::cout << "Calling MLTools::kCrossValidation with Local Search" << std::endl;
            std::cout << "---------------------------------------------------" << std::endl;

            MLTools::kCrossValidation(data, MLTools::localSearch, 5, hyperParams);

            std::cout << "------------------------------------------" << std::endl;
            std::cout << "Calling MLTools::kCrossValidation with MBO" << std::endl;
            std::cout << "------------------------------------------" << std::endl;

            MLTools::kCrossValidation(data, MLTools::mbo, 5, hyperParams);

            std::cout << "----------------------------------------------------------" << std::endl;
            std::cout << "Calling MLTools::kCrossValidation with Simulated Annealing" << std::endl;
            std::cout << "----------------------------------------------------------" << std::endl;

            MLTools::kCrossValidation(data, MLTools::simulatedAnnealing, 5, hyperParams);

            if (run_test)
            {
                Tests::runTests(data, option);
            }

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
