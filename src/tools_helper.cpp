#include "tools_helper.h"
#include "ml_tools.h"
#include <cctype>

// Define and initialize the static members
std::mt19937 ToolsHelper::randomGenerator;
std::uniform_int_distribution<int> ToolsHelper::randomIntDistribution;
std::uniform_real_distribution<double> ToolsHelper::randomRealDistribution;

std::string ToolsHelper::toUpperCase(const std::string &str)
{
    std::string result = str;
    for (char &c : result)
    {
        c = std::toupper(c);
    }
    return result;
}

void ToolsHelper::displayDataInfo(const Data &data, const std::string &separator)
{
    std::vector<std::vector<double>> data_matrix = data.getData();
    std::vector<char> data_labels = data.getLabels();

    if (data_labels.empty() || data_matrix.empty())
    {
        std::cout << "No data to display." << std::endl;
        return;
    }

    const size_t numElements = data.size();

    // Display dataset information
    std::cout << "Dataset Information:" << std::endl;
    std::cout << "Number of Elements: " << numElements << std::endl;

    // Extract unique class labels
    std::set<char> uniqueClasses(data_labels.begin(), data_labels.end());

    std::cout << "Number of Classes: " << uniqueClasses.size() << std::endl;
    std::cout << "Class Labels: ";
    for (char label : uniqueClasses)
    {
        std::cout << label << " ";
    }
    std::cout << std::endl
              << std::endl;

    // Display instances with enumeration and separator
    for (size_t i = 0; i < numElements; ++i)
    {
        std::cout << "Instance [" << i + 1 << "] -> "; // Numerical index enclosed in square brackets
        for (size_t j = 0; j < data_matrix[0].size(); ++j)
        {
            std::cout << data_matrix[i][j] << " ";
        }
        std::cout << "- CLASS [" << data_labels[i] << "]" << std::endl;
        if (i != data_matrix.size() - 1)
        {
            std::cout << separator << std::endl;
        }
    }
}

void ToolsHelper::normalizeData(Data &data)
{
    std::vector<std::vector<double>> data_matrix = data.getData();
    if (data_matrix.empty() || data_matrix[0].empty())
    {
        return; // Handle empty data to avoid division by zero
    }

    double max_item = -std::numeric_limits<double>::infinity();
    double min_item = std::numeric_limits<double>::infinity();

    // Find the maximum and minimum values
    for (const auto &row : data_matrix)
    {
        for (const double &item : row)
        {
            max_item = std::max(max_item, item);
            min_item = std::min(min_item, item);
        }
    }

    // Avoid division by zero if max_item and min_item are the same
    if (max_item == min_item)
    {
        return;
    }

    // Normalize using x_iN = (x_i - min) / (max - min)
    const double range = max_item - min_item;
    for (auto &row : data_matrix)
    {
        for (double &item : row)
        {
            item = (item - min_item) / range;
        }
    }

    data.setData(data_matrix);
}

int ToolsHelper::generateRandomNumberInteger(int min, int max, std::random_device::result_type seed)
{
    // Seed the random number generator with the provided seed
    randomGenerator.seed(seed);

    return randomIntDistribution(randomGenerator, std::uniform_int_distribution<int>::param_type(min, max));
}

double ToolsHelper::generateRandomNumberDouble(double min, double max, std::random_device::result_type seed)
{
    // Seed the random number generator with the provided seed
    randomGenerator.seed(seed);

    return randomRealDistribution(randomGenerator, std::uniform_real_distribution<double>::param_type(min, max));
}

double ToolsHelper::computeEuclideanDistance(const std::vector<double> &point1, const std::vector<double> &point2, const std::vector<double> &weights)
{
    if (point1.size() != point2.size() || point1.size() != weights.size())
    {
        throw std::invalid_argument("Vector dimensions and weights must match.");
    }

    double sum = 0.0;
    for (size_t i = 0; i < point1.size(); ++i)
    {
        double diff = point1[i] - point2[i];
        sum += weights[i] * (diff * diff);
    }
    return sqrt(sum);
}

double ToolsHelper::computeAccuracy(const Data &sample, const std::vector<double> &weights)
{
    double correctlyClassifiedInstances = 0.0;
    const std::vector<std::vector<double>> &samples = sample.getData();
    const std::vector<char> &classes = sample.getLabels();
    size_t totalInstances = sample.size();

    for (size_t i = 0; i < totalInstances; ++i)
    {
        char predictedClass = MLTools::KNNClassifier(sample, samples[i], weights);

        if (predictedClass == classes[i])
            correctlyClassifiedInstances += 1.0;
    }

    return correctlyClassifiedInstances / static_cast<double>(totalInstances);
}

std::string ToolsHelper::getDatasetTitle(const int &option)
{
    switch (option)
    {
    case 1:
        return "SPECTF-Heart";
    case 2:
        return "Parkinsons";
    case 3:
        return "Ionosphere";
    default:
        return "Unknown Dataset";
    }
}
