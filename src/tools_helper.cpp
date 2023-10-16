#include "tools_helper.h"
#include "ml_tools.h"
#include <cctype>

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

void ToolsHelper::zScoreNormalize(Data &data)
{
    double mean = data.getMean();
    double stdDev = std::sqrt(data.getVariance());

    std::vector<std::vector<double>> sample = data.getData();

    for (std::vector<double> &row : sample)
    {
        for (double &value : row)
            value = (value - mean) / stdDev;
    }

    data.setData(sample);
}

void ToolsHelper::normalizeDataZeroOne(Data &data)
{
    std::vector<std::vector<double>> dataMatrix = data.getData();
    if (dataMatrix.empty() || dataMatrix[0].empty())
    {
        return; // Handle empty data to avoid division by zero
    }

    double maxItem = -std::numeric_limits<double>::infinity();
    double minItem = std::numeric_limits<double>::infinity();

    // Find the maximum and minimum values
    for (const auto &row : dataMatrix)
    {
        for (const double &item : row)
        {
            maxItem = std::max(maxItem, item);
            minItem = std::min(minItem, item);
        }
    }

    // Avoid division by zero if max_item and min_item are the same
    if (maxItem == minItem)
    {
        return;
    }

    // Normalize using x_iN = (x_i - min) / (max - min)
    const double range = maxItem - minItem;
    for (auto &row : dataMatrix)
    {
        for (double &item : row)
        {
            item = (item - minItem) / range;
        }
    }

    data.setData(dataMatrix);
    data.computeMean();
    data.computeVariance();
}

void ToolsHelper::normalizeDataMinusOneOne(Data &data)
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
            item = (2 * (item - min_item) / range) - 1;
        }
    }

    data.setData(data_matrix);
    data.computeMean();
    data.computeVariance();
}

int ToolsHelper::generateUniformRandomNumberInteger(int min, int max, std::random_device::result_type seed)
{
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<> distribution(min, max);

    // std::cout << distribution(generator) << " -- Min: " << min << " Max: " << max << std::endl;

    return distribution(generator);
}

double ToolsHelper::generateUniformRandomNumberDouble(double min, double max, std::random_device::result_type seed)
{
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(min, max);

    return distribution(generator);
}

double ToolsHelper::generateNormalRandomNumber(double mean, double stddev, std::random_device::result_type seed)
{
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(mean, stddev);

    return distribution(generator);
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

void ToolsHelper::progress_bar(float progress)
{
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

std::vector<double> ToolsHelper::initializeWeights(const int &size)
{
    std::vector<double> w(size);
    std::generate(w.begin(), w.end(), [&]()
                  { return ToolsHelper::generateNormalRandomNumber(0.0, 1.0, Seed::getInstance().getSeed()); });
    return w;
}
