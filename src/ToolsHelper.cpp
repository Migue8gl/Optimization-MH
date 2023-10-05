#include "ToolsHelper.h"

// Define and initialize the static members
std::mt19937 ToolsHelper::randomGenerator;
std::uniform_int_distribution<int> ToolsHelper::randomIntDistribution;
std::uniform_real_distribution<double> ToolsHelper::randomRealDistribution;

Data ToolsHelper::readDataARFF(const std::string &file)
{
    std::ifstream ifile(file);
    if (!ifile)
    {
        std::cerr << "[ERROR] Couldn't open the file" << std::endl;
        std::cerr << "[Ex.] Are you sure you are in the correct path?" << std::endl;
        exit(1);
    }

    std::string line;
    int num_attributes = 0;
    bool reading_data = false;
    std::vector<std::vector<double>> data;
    std::vector<char> labels;

    while (std::getline(ifile, line))
    {
        // Trim leading and trailing whitespace
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

        if (line.empty())
            continue;

        if (line == "@data")
        {
            reading_data = true;
            continue;
        }

        if (!reading_data && line.compare(0, 10, "@attribute") == 0)
            num_attributes++;

        if (reading_data)
        {
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;

            while (std::getline(iss, token, ','))
                tokens.push_back(token);

            if (tokens.size() != num_attributes)
            {
                std::cerr << "[WARNING] Skipping inconsistent data line: " << line << std::endl;
                continue; // Skip inconsistent rows
            }

            std::vector<double> data_row;
            for (size_t i = 0; i < tokens.size(); i++)
            {
                if (i == tokens.size() - 1)
                    labels.push_back(tokens[i][0]); // Store the first character as a char in class_vector
                else
                    data_row.push_back(std::stod(tokens[i])); // Store data in a temporary row vector
            }

            data.push_back(data_row); // Append the row to data_matrix
        }
    }

    if (data.empty())
    {
        std::cerr << "[ERROR] No valid data rows found in the file." << std::endl;
        exit(1);
    }

    return Data(data, labels);
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

    const size_t numAttributes = data.size();

    // Display dataset information
    std::cout << "Dataset Information:" << std::endl;
    std::cout << "Number of Features (Attributes): " << numAttributes << std::endl;

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
    for (size_t i = 0; i < data_matrix.size(); ++i)
    {
        std::cout << "Instance [" << i + 1 << "] -> "; // Numerical index enclosed in square brackets
        for (size_t j = 0; j < numAttributes; ++j)
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

std::vector<Data> ToolsHelper::createPartitions(const Data &data, int k)
{
    const std::vector<std::vector<double>> &data_matrix = data.getData();
    const std::vector<char> &class_vector = data.getLabels();

    if (k <= 0 || k > static_cast<int>(data_matrix.size()))
    {
        throw std::invalid_argument("Invalid value of k.");
    }

    // Shuffle the data and class vectors together
    std::vector<std::pair<std::vector<double>, char>> shuffled_data;
    for (size_t i = 0; i < data_matrix.size(); ++i)
    {
        shuffled_data.push_back(std::make_pair(data_matrix[i], class_vector[i]));
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), gen);

    // Initialize partitions
    std::vector<Data> partitions(k);

    // Fill partitions while preserving classes
    for (size_t i = 0; i < shuffled_data.size(); ++i)
    {
        partitions[i % k].addDataPoint(shuffled_data[i].first, shuffled_data[i].second);
    }

    return partitions;
}

double ToolsHelper::calculateEuclideanDistance(const std::vector<double> &point1, const std::vector<double> &point2)
{
    double sum = 0.0;
    for (size_t i = 0; i < point1.size(); ++i)
    {
        double diff = point1[i] - point2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}
