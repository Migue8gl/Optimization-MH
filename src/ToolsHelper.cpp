#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <set>
#include <map>
#include <algorithm>
#include <limits>
#include <random>

class ToolHelper
{
public:
    /**
     * @brief Functionality to read data from an ARFF file and extract its data and classes
     * into two containers (a matrix for data and a vector for classes).
     * Each class value corresponds to a row in the matrix.
     *
     * @param file ARFF file to read
     * @param data_matrix Matrix with data extracted from the file
     * @param class_vector Vector with the data classes
     */
    void readDataARFF(const std::string &file, std::vector<std::vector<double>> &data_matrix, std::vector<char> &class_vector)
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
                        class_vector.push_back(tokens[i][0]); // Store the first character as a char in class_vector
                    else
                        data_row.push_back(std::stod(tokens[i])); // Store data in a temporary row vector
                }

                data_matrix.push_back(data_row); // Append the row to data_matrix
            }
        }

        if (data_matrix.empty())
        {
            std::cerr << "[ERROR] No valid data rows found in the file." << std::endl;
            exit(1);
        }
    }

    /**
     * @brief This function displays the contents of a data matrix and class vector in a tabular format
     * with proper alignment and headers. It provides information about the dataset's attributes,
     * the number of unique classes, and the data instances.
     *
     * @param data_matrix   Matrix with data to display
     * @param class_vector  Vector with class labels to display
     */
    void displayDataInfo(const std::vector<std::vector<double>> &data_matrix, const std::vector<char> &class_vector, const std::string &separator = "\n")
    {
        if (data_matrix.empty() || class_vector.empty())
        {
            std::cout << "No data to display." << std::endl;
            return;
        }

        const size_t numAttributes = data_matrix.empty() ? 0 : data_matrix[0].size();

        // Display dataset information
        std::cout << "Dataset Information:" << std::endl;
        std::cout << "Number of Features (Attributes): " << numAttributes << std::endl;

        // Extract unique class labels
        std::set<char> uniqueClasses(class_vector.begin(), class_vector.end());

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
            std::cout << "Instance [" << i + 1 << "] ->"; // Numerical index enclosed in square brackets
            for (size_t j = 0; j < numAttributes; ++j)
            {
                std::cout << std::setw(12) << data_matrix[i][j];
            }
            std::cout << std::setw(12) << class_vector[i] << std::endl;
            if (i != data_matrix.size() - 1)
            {
                std::cout << separator << std::endl;
            }
        }
    }

    /**
     * @brief This function normalizes a two-dimensional data matrix to the [0, 1] range.
     * Each element in the matrix is scaled so that it falls within this range while
     * preserving the data's relative proportions.
     *
     * @param data Data to be nornalized
     *
     * @note If the input matrix is empty or contains empty rows, no normalization is performed.
     *       If the maximum and minimum values in the data are the same, no normalization is applied
     *       to avoid division by zero.
     */
    void normalizeData(std::vector<std::vector<double>> &data)
    {
        if (data.empty() || data[0].empty())
        {
            return; // Handle empty data to avoid division by zero
        }

        double max_item = -std::numeric_limits<double>::infinity();
        double min_item = std::numeric_limits<double>::infinity();

        // Find the maximum and minimum values
        for (const auto &row : data)
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
        for (auto &row : data)
        {
            for (double &item : row)
            {
                item = (item - min_item) / range;
            }
        }
    }

    /**
     * @brief Function to generate a random int number in a given range
     *
     * @param min Lower bound
     * @param max Upper bound
     * @param seed seed An optional seed value to initialize the random number generator.
     *             If not provided, a random seed is generated using std::random_device.
     * @return int A random integer value within the specified range [min, max].
     */
    int generateRandomNumberInteger(int min, int max, std::random_device::result_type seed = std::random_device{}())
    {
        std::random_device rd;
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> distribution(min, max);

        return distribution(gen);
    }

    /**
     * @brief Function to generate a random double number in a given range
     *
     * @param min Lower bound
     * @param max Upper bound
     * @param seed seed An optional seed value to initialize the random number generator.
     *             If not provided, a random seed is generated using std::random_device.
     * @return int A random double value within the specified range [min, max].
     */
    double generateRandomNumberDouble(double min, double max, std::random_device::result_type seed = std::random_device{}())
    {
        std::random_device rd;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> distribution(min, max);

        return distribution(gen);
    }

    /**
     * @brief Create Partitions of Data and Class Labels
     *
     * This function takes a data matrix and corresponding class labels, and it creates
     * partitions of the data while preserving the class labels. The data is shuffled
     * before partitioning to ensure randomness.
     *
     * @param data_matrix The input data matrix represented as a vector of vectors of doubles.
     * @param class_vector The corresponding class labels represented as a vector of chars.
     * @param k The number of partitions to create. It should be greater than 0 and less than or equal to the number of data samples.
     *
     * @throws std::invalid_argument If k is not within a valid range.
     *
     * @return A pair containing two vectors:
     *   - The first vector is a vector of vectors of doubles representing the data partitions.
     *   - The second vector is a vector of vectors of chars representing the class labels corresponding to the data partitions.
     */
    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<char>>> createPartitions(
        const std::vector<std::vector<double>> &data_matrix,
        const std::vector<char> &class_vector,
        int k)
    {
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
        std::vector<std::vector<std::vector<double>>> partitions(k);
        std::vector<std::vector<char>> partitions_classes(k);

        // Fill partitions while preserving classes
        for (size_t i = 0; i < shuffled_data.size(); ++i)
        {
            partitions[i % k].push_back(shuffled_data[i].first);
            partitions_classes[i % k].push_back(shuffled_data[i].second);
        }

        return std::make_pair(partitions, partitions_classes);
    }
};
