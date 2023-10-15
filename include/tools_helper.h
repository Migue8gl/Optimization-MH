#ifndef TOOLS_HELPER_H
#define TOOLS_HELPER_H

#include <vector>
#include <string>
#include <stdexcept>
#include <set>
#include <limits>
#include <random>
#include "data.h"
#include "ml_tools.h"

/**
 * @brief Utility class for handling data, reading, and display operations.
 *
 * This class provides various utility methods for working with data, including reading data from ARFF files,
 * displaying data information, normalizing data, generating random numbers, creating data partitions,
 * and calculating Euclidean distances.
 */
class ToolsHelper
{
public:
    /**
     * @brief Converts a given string to uppercase.
     *
     * This function takes a string as input and converts all its characters to uppercase.
     *
     * @param str The input string to be converted to uppercase.
     * @return The uppercase version of the input string.
     */
    static std::string toUpperCase(const std::string &str);

    /**
     * @brief Display the contents of a data matrix and class vector in a tabular format.
     *
     * This function displays data information, including the number of features (attributes), the number of classes,
     * class labels, and data instances.
     *
     * @param data Instace of Data class, contains information about data labels and data points.
     * @param separator Optional separator between instances.
     */
    static void displayDataInfo(const Data &data, const std::string &separator = "\n");

    /**
     * @brief Normalize the data using Z-score (Standard Score) normalization.
     *
     * Z-score normalization transforms data into a standard normal distribution with a mean of 0
     * and a standard deviation of 1. This function takes a reference to a Data object and
     * normalizes its values using Z-score normalization.
     *
     * @param data A reference to the Data object to be normalized.
     */
    static void zScoreNormalize(Data &data);

    /**
     * @brief Normalize a two-dimensional data matrix to the [0, 1] range.
     *
     * This function normalizes each element in the matrix to fall within the [0, 1] range while preserving
     * the data's relative proportions.
     *
     * @param data Data to be normalized.
     */
    static void normalizeDataZeroOne(Data &data);

    /**
     * @brief Normalize a two-dimensional data matrix to the [-1, 1] range.
     *
     * This function normalizes each element in the matrix to fall within the [-1, 1] range while preserving
     * the data's relative proportions.
     *
     * @param data Data to be normalized.
     */
    static void normalizeDataMinusOneOne(Data &data);

    /**
     * @brief Generate a random uniform integer number in a given range.
     *
     * @param min Lower bound.
     * @param max Upper bound.
     * @param seed Optional seed value to initialize the random number generator.
     * @return A random integer value within the specified range [min, max].
     */
    static int generateUniformRandomNumberInteger(int min, int max, std::random_device::result_type seed = std::random_device{}());

    /**
     * @brief Generate a random uniform double number in a given range.
     *
     * @param min Lower bound.
     * @param max Upper bound.
     * @param seed Optional seed value to initialize the random number generator.
     * @return A random double value within the specified range [min, max].
     */
    static double generateUniformRandomNumberDouble(double min, double max, std::random_device::result_type seed = std::random_device{}());

    /**
     * @brief Generate a random normal number in a given range.
     *
     * @param mean The mean of the normal distribution.
     * @param stddev The standard deviation of the normal distribution.
     * @param seed Optional seed value to initialize the random number generator.
     * @return A random double value within the specified range [min, max].
     */
    static double generateNormalRandomNumber(double mean, double stddev, std::random_device::result_type seed = std::random_device{}());

    /**
     * @brief Get the dataset title based on the option.
     *
     * This function takes an integer option and returns the corresponding dataset title.
     *
     * @param option The integer option to determine the dataset title.
     * @return The dataset title as a string.
     *
     * @details
     * Valid options:
     * - 1: "SPECTF-Heart"
     * - 2: "Parkinsons"
     * - 3: "Ionosphere"
     * - Other values: "Unknown Dataset"
     */
    static std::string getDatasetTitle(const int &option);

    /**
     * @brief Display a progress bar.
     *
     * This function displays a progress bar to visualize the progress of a task.
     * The progress should be a value between 0.0 and 1.0, where 0.0 indicates no progress,
     * and 1.0 indicates the task is complete.
     *
     * @param progress The progress value between 0.0 and 1.0.
     *
     * @note https://stackoverflow.com/questions/14539867/how-to-display-a-progress-indicator-in-pure-c-c-cout-printf
     */
    static void progress_bar(float progress);
};

#endif // TOOLS_HELPER_H