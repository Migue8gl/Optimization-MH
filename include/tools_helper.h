#ifndef TOOLS_HELPER_H
#define TOOLS_HELPER_H

#include <vector>
#include <string>
#include <stdexcept>
#include <set>
#include <limits>
#include <random>
#include "data.h"

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
     * @brief Normalize a two-dimensional data matrix to the [0, 1] range.
     *
     * This function normalizes each element in the matrix to fall within the [0, 1] range while preserving
     * the data's relative proportions.
     *
     * @param data Data to be normalized.
     */
    static void normalizeData(Data &data);

    /**
     * @brief Generate a random integer number in a given range.
     *
     * @param min Lower bound.
     * @param max Upper bound.
     * @param seed Optional seed value to initialize the random number generator.
     * @return A random integer value within the specified range [min, max].
     */
    static int generateRandomNumberInteger(int min, int max, std::random_device::result_type seed = std::random_device{}());

    /**
     * @brief Generate a random double number in a given range.
     *
     * @param min Lower bound.
     * @param max Upper bound.
     * @param seed Optional seed value to initialize the random number generator.
     * @return A random double value within the specified range [min, max].
     */
    static double generateRandomNumberDouble(double min, double max, std::random_device::result_type seed = std::random_device{}());

    /**
     * @brief Calculate the Euclidean distance between two data points.
     *
     * @param point1 First data point.
     * @param point2 Second data point.
     * @return The Euclidean distance between the two data points.
     */
    static double computeEuclideanDistance(const std::vector<double> &point1, const std::vector<double> &point2, const std::vector<double> &weights);

    /**
     * @brief Compute the accuracy of a classifier on a given sample.
     *
     * This function calculates the accuracy of a classifier on a provided sample
     * using the specified weights.
     *
     * @param sample The sample for which accuracy is computed.
     * @param weights The weights used by the classifier.
     *
     * @return The accuracy as a double value, ranging from 0.0 (0% accuracy) to 1.0 (100% accuracy).
     */
    static double computeAccuracy(const Data &sample, const std::vector<double> &weights);

    static void execute(const std::vector<Data> &partitions, const std::string &option);

private:
    // Static random number generator and distribution
    static std::mt19937 randomGenerator;
    static std::uniform_int_distribution<int> randomIntDistribution;
    static std::uniform_real_distribution<double> randomRealDistribution;
};

#endif // TOOLS_HELPER_H