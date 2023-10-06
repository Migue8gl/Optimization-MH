#ifndef TOOLSHELPER_H
#define TOOLSHELPER_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <set>
#include <algorithm>
#include <limits>
#include <random>
#include "Data.h"

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
     * @brief Read data from an ARFF file and extract its data and classes.
     *
     * This function reads data from an ARFF file and extracts the data into a matrix and the classes into a vector.
     *
     * @param file The ARFF file to read.
     * @param data Instace of Data class, contains information about data labels and data points.
     */
    static Data readDataARFF(const std::string &file);

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
     * @brief Create partitions of data and class labels.
     *
     * This function partitions one data instance k partitions.
     * The data is shuffled before partitioning to ensure randomness.
     *
     * @param data Instace of Data class, contains information about data labels and data points.
     * @param k The number of partitions to create.
     * @throws std::invalid_argument If k is not within a valid range.
     * @return A vector containing data partitions.
     */
    static std::vector<Data> createPartitions(const Data &data, int k);

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

#endif // TOOLSHELPER_H