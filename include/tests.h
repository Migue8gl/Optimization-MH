#ifndef TESTS_H
#define TESTS_H

#include "data.h"

/**
 * @brief Utility class for debugging and testing machine learning algorithms.
 */
class Tests
{
public:
    /**
     * @brief Test function to read and display data information.
     *
     * This function takes data matrices and class vectors as input and displays
     * information about the data using the ToolsHelper::displayDataInfo function.
     *
     * @param data Instance of Data class, contains information about data labels and data points.
     */
    static void testReadAndDisplayData(const Data &data);

    /**
     * @brief Test function for data partitioning.
     *
     * This function takes data matrices, class vectors, and an optional 'k' parameter
     * (default is 5) for data partitioning testing.
     *
     * @param data Instance of Data class, contains information about data labels and data points.
     * @param k The number of partitions for testing (default is 5).
     */
    static void testPartitions(const Data &data, int k = 5);

    /**
     * @brief Test function for the KNN classifier algorithm.
     *
     * This function takes a Data instance and a 'k' parameter (default is 1)
     * for testing the KNN classifier.
     *
     * @param data Instance of the Data class, containing information about data labels and data points.
     * @param k The number of neighbors (k) for testing the KNN classifier (default is 1).
     */
    static void testKNNClassifier(const Data &data, const unsigned int &k = 1);

    /**
     * @brief Test and print information about shuffled data.
     *
     * This function takes a Data object, shuffles its contents multiple times using
     * the specified seed value, and prints information about the data before and after each shuffle.
     *
     * @param data The Data object to be shuffled and tested.
     * @param seedValue The seed value used for random shuffling.
     * @param numShuffles The number of times to shuffle the data.
     */
    static void testShuffledData(const Data &data, int seedValue, int numShuffles);

    /**
     * @brief Print information about the given Data object.
     *
     * This function prints details about the Data object, including the number of data points,
     * the number of features per data point, and the first few data points with their labels.
     *
     * @param data The Data object to be described.
     */

    /**
     * @brief Test function to debug.
     *
     * This function takes data information and dumps all information
     * for debuggin purposes in a txt file.
     *
     * @param data Instace of Data class, contains information about data labels and data points.
     * @param dataset Representative integer of the dataset
     */
    static void runTests(const Data &data, int dataset);

private:
    /**
     * @brief Prints information about the Data.
     *
     * This static function prints various information about the provided Data object,
     * including details about data labels and data points.
     *
     * @param data The Data object containing information to be printed.
     */
    static void printDataInfo(const Data &data);
};

#endif // TESTS_H
