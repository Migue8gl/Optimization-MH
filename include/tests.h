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
    static void testKNNClassifier(const Data &data, int k = 1);
};

#endif // TESTS_H
