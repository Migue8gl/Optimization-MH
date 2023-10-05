#ifndef ML_ALGORITHMS_H
#define ML_ALGORITHMS_H

#include <vector>
#include "Data.h" // You should include the appropriate header for the Data class here

/**
 * @brief Utility class containing machine learning algorithms.
 */
class MLAlgorithms
{
public:
    /**
     * @brief Perform k-Nearest Neighbors (k-NN) classification to predict the class of an element.
     *
     * This function uses the k-Nearest Neighbors algorithm to predict the class of a given element
     * based on the provided dataset.
     *
     * @param data An instance of the Data class containing information about data labels and data points.
     * @param element The data point for which the class is to be predicted.
     * @param k The number of nearest neighbors to consider for classification (default is 1).
     * @return The predicted class for the given element.
     *
     * @throws std::invalid_argument if an invalid value of k is provided.
     */
    static char KNNClassifier(const Data &data, const std::vector<double> &element, int k = 1);
};

#endif // ML_ALGORITHMS_H
