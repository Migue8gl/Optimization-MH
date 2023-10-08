#ifndef ML_TOOLS_H
#define ML_TOOLS_H

#include <vector>
#include "data.h" // You should include the appropriate header for the Data class here

/**
 * @brief Utility class containing machine learning algorithms.
 */
class MLTools
{
public:
    // Define a optimization function type for the ml algorithm
    using Optimizer = std::vector<double> (*)(const Data &, const std::string &);

    /**
     * @brief Perform k-Nearest Neighbors (k-NN) classification to predict the class of an element.
     *
     * This function uses the k-Nearest Neighbors algorithm to predict the class of a given element
     * based on the provided dataset.
     *
     * @param data An instance of the Data class containing information about data labels and data points.
     * @param element The data point for which the class is to be predicted.
     * @param weights The weights to apply to each dimension when calculating distances.
     * @param k The number of nearest neighbors to consider for classification (default is 1).
     * @return char The predicted class for the given element.
     *
     * @throws std::invalid_argument if an invalid value of k is provided.
     */
    static char KNNClassifier(const Data &data, const std::vector<double> &element, const std::vector<double> &weigths, int k = 1);

    /**
     * @brief Function that returns weights to perfom k-Nearest Neighbors classification
     *
     * kNN classifier does not use weights to make his predictions. Due to that, weights returned are all 1's.
     *
     * @param data An instance of the Data class containing information about data labels and data points.
     * @return std::vector<double> The weights for each of the data points
     */
    static std::vector<double> KNN(const Data &data, const std::string &opt = "");

    /**
     * @brief Perform k-fold cross-validation.
     *
     * This method performs k-fold cross-validation for a given optimizer algorithm
     * on the provided dataset. Cross-validation is used to assess the model's performance
     * by dividing the dataset into 'numberPartitions' parts and training/testing the model
     * 'numberPartitions' times, each time using a different part as the test set and the
     * remaining parts as the training set.
     *
     * @param data The dataset on which cross-validation will be performed.
     * @param optimizer The optimizer to be used for training and testing.
     * @param numberPartitions The number of partitions (folds) for cross-validation. Default is 5.
     * @param opt Additional options or parameters for the optimizer. Default is an empty string.
     *
     * @details
     * Cross-validation is a common technique to assess the generalization performance of a
     * machine learning model. It helps in estimating how well the model will perform on
     * unseen data by evaluating its performance on multiple subsets of the dataset.
     *
     */
    static void kCrossValidation(const Data &data, MLTools::Optimizer, const int numberPartitions = 5, const std::string &opt = "");
};

#endif // ML_TOOLS_H
