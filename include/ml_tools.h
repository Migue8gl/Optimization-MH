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
    static char KNNClassifier(const Data &data, const std::vector<double> &element, const std::vector<double> &weigths, const unsigned int &k = 1);

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
    static void kCrossValidation(const Data &data, const MLTools::Optimizer &optimizer, const int numberPartitions = 5, const std::string &opt = "");

    /**
     * @brief Perform local search optimization on a given dataset.
     *
     * This function applies a local search optimization algorithm to find an optimal
     * set of weights for a given dataset. It starts with an initial set of weights and
     * iteratively explores neighboring solutions to maximize a specified objective function.
     *
     * @param data The dataset represented as an instance of the Data class.
     * @param maxIter The maximum number of iterations for the optimization (default: 15000).
     * @param maxNeighbour The maximum number of neighbors to explore during each iteration
     *                     (default: 0, which is determined as twice the size of the input data).
     * @return A vector of double values representing the optimized weights for the dataset.
     *
     */
    static std::vector<double> localSearch(const Data &data, const std::string &opt = "");

private:
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

    /**
     * @brief Compute the fitness value based on the given data and weights.
     *
     * This function calculates the fitness value based on the provided data and weights.
     *
     * @param data The data object containing the dataset and labels.
     * @param weights The vector of weights to be used for fitness computation.
     * @return The computed fitness value.
     */
    static double computeFitness(const Data &data, std::vector<double> &weights, const double &alpha = 0.5);
};

#endif // ML_TOOLS_H
