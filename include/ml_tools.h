#ifndef ML_TOOLS_H
#define ML_TOOLS_H

#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include "ml_tools.h"
#include "tools_helper.h"
#include <chrono>
#include <iomanip>
#include <thread>
#include "seed.h"
#include <future>
#include "data.h"
#include "optimizers.h"

/**
 * @brief Utility class containing machine learning algorithms.
 */
class MLTools
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
     * @param weights The weights to apply to each dimension when calculating distances.
     * @param k The number of nearest neighbors to consider for classification (default is 1).
     * @return char The predicted class for the given element.
     *
     * @throws std::invalid_argument if an invalid value of k is provided.
     */
    static char kNNClassifier(const Data &data, const std::vector<double> &element, const std::vector<double> &weigths, const unsigned int &k = 1);

    /**
     * @brief Perform k-fold cross-validation.
     *
     * This method performs k-fold cross-validation for a given optimizer algorithm
     * on the provided dataset. Cross-validation assesses the model's performance by
     * dividing the dataset into 'numberPartitions' parts, training and testing 'numberPartitions'
     * times with different partitions as the test set each time.
     *
     * @param data The dataset for cross-validation.
     * @param optimizer The optimizer for training and testing.
     * @param numberPartitions The number of partitions (folds) for cross-validation. Default is 5.
     * @param hyperParams A map of hyperparameters.
     */
    static void kCrossValidation(const Data &data, const Optimizers::Optimizer &optimizer, const int numberPartitions, std::map<std::string, std::string> &hyperParams);

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
     * @param alpha The alpha value (default: 0.5) for combining classification and reduction rates.
     * @return The computed fitness value.
     */
    static double computeFitness(const Data &data, std::vector<double> &weights, const double &alpha = 0.5);

    /**
     * @brief Calculate the fitness values for a population of weight vectors.
     *
     * This function calculates the fitness values for a population of weight vectors based on the provided data.
     *
     * @param data The data object containing the dataset and labels.
     * @param populationWeights A vector of weight vectors representing the population.
     * @param alpha The alpha value (default: 0.5) for combining classification and reduction rates.
     * @return A vector of computed fitness values for the population.
     */
    static std::vector<double> computePopulationFitness(const Data &data, std::vector<std::vector<double>> &populationWeights, const double &alpha = 0.5);
};

#endif // ML_TOOLS_H
