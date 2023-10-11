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
     * @param opt An optional string parameter for specifying optimization settings (default: empty string).).
     * @return A vector of double values representing the optimized weights for the dataset.
     *
     */
    static std::vector<double> localSearch(const Data &data, const std::string &opt = "");

    /**
     * @brief Perform the Monarch Butterfly Optimization (MBO) algorithm.
     *
     * This static function performs the Monarch Butterfly Optimization (MBO) algorithm on the given data.
     *
     * @param data The data object containing the dataset and labels.
     * @param ls Flag indicating whether to perform local search.
     * @return The optimized solution found by the algorithm.
     */
    static std::vector<double> mbo(const Data &data, bool ls);

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
     * @param alpha The alpha value (default: 0.5) for combining classification and reduction rates.
     * @return The computed fitness value.
     */
    static double computeFitness(const Data &data, std::vector<double> weights, const double &alpha = 0.5);

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
    static std::vector<double> computePopulationFitness(const Data &data, std::vector<std::vector<double>> populationWeights, const double &alpha = 0.5);

    /**
     * @brief Generate a random number using the Levy flight distribution.
     *
     * This static function generates a random number using the Levy flight distribution with specified parameters.
     *
     * @return The generated random number.
     */
    static double levyFlight();

    /**
     * @brief Perform migration of subpopulations between mariposas.
     *
     * This static function performs migration between two subpopulations of mariposas.
     *
     * @param subpob1 The first subpopulation.
     * @param subpob2 The second subpopulation.
     * @param period The migration period.
     * @param p The migration probability.
     */
    static void migration(std::vector<std::vector<double>> &subpob1, std::vector<std::vector<double>> &subpob2, double period, double p);

    /**
     * @brief Adjust the subpopulations based on fitness values.
     *
     * This static function adjusts the subpopulations based on fitness values and other parameters.
     *
     * @param subpob1 The first subpopulation.
     * @param subpob2 The second subpopulation.
     * @param indexbest1 The indices of the best mariposas in subpob1.
     * @param indexbest2 The indices of the best mariposas in subpob2.
     * @param p The adjustment probability.
     * @param fitnessPopulation The fitness values of the population.
     * @param BAR The threshold for adjustment.
     * @param alpha The adjustment parameter.
     */
    static void adjust(std::vector<std::vector<double>> &subpob1, std::vector<std::vector<double>> &subpob2, const std::vector<int> &indexbest1, const std::vector<int> &indexbest2, double p, const std::vector<double> &fitnessPopulation, double BAR, double alpha);
};

#endif // ML_TOOLS_H
