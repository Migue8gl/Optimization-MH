#ifndef OPTIMIZERS_H
#define OPTIMIZERS_H

#include <vector>
#include "data.h"
#include <map>

/**
 * @brief A collection of optimization algorithms for machine learning.
 */
class Optimizers
{
public:
    // Define an optimization function type for machine learning algorithms
    using Optimizer = std::vector<double> (*)(const Data &, std::vector<double> &, std::map<std::string, std::string> &);

    /**
     * @brief Retrieve weights for k-Nearest Neighbors classification.
     *
     * The k-Nearest Neighbors (kNN) classifier typically does not use custom weights
     * for making predictions. This function returns a vector of weights, with each weight set to 1,
     * as they don't affect kNN's standard behavior.
     *
     * @param data An instance of the Data class containing data labels and data points.
     * @param weights A vector to store the weights (ignored for kNN).
     * @param hyperParams A map of hyperparameters (ignored for kNN).
     * @return std::vector<double> The weights for each data point (all set to 1 for kNN).
     */
    static std::vector<double> knn(const Data &data, std::vector<double> &weights, std::map<std::string, std::string> &hyperParams);

    /**
     * @brief Perform local search optimization on a given dataset.
     *
     * This function applies a local search optimization algorithm to find an optimal
     * set of weights for a given dataset. It iteratively explores neighboring solutions to
     * maximize a specified objective function.
     *
     * @param data The dataset represented as an instance of the Data class.
     * @param weights A vector of initial weights.
     * @param hyperParams A map of hyperparameters (optional).
     * @return A vector of double values representing the optimized weights for the dataset.
     */
    static std::vector<double> localSearch(const Data &data, std::vector<double> &weights, std::map<std::string, std::string> &hyperParams);

    /**
     * @brief Perform a strong local search optimization on a given dataset.
     *
     * This function applies a strong local search optimization algorithm to find an optimal
     * set of weights for a given dataset. It iteratively explores neighboring solutions to
     * maximize a specified objective function by adjusting multiple weights simultaneously.
     *
     * @param data The dataset represented as an instance of the Data class.
     * @param weights A vector of initial weights.
     * @param hyperParams A map of hyperparameters (optional).
     * @return A vector of double values representing the optimized weights for the dataset.
     *
     * @note The strong local search explores a broader range of weight adjustments and may converge faster
     * or escape local optima. Experiment with hyperparameters for the best performance on your specific problem.
     */
    static std::vector<double> localSearchStrong(const Data &data, std::vector<double> &weights, std::map<std::string, std::string> &hyperParams);

    /**
     * @brief Apply the Monarch Butterfly Optimization (MBO) algorithm to optimize weights.
     *
     * This static function performs the Monarch Butterfly Optimization (MBO) algorithm on the given data.
     *
     * @param data The data object containing the dataset and labels.
     * @param weights A vector of initial weights.
     * @param hyperParams A map of hyperparameters (optional).
     * @return The optimized solution found by the algorithm.
     */
    static std::vector<double> mbo(const Data &data, std::vector<double> &weights, std::map<std::string, std::string> &hyperParams);

    /**
     * @brief Apply the Simulated Annealing algorithm to optimize a set of weights.
     *
     * This static function applies the Simulated Annealing algorithm to optimize a set of
     * weights for a given dataset using the specified hyperparameters.
     *
     * @param data The dataset on which to perform optimization.
     * @param weights A vector of initial weights.
     * @param hyperParams A map of hyperparameters (optional).
     * @return The optimized set of weights determined by the Simulated Annealing algorithm.
     */
    static std::vector<double> simulatedAnnealing(const Data &data, std::vector<double> &weights, std::map<std::string, std::string> &hyperParams);

private:
    /**
     * @brief Generate a random number using the Levy flight distribution.
     *
     * This static function generates a random number using the Levy flight distribution with specified parameters.
     *
     * @return The generated random number as a vector of double values.
     */
    static std::vector<double> levyFlight(const std::vector<double> &butterfly, double alpha);

    /**
     * @brief Perform migration of subpopulations between mariposas.
     *
     * This static function performs migration between two subpopulations of mariposas.
     *
     * @param subpob1 The first subpopulation represented as a vector of vectors.
     * @param subpob2 The second subpopulation represented as a vector of vectors.
     * @param period The migration period.
     * @param p The migration probability.
     */
    static void migration(std::vector<std::vector<double>> &subpob1, std::vector<std::vector<double>> &subpob2, double period, double p);

    /**
     * @brief Adjust the subpopulations based on fitness values.
     *
     * This static function adjusts the subpopulations based on fitness values and other parameters.
     *
     * @param subpob1 The first subpopulation represented as a vector of vectors.
     * @param subpob2 The second subpopulation represented as a vector of vectors.
     * @param bestButterfly The best butterfly from the population.
     * @param p The adjustment probability.
     * @param BAR The threshold for adjustment.
     * @param alpha The adjustment parameter.
     */
    static void adjust(std::vector<std::vector<double>> &subpob2, const std::vector<double> &bestButterfly, double p, double BAR, double alpha);

    /**
     * @brief Implement elitism by selecting and preserving the best individuals in the population.
     *
     * This function selects the top-performing individuals (butterflies) based on their fitness
     * and replaces the worst-performing individuals with the selected elite individuals.
     *
     * @param data The data used for evaluating the fitness of individuals.
     * @param np The population of individuals represented as a vector of vectors.
     * @param numElite The number of top-performing individuals to preserve as elite.
     *
     * @return The population with the worst individuals replaced by the elite individuals.
     */
    static std::vector<std::vector<double>> elitism(const Data &data, std::vector<std::vector<double>> &np, unsigned int numElite);
};

#endif // OPTIMIZERS_H