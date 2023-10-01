#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include "ToolsHelper.cpp"

class MLAlgorithms
{
public:
    /**
     * @brief Perform k-Nearest Neighbors (k-NN) classification to predict the class of an element.
     *
     * This function uses the k-Nearest Neighbors algorithm to predict the class of a given element based on the provided dataset.
     *
     * @param data_matrix The dataset containing data points.
     * @param element The data point for which the class is to be predicted.
     * @param class_vector The corresponding class labels for the dataset.
     * @param k The number of nearest neighbors to consider for classification.
     * @return The predicted class for the given element.
     *
     * @throws std::invalid_argument if an invalid value of k is provided.
     */
    static char KNNClassifier(const std::vector<std::vector<double>> &data_matrix,
                              const std::vector<double> &element,
                              const std::vector<char> &class_vector, int k)
    {
        if (k <= 0 || k > data_matrix.size())
        {
            throw std::invalid_argument("Invalid value of k");
        }

        std::vector<std::pair<double, char>> distancesAndClasses;

        for (size_t i = 0; i < data_matrix.size(); ++i)
        {
            if (element != data_matrix[i])
            { // Skip the same element
                double distance = ToolsHelper::calculateEuclideanDistance(element, data_matrix[i]);
                distancesAndClasses.emplace_back(distance, class_vector[i]);
            }
        }

        // Sort distancesAndClasses by distance (ascending order)
        std::sort(distancesAndClasses.begin(), distancesAndClasses.end(), [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        // Count the occurrences of each class among the k nearest neighbors
        std::unordered_map<char, int> classCounts;
        for (int i = 0; i < k; ++i)
        {
            char cls = distancesAndClasses[i].second;
            classCounts[cls]++;
        }

        // Find the class with the highest count (mode)
        char predictedClass = '\0';
        int maxCount = -1;
        for (const auto &pair : classCounts)
        {
            if (pair.second > maxCount)
            {
                predictedClass = pair.first;
                maxCount = pair.second;
            }
        }

        return predictedClass;
    }
};