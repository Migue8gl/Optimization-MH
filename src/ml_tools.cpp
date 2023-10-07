#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>
#include "ml_tools.h"
#include "tools_helper.h"

char MLTools::KNNClassifier(const Data &data, const std::vector<double> &element, const std::vector<double> &weigths, int k)
{
    {
        size_t dataSize = data.size();
        std::vector<std::vector<double>> dataMatrix = data.getData();
        std::vector<char> dataLabels = data.getLabels();

        if (k <= 0 || k > dataSize)
        {
            throw std::invalid_argument("Invalid value of k");
        }

        std::vector<std::pair<double, char>> distancesAndClasses;

        for (size_t i = 0; i < dataSize; ++i)
        {
            if (element != dataMatrix[i])
            { // Skip the same element
                double distance = ToolsHelper::computeEuclideanDistance(element, dataMatrix[i], weigths);
                distancesAndClasses.emplace_back(distance, dataLabels[i]);
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
}