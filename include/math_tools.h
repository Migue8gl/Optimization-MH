#ifndef MATH_TOOLS_H
#define MATH_TOOLS_H

#include <vector>

/**
 * @brief Utility class for math calculations.
 *
 */
class MathTools
{
public:
    /**
     * @brief Calculate the Euclidean distance between two data points.
     *
     * @param point1 First data point.
     * @param point2 Second data point.
     * @return The Euclidean distance between the two data points.
     */
    static double computeEuclideanDistance(const std::vector<double> &point1, const std::vector<double> &point2, const std::vector<double> &weights);
};

#endif // MATH_TOOLS_H