#include "math_tools.h"
#include <math.h>
#include <stdexcept>


double MathTools::computeEuclideanDistance(const std::vector<double> &point1, const std::vector<double> &point2, const std::vector<double> &weights)
{
    if (point1.size() != point2.size() || point1.size() != weights.size())
    {
        throw std::invalid_argument("Vector dimensions and weights must match.");
    }

    double sum = 0.0;

    for (std::vector<double>::const_iterator it1 = point1.begin(), it2 = point2.begin(), itWeights = weights.begin(); it1 != point1.end(); ++it1, ++it2, ++itWeights)
    {
        double diff = *it1 - *it2;
        sum += *itWeights * (diff * diff);
    }

    return std::sqrt(sum);
}