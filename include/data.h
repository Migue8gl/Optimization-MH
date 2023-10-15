#ifndef DATA_H
#define DATA_H

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

/**
 * @brief Utility class for storage and handling data.
 */
class Data
{
private:
    std::vector<std::vector<double>> data; // The data storage.
    std::vector<char> labels;              // The labels associated with data.
    double mean;
    double variance;

public:
    /**
     * @brief Default constructor for Data class
     */
    Data();

    /**
     * @brief Constructor for Data class.
     * @param data The initial data to store.
     * @param labels The initial labels for the data.
     */
    Data(std::vector<std::vector<double>> data, std::vector<char> labels);

    /**
     * @brief Destructor for Data class.
     */
    ~Data();

    // Accessors

    /**
     * @brief Get the data stored in this object.
     * @return A constant reference to the data.
     */
    const std::vector<std::vector<double>> &getData() const;

    /**
     * @brief Get the labels associated with the data.
     * @return A constant reference to the labels.
     */
    const std::vector<char> &getLabels() const;

    /**
     * @brief Get the mean value of the data stored in this object.
     * @return The mean value.
     */
    const double &getMean() const;

    /**
     * @brief Get the variance of the data stored in this object.
     * @return The variance value.
     */
    const double &getVariance() const;

    /**
     * @brief Compute the mean value of the data stored in this object.
     */
    void computeMean();

    /**
     * @brief Compute the variance of the data stored in this object.
     */
    void computeVariance();

    // Mutators

    /**
     * @brief Set the data for this object.
     * @param newData The new data to store.
     * @return A reference to this object.
     */
    Data &setData(const std::vector<std::vector<double>> &newData);

    /**
     * @brief Set the labels for this object.
     * @param newLabels The new labels to associate with the data.
     * @return A reference to this object.
     */
    Data &setLabels(const std::vector<char> &newLabels);

    /**
     * @brief Add a new data point and label to the object.
     * @param newDataPoint The new data point to add.
     * @param newLabel The label associated with the new data point.
     */
    void addDataPoint(const std::vector<double> &newDataPoint, char newLabel);

    /**
     * @brief Clear all data and labels from this object.
     */
    void clearData();

    /**
     * @brief Get the number of data points stored in this object.
     * @return The number of data points.
     */
    unsigned int size() const;

    /**
     * @brief Read data from an ARFF file and extract its data and classes.
     *
     * This function reads data from an ARFF file and extracts the data into a matrix and the classes into a vector.
     *
     * @param file The ARFF file to read.
     * @param data Instace of Data class, contains information about data labels and data points.
     */
    void readDataARFF(const std::string &file);

    /**
     * @brief Create partitions of data and class labels.
     *
     * This function partitions one data instance k partitions.
     * The data is shuffled before partitioning to ensure randomness.
     *
     * @param k The number of partitions to create.
     * @throws std::invalid_argument If k is not within a valid range.
     * @return A vector containing data partitions.
     */
    std::vector<Data> createPartitions(int k) const;
};

#endif // DATA_H
