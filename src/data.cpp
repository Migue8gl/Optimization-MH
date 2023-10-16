#include "data.h"
#include "seed.h"
#include <limits>

Data::Data() : data(), labels(), mean(std::numeric_limits<double>::quiet_NaN()), variance(std::numeric_limits<double>::quiet_NaN()) {}

Data::Data(std::vector<std::vector<double>> data, std::vector<char> labels)
{
    this->labels = labels;
    this->data = data;
    this->computeMean();
    this->computeVariance();
}

Data::~Data() {}

const std::vector<std::vector<double>> &Data::getData() const
{
    return this->data;
}

void Data::computeMean()
{
    unsigned int totalElements = 0.0;

    double sum = 0.0;

    for (const std::vector<double> &innerVector : this->getData())
    {
        for (double value : innerVector)
        {
            sum += value;
            totalElements++;
        }
    }

    this->mean = sum / static_cast<double>(totalElements);
}

void Data::computeVariance()
{
    unsigned int totalElements = 0.0;

    double sum = 0.0;
    double squaredSum = 0.0;

    for (const std::vector<double> &innerVector : this->getData())
    {
        for (double value : innerVector)
        {
            sum += value;
            squaredSum += value * value;
            totalElements++;
        }
    }

    double mean = sum / static_cast<double>(totalElements);
    double meanOfSquares = squaredSum / static_cast<double>(totalElements);

    this->variance = meanOfSquares - (mean * mean);
}

const double &Data::getVariance() const
{
    return this->variance;
}

const double &Data::getMean() const
{
    return this->mean;
}

const std::vector<char> &Data::getLabels() const
{
    return this->labels;
}

Data &Data::setData(const std::vector<std::vector<double>> &newData)
{
    this->data = newData;
    return *this;
}

Data &Data::setLabels(const std::vector<char> &newLabels)
{
    this->labels = newLabels;
    return *this;
}

void Data::addDataPoint(const std::vector<double> &newDataPoint, char newLabel)
{
    this->data.push_back(newDataPoint);
    this->labels.push_back(newLabel);

    this->computeMean();
    this->computeVariance();
}

void Data::clearData()
{
    this->data.clear();
    this->labels.clear();
    this->variance = std::numeric_limits<double>::quiet_NaN();
    this->mean = std::numeric_limits<double>::quiet_NaN();
}

unsigned int Data::size() const
{
    return this->data.size();
}

unsigned int Data::parameterSize() const
{
    return this->data[0].size();
}

void Data::readDataARFF(const std::string &file)
{
    std::ifstream ifile(file);
    if (!ifile)
    {
        std::cerr << "[ERROR] Couldn't open the file" << std::endl;
        std::cerr << "[Ex.] Are you sure you are in the correct path?" << std::endl;
        exit(1);
    }

    std::string line;
    unsigned int num_attributes = 0;
    bool reading_data = false;
    std::vector<std::vector<double>> data;
    std::vector<char> labels;

    while (std::getline(ifile, line))
    {
        // Trim leading and trailing whitespace
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

        if (line.empty())
            continue;

        if (line == "@data")
        {
            reading_data = true;
            continue;
        }

        if (!reading_data && line.compare(0, 10, "@attribute") == 0)
            num_attributes++;

        if (reading_data)
        {
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string token;

            while (std::getline(iss, token, ','))
                tokens.push_back(token);

            if (tokens.size() != num_attributes)
            {
                std::cerr << "[WARNING] Skipping inconsistent data line: " << line << std::endl;
                continue; // Skip inconsistent rows
            }

            std::vector<double> data_row;
            for (size_t i = 0; i < tokens.size(); i++)
            {
                if (i == tokens.size() - 1)
                    labels.push_back(tokens[i][0]); // Store the first character as a char in class_vector
                else
                    data_row.push_back(std::stod(tokens[i])); // Store data in a temporary row vector
            }

            data.push_back(data_row); // Append the row to data_matrix
        }
    }

    if (data.empty())
    {
        std::cerr << "[ERROR] No valid data rows found in the file." << std::endl;
        exit(1);
    }

    this->setData(data);
    this->setLabels(labels);
    this->computeMean();
    this->computeVariance();
}

std::vector<Data> Data::createPartitions(int k) const
{
    const std::vector<std::vector<double>> &data_matrix = this->getData();
    const std::vector<char> &class_vector = this->getLabels();

    if (k <= 0 || k > static_cast<int>(data_matrix.size()))
    {
        throw std::invalid_argument("Invalid value of k.");
    }

    // Shuffle the data and class vectors together
    std::vector<std::pair<std::vector<double>, char>> shuffled_data;
    for (unsigned int i = 0; i < data_matrix.size(); ++i)
    {
        shuffled_data.push_back(std::make_pair(data_matrix[i], class_vector[i]));
    }

    std::mt19937 gen(Seed::getInstance().getSeed());
    std::shuffle(shuffled_data.begin(), shuffled_data.end(), gen);

    // Initialize partitions
    std::vector<Data> partitions(k);

    // Fill partitions
    for (unsigned int i = 0; i < shuffled_data.size(); ++i)
    {
        partitions[i % k].addDataPoint(shuffled_data[i].first, shuffled_data[i].second);
    }

    return partitions;
}

void Data::mergeData(const Data &otherData)
{
    this->data.insert(data.end(), otherData.getData().begin(), otherData.getData().end());
    this->labels.insert(labels.end(), otherData.getLabels().begin(), otherData.getLabels().end());
}