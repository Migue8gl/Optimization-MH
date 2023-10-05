#include "Data.h"

Data::Data() : data(), labels() {}

Data::Data(std::vector<std::vector<double>> data, std::vector<char> labels)
    : data(data), labels(labels)
{
}

Data::~Data()
{
}

const std::vector<std::vector<double>> &Data::getData() const
{
    return this->data;
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
}

void Data::clearData()
{
    this->data.clear();
    this->labels.clear();
}

std::size_t Data::size() const
{
    return this->data.size();
}
