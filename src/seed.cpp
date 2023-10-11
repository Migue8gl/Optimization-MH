#include "seed.h"
#include "tools_helper.h"
#include <random> // Include the <random> header for random number generation

Seed *Seed::instance = nullptr;

Seed &Seed::getInstance()
{
    if (!instance)
    {
        instance = new Seed();
    }
    return *instance;
}

void Seed::setSeed(int value)
{
    seedValue = value;
    isSeedSet = true;
}

int Seed::getSeed() const
{
    if (isSeedSet)
    {
        return seedValue;
    }
    else
    {
        return generateRandomSeed();
    }
}

int Seed::generateRandomSeed() const
{
    // Generate a random seed within the allowed range
    return ToolsHelper::generateUniformRandomNumberInteger(0, std::numeric_limits<int>::max());
}

Seed::Seed() : seedValue(generateRandomSeed()), isSeedSet(false) {}

Seed::Seed(int seed) : seedValue(seed), isSeedSet(true) {}

Seed::~Seed() {}
