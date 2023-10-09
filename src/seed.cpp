#include "seed.h"
#include "tools_helper.h"

Seed *Seed::instance = nullptr;

Seed &Seed::getInstance()
{
    if (!instance)
    {
        // Use integer limits for generating a random integer seed
        int minSeed = 0;
        int maxSeed = std::numeric_limits<int>::max();
        instance = new Seed(ToolsHelper::generateUniformRandomNumberInteger(minSeed, maxSeed));
    }
    return *instance;
}

void Seed::setSeed(int value)
{
    seedValue = value;
}

int Seed::getSeed() const
{
    return seedValue;
}

Seed::Seed() : seedValue(0) {}

Seed::Seed(int seed) : seedValue(seed) {}

Seed::~Seed() {}
