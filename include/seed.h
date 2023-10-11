#ifndef SEED_H
#define SEED_H

/**
 * @class Seed
 * @brief Singleton class for managing seed values.
 */
class Seed
{
public:
    /**
     * @brief Get the instance of the Seed singleton.
     * @return Reference to the Seed instance.
     */
    static Seed &getInstance();

    /**
     * @brief Set a specific seed value.
     * @param value The seed value to set.
     */
    void setSeed(int value);

    /**
     * @brief Get the current seed value.
     * @return The current seed value, either explicitly set or randomly generated.
     */
    int getSeed() const;

    /**
     * @brief Destructor for Seed.
     */
    ~Seed();

private:
    /**
     * @brief Private constructor for Seed.
     */
    Seed();

    /**
     * @brief Constructor for Seed with a specified seed value.
     * @param seed The seed value to set.
     */
    Seed(int seed);

    /**
     * @brief Generate a random seed value within the allowed range.
     * @return The generated random seed.
     */
    int generateRandomSeed() const;

    int seedValue;         ///< The current seed value.
    bool isSeedSet;        ///< Flag to track whether a seed is explicitly set.
    static Seed *instance; ///< Singleton instance of Seed.
};

#endif
