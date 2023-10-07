#ifndef SEED_H
#define SEED_H

/**
 * @class Seed
 * @brief Singleton class for managing a seed value.
 */
class Seed
{
public:
    /**
     * @brief Static member function to get the singleton instance of Seed.
     * @return Reference to the singleton Seed instance.
     */
    static Seed &getInstance();

    /**
     * @brief Set the seed value.
     * @param value The new seed value to set.
     */
    void setSeed(int value);

    /**
     * @brief Get the current seed value.
     * @return The current seed value.
     */
    int getSeed() const;

    /**
     * @brief Delete the copy constructor to prevent copying the instance.
     */
    Seed(const Seed &) = delete;

    /**
     * @brief Delete the assignment operator to prevent copying the instance.
     * @return Reference to this Seed instance.
     */
    Seed &operator=(const Seed &) = delete;

private:
    /**
     * @brief Private constructor to prevent direct instantiation of Seed.
     */
    Seed();

    /**
     * @brief Private constructor to instantiate seed given a seed.
     *
     * @param seed
     */
    Seed(int seed);

    /**
     * @brief Private destructor to prevent deletion through pointers.
     */
    ~Seed();

    // Private data members
    int seedValue;

    // Private static pointer to the singleton instance
    static Seed *instance;
};

#endif // SEED_H
