
# Markov Chain

A Markov Chain is a stochastic (probabilistic) mathematical model that describes a sequence of events where the next state depends only on the current state, not on the sequence of previous states.

This is called the Markov Property or Memoryless Property.

## What is a "State"?

A state is simply the current condition or situation.

Examples:

Weather

    Sunny
    Rainy
    Cloudy

Traffic

    Light
    Moderate
    Heavy

Stock Market

    Bull
    Bear
    Stable

Each of these is a state.


## Markov Property (Memoryless Property)

The future depends only on the present, not on the past.

Suppose today's weather is Sunny.

Tomorrow's weather depends only on today's weather.

It does not matter whether it was rainy or cloudy three days ago.

    Yesterday → Today → Tomorrow
    
     Rainy     Sunny     ?

Tomorrow depends only on Sunny.

# Example

Imagine the weather behaves like this:

    Sunny
    │
    ├── 80% → Sunny
    └── 20% → Rainy

    Rainy
    │
    ├── 60% → Rainy
    └── 40% → Sunny

If today is Sunny:

* 80% chance tomorrow is Sunny.
* 20% chance tomorrow is Rainy.

