
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

# State Transition Diagram

               0.8
         +-------------+
         |             |
         ▼             |
      Sunny ---------> Rainy
        ▲               │
        │               │0.6
        │0.4            ▼
        +---------------+

* Sunny → Sunny = 0.8
* Sunny → Rainy = 0.2
* Rainy → Sunny = 0.4
* Rainy → Rainy = 0.6

# Transition Probability Matrix

Instead of a diagram, we can use a matrix.
        
        Current State	Sunny	    Rainy
        Sunny	        0.8	        0.2
        Rainy	        0.4	        0.6

Each row sums to 1 because one of the possible next states must occur.

# How Markov Chain Works

Suppose

Today

    Sunny

Tomorrow

    80% Sunny
    
    20% Rainy

Suppose tomorrow becomes Rainy.

The next day's probabilities become

    Rainy
    
    ↓
    
    60% Rainy
    
    40% Sunny

Notice:

Only the current state matters.

# Real-Life Example: Customer Behavior

States

    Browsing
    
    ↓
    
    Cart
    
    ↓
    
    Purchase

Transition probabilities

<img width="851" height="302" alt="image" src="https://github.com/user-attachments/assets/5cba9105-3f32-4fda-96a7-225688bb9c3b" />

Businesses use Markov Chains to predict customer journeys.

Components of a Markov Chain

A Markov Chain has four main components.

1. States

Example

Sunny

Rainy

Cloudy
2. Transition Probabilities

Probability of moving from one state to another.

Example

Sunny → Rainy = 0.2
