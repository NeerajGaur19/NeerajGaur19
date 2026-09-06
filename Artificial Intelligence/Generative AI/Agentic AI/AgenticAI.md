
# Agentic AI

Agentic AI is an AI system that can understand a goal, decide what actions are required, use tools, execute those actions, evaluate the results, and continue working until the goal is achieved.


## Goal

     You give the AI a goal, rather than explicitly programming every step.

---

## 1. Evolution from Traditional Programming

     Traditional Programming
            ↓
     Machine Learning
            ↓
       Deep Learning
            ↓
      Generative AI
            ↓
           LLMs
            ↓
        LangChain
            ↓
           RAG
            ↓
    AI Agents / Agentic AI
            ↓
    Multi-Agent Systems
            ↓
    Agentic Workflows + Production Agentic AI


## Traditional program
    
    Input → Fixed Logic → Output


## LLM application

    User Question → Prompt → LLM → Answer


## RAG application

     User Question
          ↓
      Retriever
          ↓
    Vector Database
          ↓
    Relevant Documents
          ↓
         LLM
          ↓
        Answer

---

## 2. Traditional program vs Agentic AI

### Traditional program 

     Step 1: Search flights
     Step 2: Search hotels
     Step 3: Compare prices
     Step 4: Create itinerary

You define everything.


### Agentic AI

  You say:

  "Plan my 5-day trip to Goa within ₹50,000."

  The agent can determine:

     Goal: Plan Goa trip
            ↓
     Need travel information?
            ↓
     Search flights
            ↓
     Need accommodation?
            ↓
     Search hotels
            ↓
     Need to calculate budget?
            ↓
     Use calculator
            ↓
     Need itinerary?
            ↓
     Create daily plan
            ↓
     Check budget
            ↓
     Return final answer

 This ability to reason about actions and execute them is the basic idea behind Agentic AI.

---

## 3. AI Agent vs Agentic AI

  These terms are related but slightly different.

  ### AI Agent

   An AI agent is an individual entity that can perform actions.

      User
       ↓
     AI Agent
      ├── LLM
      ├── Memory
      ├── Tools
      └── Actions

   Example:

  * Travel agent
  * Coding agent
  * Customer support agent
  * Research agent

  ### Agentic AI

  Agentic AI is the broader approach or system where AI behaves in a goal-directed, autonomous way.
  
  It may contain:
  
     Agentic AI System
            │
            ├── Agent 1: Researcher
            ├── Agent 2: Analyst
            ├── Agent 3: Writer
            └── Agent 4: Reviewer
  
  Simple way to remember
     
     AI Agent = the worker
     
     Agentic AI = the way the workers operate and collaborate

---
  
## 4. Core architecture of an AI Agent

  A basic agent has several important components.

                 ┌───────────────────┐
                 │       USER        │
                 └─────────┬─────────┘
                           │ Goal
                           ▼
                 ┌───────────────────┐
                 │       AGENT       │
                 │                   │
                 │  ┌─────────────┐  │
                 │  │    LLM      │  │
                 │  │   BRAIN     │  │
                 │  └──────┬──────┘  │
                 │         │         │
                 │  ┌──────▼──────┐  │
                 │  │   Memory    │  │
                 │  └─────────────┘  │
                 │                   │
                 │  ┌─────────────┐  │
                 │  │    Tools    │  │
                 │  └──────┬──────┘  │
                 │         │         │
                 │  ┌──────▼──────┐  │
                 │  │   Actions   │  │
                 │  └─────────────┘  │
                 └───────────────────┘

   Let's understand each one.

---

## 5. Component 1: LLM = Brain

The LLM is the reasoning engine.

Examples include:
     
     * GPT models
     * Claude models
     * Gemini models
     * Llama models

The LLM decides:
     
     What does the user want?
             ↓
     What information do I need?
             ↓
     Do I need a tool?
             ↓
     Which tool should I use?
             ↓
     What should I do next?

For example:

     "Find the latest sales data and calculate the growth."

The LLM may decide:

     1. Need sales data
     2. Use database tool
     3. Retrieve current sales
     4. Calculate growth
     5. Check calculation
     6. Generate report

---

## 6. Component 2: Tools

An LLM by itself mainly generates text.

An agent becomes much more powerful when it can use tools.

Examples:

     AI Agent
        │
        ├── Web Search
        ├── Calculator
        ├── Python
        ├── SQL Database
        ├── RAG Retriever
        ├── Email
        ├── Calendar
        ├── API
        └── File System

Example user request:

     "What was our company's revenue last quarter?"


The agent may do:

     User Question
           ↓
     LLM decides:
     "I need company financial data"
           ↓
       SQL Tool
           ↓
       Database
           ↓
     Revenue Data
           ↓
     LLM analyzes data
           ↓
     Final Answer

This is fundamentally different from a normal chatbot.

---

## 7. Component 3: Memory

An agent may need to remember information.

### Short-term memory

Information relevant to the current task.

     User: Plan a trip to Goa
     User: Make it cheaper
     User: Add water sports

The agent remembers the context of the conversation.

### Long-term memory

Information useful across interactions.

For example:

     User preferences:
     - Budget traveler
     - Prefers vegetarian food
     - Likes beaches
     - Prefers direct flights

The agent can use this information in future tasks.

### Working memory

Temporary information while solving a problem.

     Flight cost = ₹12,000
     Hotel = ₹20,000
     Activities = ₹5,000
     Food = ₹8,000
     Total = ₹45,000

---

## 8. Component 4: Planning

This is one of the most important parts of Agentic AI.

A complex goal may need to be divided into smaller tasks.

User:

     "Analyze my sales data and recommend how to improve revenue."

The agent creates a plan:

     Goal
      │
      ├── Step 1: Get sales data
      │
      ├── Step 2: Clean data
      │
      ├── Step 3: Analyze trends
      │
      ├── Step 4: Identify low-performing products
      │
      ├── Step 5: Identify regions
      │
      ├── Step 6: Find revenue opportunities
      │
      └── Step 7: Create recommendations

  This is called:

     Task decomposition

  The agent converts:

     Complex Goal
          ↓
     Smaller Tasks
          ↓
     Actions
          ↓
     Results

---

## 9. Component 5: Action

The agent doesn't just think.

It can act.

  Examples:

     Search web
           ↓
     Read document
           ↓
     Query database
           ↓
     Call API
           ↓
     Execute Python
           ↓
     Send email
           ↓
     Update CRM

  Conceptually:

     thought = llm("What should I do?")
     
     if thought == "search":
         search()
     
     elif thought == "query_database":
         query_database()
     
     elif thought == "calculate":
         calculator()

  Of course, modern agent frameworks do this more systematically.

---

## 10. Component 6: Reflection

An advanced agent can evaluate its own output.

Example:

     Agent creates report
             ↓
     Agent evaluates report
             ↓
     "Is information sufficient?"
             ↓
     No
             ↓
     Search for more data
             ↓
     Improve report

This is sometimes described as:

     Generate
        ↓
     Evaluate
        ↓
     Critique
        ↓
     Improve

Example:

     "I could not find sufficient evidence to support this conclusion. I should retrieve more information."

This creates an iterative loop.

---

## 11. The most important Agentic AI loop

At the heart of many agents is:

### Think → Act → Observe → Repeat
           ┌───────────────┐
           │     THINK     │
           └───────┬───────┘
                   ↓
           ┌───────────────┐
           │      ACT      │
           └───────┬───────┘
                   ↓
           ┌───────────────┐
           │    OBSERVE    │
           └───────┬───────┘
                   ↓
                   │
              Goal done?
               /      \
             No        Yes
             ↓           ↓
          THINK         END

Example:

     "Find the cheapest flight."

Think
          
          I need flight prices.

Act
          
          Call flight search tool.

Observe

          I found three flights.

Think

          I need to compare them.

Act

          Compare prices.

Observe

          Flight A is cheapest.

Final answer

          Recommend Flight A.


# Agent types

A useful way to learn these:

## Simple Reflex Agent

     Perception
        ↓
     IF-THEN Rule
        ↓
     Action

Example:

     Dirty → Clean

## Model-Based Reflex Agent
     
     Perception
        ↓
     Update World Model
        ↓
     Rule
        ↓
     Action

Example:

     Kitchen = dirty
     Living room = clean

## Goal-Based Agent

     Now the agent asks:

     What goal am I trying to achieve?

     Current State
          ↓
     Goal
          ↓
     Plan
          ↓
     Action

Example:

     Goal = Entire house should be clean

It determines:

     Kitchen dirty
     Living room clean
     Bedroom dirty

→ Clean Kitchen
→ Clean Bedroom

## Utility-Based Agent

Now there can be multiple possible solutions.

The agent asks:

     Which solution is the best?

For example:

     Plan A → 30 minutes
     Plan B → 20 minutes
     Plan C → 15 minutes but high energy

The agent evaluates utility:
     
     Cost
     Time
     Energy
     Safety
     Quality

and chooses the best option.

## Learning Agent

The agent learns from experience.

     Experience
         ↓
     Feedback
         ↓
     Learning
         ↓
     Improve Future Decisions
     
