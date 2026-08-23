
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

