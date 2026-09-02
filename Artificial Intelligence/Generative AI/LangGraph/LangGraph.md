
#### LangGraph represents an application as a graph.

A graph consists mainly of:

State
  +
Nodes
  +
Edges

              ┌───────────────┐
              │     START     │
              └───────┬───────┘
                      ↓
              ┌───────────────┐
              │   LLM Node    │
              └───────┬───────┘
                      ↓
               ┌─────────────┐
               │ Need Tool?  │
               └──────┬──────┘
                  YES  │  NO
                 ↓     │    ↓
          ┌──────────┐ │ ┌────────┐
          │Tool Node │ │ │  END   │
          └────┬─────┘ │ └────────┘
               │       │
               └───────┘
                   ↓
               LLM Node


The official Graph API is built around defining state, adding nodes, and connecting those nodes with edges.

## What is a Node?

A node is a step in your AI workflow.

For example:

def call_llm(state):
    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }

That function can become a node.

Another node:

def search_database(state):
    result = database.search(state["query"])

    return {
        "documents": result
    }

Another:

def generate_answer(state):
    answer = llm.invoke(...)

    return {
        "answer": answer
    }

So:
    
    Node 1             Node 2              Node 3
    
    Retrieve      →    Analyze       →     Generate
    documents          documents            answer
    
## What is an Edge?

An edge determines where execution goes next.

For example:

    START
      ↓
    Retrieve
      ↓
    Generate
      ↓
    END

In code:

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

So you can think:

    Node = WHAT to do
    
    Edge = WHERE to go next


## What is State?

This is probably the most important LangGraph concept.

State is the information that travels through your graph.

For example:

    class State(TypedDict):
        question: str
        documents: list
        answer: str

Imagine the user asks:

    "What is Power BI?"

Initial state:

    {
        "question": "What is Power BI?",
        "documents": [],
        "answer": ""
    }

After retrieval:

    {
        "question": "What is Power BI?",
        "documents": [
            "Power BI is Microsoft's BI platform..."
        ],
        "answer": ""
    }

After generation:

    {
        "question": "What is Power BI?",
        "documents": [
            "Power BI is Microsoft's BI platform..."
        ],
        "answer": "Power BI is a business intelligence..."
    }

So the state acts like the shared memory/context of the running graph.

The official quickstart similarly uses graph state to maintain information such as messages and LLM-call counts throughout execution.


