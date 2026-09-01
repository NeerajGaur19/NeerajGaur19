
### LangGraph represents an application as a graph.

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


