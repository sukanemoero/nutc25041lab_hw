from langgraph.graph import END, START, StateGraph

from langgraph.checkpoint.memory import MemorySaver
from nodes.cache_checking import cache_checking_node
from nodes.chat import chat_node
from nodes.coordinator import coordinator_node
from nodes.planner import planner_node
from utils.state import State


def graph_build():
    builder = StateGraph(State)
    builder.add_edge(START, "cache_checking")
    builder.add_node("cache_checking", cache_checking_node)
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("planner", planner_node)
    builder.add_node("chat", chat_node)
    builder.add_edge("chat", END)

    memory = MemorySaver()

    return builder.compile(checkpointer=memory)
