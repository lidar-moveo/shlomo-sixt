from __future__ import annotations
import os
from typing import Annotated, TypedDict, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Import the existing agents
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rent_cars_agent import graph as rental_graph, CarRentalState
from sales_cars_agent import graph as sales_graph, CarSalesState

load_dotenv()

# Configuration
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Master state that can handle both rental and sales
class MasterAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Literal["rental", "sales", "unknown"]
    
def detect_user_intent(messages) -> str:
    """Detect if user wants rental or sales service"""
    user_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            if isinstance(msg.content, str):
                user_messages.append(msg.content)
            elif isinstance(msg.content, list):
                content_str = " ".join([str(item) if isinstance(item, str) else str(item.get('text', '')) for item in msg.content])
                user_messages.append(content_str)
    
    user_text = "\n".join(user_messages).lower()
    
    # Keywords for rental intent
    rental_keywords = [
        "השכרה", "להשכיר", "לשכור", "rental", "rent", "השכר", 
        "לתקופה", "לכמה ימים", "לשבוע", "לחודש", "זמני", "קצר טווח"
    ]
    
    # Keywords for sales intent  
    sales_keywords = [
        "לקנות", "קניה", "מכירה", "רכישה", "purchase", "buy", "sale",
        "יד ראשונה", "זירו קמ", "ליסינג", "מימון", "תשלומים", 
        "לרכוש", "בעלות", "קבע", "ארוך טווח"
    ]
    
    rental_score = sum(1 for keyword in rental_keywords if keyword in user_text)
    sales_score = sum(1 for keyword in sales_keywords if keyword in user_text)
    
    if rental_score > sales_score:
        return "rental"
    elif sales_score > rental_score:
        return "sales"
    else:
        return "unknown"

def master_router(state: MasterAgentState):
    """Main routing node that determines user intent and directs to appropriate service"""
    
    current_intent = detect_user_intent(state["messages"])
    
    if current_intent == "unknown":
        # Ask user to clarify their intent
        system_message = SystemMessage(content="""
        You are Shlomo SIXT's main assistant in Israel. Help users choose the right service.
        
        Always respond in Hebrew (עברית).
        
        Ask the user to clarify what they need:
        1. 🚗 השכרת רכב - לתקופות קצרות (ימים/שבועות)
        2. 💰 קניית/ליסינג רכב - לבעלות ארוכת טווח
        
        Be friendly and explain the difference between the services.
        """)
        
        messages = [system_message] + state["messages"]
        response = llm.invoke(messages)
        
        return {
            "messages": response,
            "intent": "unknown"
        }
    
    return {
        "messages": [],
        "intent": current_intent
    }

def rental_service_adapter(state: MasterAgentState):
    """Adapter to run the rental service graph"""
    # Convert master state to rental state
    rental_state = CarRentalState(
        messages=state["messages"],
        rental_info_complete=False
    )
    
    # Run the rental graph
    result = rental_graph.invoke(rental_state)
    
    return {
        "messages": result["messages"],
        "intent": "rental"
    }

def sales_service_adapter(state: MasterAgentState):
    """Adapter to run the sales service graph"""
    # Convert master state to sales state  
    sales_state = CarSalesState(
        messages=state["messages"],
        user_preferences_complete=False
    )
    
    # Run the sales graph
    result = sales_graph.invoke(sales_state)
    
    return {
        "messages": result["messages"], 
        "intent": "sales"
    }

# Build the master graph
master_graph_builder = StateGraph(MasterAgentState)

# Add nodes
master_graph_builder.add_node("master_router", master_router)
master_graph_builder.add_node("rental_service", rental_service_adapter)
master_graph_builder.add_node("sales_service", sales_service_adapter)

# Add edges
master_graph_builder.add_edge(START, "master_router")

# Routing logic
def route_to_service(state: MasterAgentState) -> str:
    """Route to appropriate service based on detected intent"""
    intent = state.get("intent", "unknown")
    
    if intent == "rental":
        return "rental_service"
    elif intent == "sales":
        return "sales_service"
    else:
        return "END"  # Stay in router for clarification

master_graph_builder.add_conditional_edges(
    "master_router",
    route_to_service,
    {
        "rental_service": "rental_service",
        "sales_service": "sales_service", 
        "END": END
    }
)

# Both services end the conversation
master_graph_builder.add_edge("rental_service", END)
master_graph_builder.add_edge("sales_service", END)

# Compile the master graph
graph = master_graph_builder.compile() 