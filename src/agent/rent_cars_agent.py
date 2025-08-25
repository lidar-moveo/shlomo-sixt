
from __future__ import annotations
import os
import httpx
from typing import Annotated, TypedDict, Dict, Any
from urllib.parse import quote
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()

# Configuration
BASE_URL = os.getenv("SHLOMO_BASE_URL", "https://backend-prod.shlomo.co.il")
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

@tool("search_available_cars")
def search_available_cars_tool(
    fromDate: str,
    fromTime: str,
    toDate: str,
    toTime: str,
    pickupBranch: int,
    returnBranch: int
) -> dict:
    """Search for available cars using Shlomo SIXT's rental API."""
    url = "https://backend-prod.shlomo.co.il/api/v1/rent/all-groups"
    payload = {
        "agreement": "121845",        # Fixed value
        "fromDate": fromDate,
        "fromTime": fromTime,
        "toDate": toDate,
        "toTime": toTime,
        "pickupBranch": pickupBranch,
        "returnBranch": returnBranch,
        "isTourist": False,           # Fixed value
        "product": 9807               # Fixed value
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Origin": "https://www.shlomo.co.il",
        "Referer": "https://www.shlomo.co.il/",
        "clientdetails": '{"urlPath":"/israel/search-results","clientIP":"127.0.0.1"}',
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {"error": str(e)}

@tool("get_branches")
def get_branches_tool() -> dict:
    """Get list of available Shlomo SIXT branches."""
    url = "https://backend-prod.shlomo.co.il/api/v1/rent/branches"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Origin": "https://www.shlomo.co.il",
        "Referer": "https://www.shlomo.co.il/",
        "clientdetails": '{"urlPath":"/israel/search-results","clientIP":"127.0.0.1"}',
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }
    
    try:
        response = httpx.post(url, json={}, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
            
    except Exception as e:
        return {"error": str(e)}

@tool("generate_purchase_link")
def generate_purchase_link_tool(
    fromDate: str,
    fromTime: str,
    toDate: str,
    toTime: str,
    pickupBranch: int,
    returnBranch: int,
    carGroup: int,
    pickupBranchName: str,
    returnBranchName: str,
    pickupBranchNameEn: str,
    returnBranchNameEn: str
) -> str:
    """Generate purchase link for selected car with all required parameters."""
    
    # URL encode branch names for Hebrew text
    pickup_name_encoded = quote(pickupBranchName)
    return_name_encoded = quote(returnBranchName)
    pickup_name_en_encoded = quote(pickupBranchNameEn)
    return_name_en_encoded = quote(returnBranchNameEn)
    
    # URL encode dates and times
    from_date_encoded = quote(fromDate)
    to_date_encoded = quote(toDate)
    from_time_encoded = quote(fromTime)
    to_time_encoded = quote(toTime)
    
    # Build the purchase URL
    purchase_url = (
        f"https://www.shlomo.co.il/israel/additions?"
        f"fromDate={from_date_encoded}&"
        f"toDate={to_date_encoded}&"
        f"fromTime={from_time_encoded}&"
        f"toTime={to_time_encoded}&"
        f"pickupBranch={pickupBranch}&"
        f"pickupBranchName={pickup_name_encoded}&"
        f"returnBranch={returnBranch}&"
        f"returnBranchName={return_name_encoded}&"
        f"pickupBranchNameEn={pickup_name_en_encoded}&"
        f"returnBranchNameEn={return_name_en_encoded}&"
        f"isIsraeliCitizen=true&"
        f"minimumAge=18&"
        f"carGroup={carGroup}&"
        f"product=9807&"
        f"agreement=121845&"
        f"countryCode=&"
        f"currency=NIS&"
        f"rentFromAirport=&"
        f"flyCardElAl=&"
        f"allowCoupons=&"
        f"clientCode=&"
        f"pickupCountry=&"
        f"returnCountry="
    )
    
    print(f"Generated purchase link: {purchase_url}")
    return purchase_url

# State definition
class CarRentalState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    rental_info_complete: bool

# Required rental information - including branch selection
rental_info_needed = "pickup date (DD/MM/YYYY), pickup time (HH:MM), return date (DD/MM/YYYY), return time (HH:MM), pickup branch ID, return branch ID"

def has_rental_info(messages):
    """Check if user provided all rental information"""
    user_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            # Handle both string and list content
            if isinstance(msg.content, str):
                user_messages.append(msg.content)
            elif isinstance(msg.content, list):
                # Join list content into string
                content_str = " ".join([str(item) if isinstance(item, str) else str(item.get('text', '')) for item in msg.content])
                user_messages.append(content_str)
    
    user_text = "\n".join(user_messages)
    
    prompt = f"""Return True if ALL the following rental information is present in the conversation, otherwise return False. 
    Required info: {rental_info_needed}
    User messages: {user_text}
    
    Return only True or False, no explanation."""
    
    response = llm.invoke(prompt)
    
    # Handle response content safely
    if isinstance(response.content, str):
        return "true" in response.content.lower()
    elif isinstance(response.content, list):
        content_str = " ".join([str(item) for item in response.content])
        return "true" in content_str.lower()
    else:
        return False



# Main conversation handler
def rental_assistant(state: CarRentalState):
    """Main conversation node - handles user interaction"""
    system_message = SystemMessage(content=f"""
    You are a helpful car rental assistant for Shlomo SIXT in Israel.
    
    IMPORTANT:
    - Always respond in Hebrew (עברית) 
    - Be friendly, polite and professional
    - Help users rent cars by collecting: {rental_info_needed}
    
    WORKFLOW:
    1. If user asks about locations or doesn't know branch IDs, use get_branches tool to show available branches
    2. Collect all required information from user
    3. Once you have complete rental info, use search_available_cars tool with:
      - fromDate: user date (DD/MM/YYYY format)
      - fromTime: user time (HH:MM format)
      - toDate: user date (DD/MM/YYYY format)
      - toTime: user time (HH:MM format)
      - pickupBranch: user selected branch ID (from branches list)
      - returnBranch: user selected branch ID (from branches list)
    
    4. When displaying search results, ALWAYS show:
      - Car name (groupTypeHe)
      - Price (amountIncDiscountIncVat) 
      - Car Group ID (groupCode) - VERY IMPORTANT!
      - Status (statusHe)
      
    5. After showing cars, ask user to select a car by specifying the car group ID
    
    6. When user selects a car (provides car group ID), use generate_purchase_link tool to create the purchase URL
    
    IMPORTANT NOTES:
    - The tool automatically uses fixed values: agreement="121845", isTourist=false, product=9807
    - Always show car group ID (groupCode) clearly in search results
    - Ask user to choose by car group ID for purchase link generation
    - Always show branches first if user hasn't selected them yet
    - Keep conversations natural and helpful
    - Ask for dates in DD/MM/YYYY format and times in HH:MM format
    """)
    
    messages = [system_message] + state["messages"]
    response = llm.bind_tools([search_available_cars_tool, get_branches_tool, generate_purchase_link_tool]).invoke(messages)
    
    # Check if we now have complete rental info
    info_complete = has_rental_info(state["messages"] + [response])
    
    return {
        "messages": response,
        "rental_info_complete": info_complete
    }

# Tool execution node
def tool_executor(state: CarRentalState):
    """Execute tools when needed"""
    last_message = state["messages"][-1]
    
    if not (isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls')):
        return {"messages": []}
    
    tool_results = []
    
    for tool_call in last_message.tool_calls:
        try:
            # Handle both object and dict formats for tool_call
            tool_name = getattr(tool_call, 'name', tool_call.get('name', ''))
            tool_args = getattr(tool_call, 'args', tool_call.get('args', {}))
            tool_id = getattr(tool_call, 'id', tool_call.get('id', 'unknown'))
            
            if tool_name == "search_available_cars":
                result = search_available_cars_tool.invoke(tool_args)
            elif tool_name == "get_branches":
                result = get_branches_tool.invoke(tool_args)
            elif tool_name == "generate_purchase_link":
                result = generate_purchase_link_tool.invoke(tool_args)
            else:
                result = f"Unknown tool: {tool_name}"
            
            tool_results.append(ToolMessage(
                content=result,
                tool_call_id=tool_id
            ))
        except Exception as e:
            # Handle tool_id safely for error case too
            tool_id = getattr(tool_call, 'id', tool_call.get('id', 'unknown'))
            tool_results.append(ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_id
            ))
    
    return {"messages": tool_results}



# Build the graph
graph_builder = StateGraph(CarRentalState)

# Add nodes
graph_builder.add_node("rental_assistant", rental_assistant)
graph_builder.add_node("tool_executor", tool_executor)

# Add edges
graph_builder.add_edge(START, "rental_assistant")

# Add conditional edges  
def main_router(state: CarRentalState) -> str:
    """Main routing logic"""
    if not state["messages"]:
        return "END"
        
    last_message = state["messages"][-1]
    
    # If last message is AIMessage with tool_calls, MUST go to tool_executor
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_executor"
    
    # If last message is ToolMessage, go back to conversation
    if isinstance(last_message, ToolMessage):
        return "rental_assistant"
    
    return "END"

graph_builder.add_conditional_edges(
    "rental_assistant",
    main_router,
    {
        "tool_executor": "tool_executor",
        "END": END
    }
)

graph_builder.add_conditional_edges(
    "tool_executor",
    main_router,
    {
        "rental_assistant": "rental_assistant",
        "tool_executor": "tool_executor",
        "END": END
    }
)

# Compile the graph
graph = graph_builder.compile()
