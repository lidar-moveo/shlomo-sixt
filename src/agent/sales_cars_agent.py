from __future__ import annotations
import os
import httpx
from typing import Annotated, TypedDict, Dict, Any, List
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# Configuration
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# Car categories mapping
CAR_CATEGORIES = {
    "מיני-משפחתיות": ["מיני", "קטן", "עירוני"],
    "היברידי": ["היברידי", "הייברידי", "חשמלי חלקי"],
    "קטנות": ["קטן", "קומפקטי", "חסכוני"],
    "משפחתיות": ["משפחתי", "סדאן", "האצ'בק"],
    "ג'יפונים/SUV": ["SUV", "ג'יפ", "גיפון", "קרוסאובר"],
    "מנהלים / יוקרה": ["יוקרה", "מנהלים", "פרימיום"],
    "7 מקומות ומיני וואן": ["7 מקומות", "מיני וואן", "ואן", "רב מקומות"],
    "מסחריות": ["מסחרי", "נותן שירות", "עבודה"],
    "חשמלי": ["חשמלי", "EV", "אלקטרי"]
}

# Manufacturers list
MANUFACTURERS = [
    "ב.מ.וו", "AIWAYS", "BMW", "BYD", "CHERY", "Geely", "Jaecoo", "KGM", "LEAP", 
    "LYNK&CO", "MG", "ORA", "ZEEKR", "אאודי", "אופל", "אינפיניטי", "איסוזו", 
    "אלפא רומיאו", "ב.מ.וו.", "ג'נסיס", "גנסיס", "דאצ'ה", "דאצה", "דונגפנג", 
    "די אס", "די.אס", "הונדה", "וולוו", "טויוטה", "יונדאי", "לנד רובר", "לקסוס", 
    "מאזדה", "מיצובישי", "מרצדס", "ניסאן", "סאנגיונג", "סובארו", "סוזוקי", 
    "סיאט", "סיטרואן", "סקודה", "פולסטאר", "פולקסווגן", "פורד", "פורשה", 
    "פיאט", "פיג'ו", "קאדילק", "קופרה", "קיה", "קרייזלר", "רנו", "שברולט"
]

@tool("get_first_hand_models")
def get_first_hand_models_tool() -> dict:
    """Get all available first-hand car models from Shlomo SIXT sales."""
    url = "https://sales-backend-prod.shlomo.co.il/api/shlomo/models"
    
    try:
        response = httpx.get(url, timeout=30)
        if response.status_code == 200:
            return {"service_type": "first_hand", "data": response.json()}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}", "service_type": "first_hand"}
    except Exception as e:
        return {"error": str(e), "service_type": "first_hand"}

@tool("get_zero_km_cars")
def get_zero_km_cars_tool() -> dict:
    """Get all available zero-km cars from Shlomo SIXT sales."""
    url = "https://sales-backend-prod.shlomo.co.il/api/shlomo/zero-km-cars"
    
    try:
        response = httpx.get(url, timeout=30)
        if response.status_code == 200:
            return {"service_type": "zero_km", "data": response.json()}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}", "service_type": "zero_km"}
    except Exception as e:
        return {"error": str(e), "service_type": "zero_km"}

@tool("get_first_hand_car_details")
def get_first_hand_car_details_tool(importer_model: str) -> dict:
    """Get detailed information about a specific first-hand car model."""
    url = f"https://sales-backend-prod.shlomo.co.il/api/shlomo/first-hand-cars/{importer_model}"
    
    try:
        response = httpx.get(url, timeout=30)
        if response.status_code == 200:
            return {"service_type": "first_hand_details", "importer_model": importer_model, "data": response.json()}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}", "service_type": "first_hand_details"}
    except Exception as e:
        return {"error": str(e), "service_type": "first_hand_details"}

@tool("get_zero_km_car_details")
def get_zero_km_car_details_tool(car_id: str) -> dict:
    """Get detailed information about a specific zero-km car."""
    url = f"https://sales-backend-prod.shlomo.co.il/api/shlomo/zero-km-cars/{car_id}"
    
    try:
        response = httpx.get(url, timeout=30)
        if response.status_code == 200:
            return {"service_type": "zero_km_details", "car_id": car_id, "data": response.json()}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}", "service_type": "zero_km_details"}
    except Exception as e:
        return {"error": str(e), "service_type": "zero_km_details"}

@tool("get_leasing_cars")
def get_leasing_cars_tool() -> dict:
    """Get all available leasing car models from Shlomo SIXT."""
    url = "https://shlomo-leasing-backend-prod.shlomo.co.il/api/shlomo/leasing-cars"
    
    try:
        response = httpx.get(url, timeout=30)
        if response.status_code == 200:
            return {"service_type": "leasing", "data": response.json()}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}", "service_type": "leasing"}
    except Exception as e:
        return {"error": str(e), "service_type": "leasing"}

@tool("get_leasing_car_details")
def get_leasing_car_details_tool(car_id: str) -> dict:
    """Get detailed information about a specific leasing car model."""
    url = f"https://shlomo-leasing-backend-prod.shlomo.co.il/api/shlomo/leasing-cars/{car_id}"
    
    try:
        response = httpx.get(url, timeout=30)
        if response.status_code == 200:
            return {"service_type": "leasing_details", "car_id": car_id, "data": response.json()}
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}", "service_type": "leasing_details"}
    except Exception as e:
        return {"error": str(e), "service_type": "leasing_details"}

@tool("compare_and_recommend")
def compare_and_recommend_tool(
    user_budget: int,
    preferred_category: str = "",
    preferred_manufacturer: str = "",
    payment_preference: str = "any"  # "cash", "monthly", "any"
) -> dict:
    """
    Compare deals across all Shlomo SIXT services and provide smart recommendations.
    This tool should be used after gathering user preferences.
    """
    
    recommendations = {
        "user_criteria": {
            "budget": user_budget,
            "category": preferred_category,
            "manufacturer": preferred_manufacturer,
            "payment_preference": payment_preference
        },
        "analysis": "Based on your criteria, here are the best deals I found across our services:",
        "recommendations": [],
        "savings_potential": {}
    }
    

    
    # This is a placeholder for the comparison logic
    # In a real implementation, this would:
    # 1. Filter cars by criteria
    # 2. Calculate monthly costs for each service
    # 3. Compare total cost of ownership
    # 4. Recommend the best value option
    
    return recommendations

# State definition
class CarSalesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_preferences_complete: bool

# Required user information for car sales
sales_info_needed = "budget range and car type preference (family/SUV/economical/luxury)"

def has_user_preferences(messages):
    """Check if user provided their preferences for car purchase"""
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
    
    # Simple keyword detection - more reliable than LLM for this
    text_lower = user_text.lower()
    
    # Check for budget indicators
    has_budget = any(word in text_lower for word in [
        "תקציב", "מחיר", "₪", "שקל", "אלף", "budget", "עד", "בסביבות", "כ-"
    ])
    
    # Check for car type indicators  
    has_car_type = any(word in text_lower for word in [
        "משפחתי", "suv", "יוקרה", "קטן", "היברידי", "חסכוני", "כלכלי", 
        "חדש", "יד ראשונה", "ליסינג", "רכב", "אוטו", "משפחה"
    ])
    
    return has_budget and has_car_type

# Main conversation handler
def sales_assistant(state: CarSalesState):
    """Main conversation node for car sales assistance"""
    
    system_message = SystemMessage(content=f"""
    You are an expert car sales consultant for Shlomo SIXT in Israel who provides intelligent, persuasive recommendations.
    
    IMPORTANT GUIDELINES:
    - Always respond in Hebrew (עברית)
    - Be friendly, professional, and consultative
    - Help users find the perfect car by collecting: {sales_info_needed}
    
    WORKFLOW:
    1. If user hasn't specified preferences, ask about:
       - Budget range (תקציב)
       - Car type preference (משפחתי, SUV, חסכוני, יוקרה)
    2. Once you have budget AND car type, IMMEDIATELY use search tools:
       - get_first_hand_models (for used cars with history)
       - get_zero_km_cars (for new cars at special prices)
       - get_leasing_cars (for monthly payment options)
    3. Present ALL available cars from different services
    4. Make intelligent comparisons and recommendations
    
    IMPORTANT: Always search across ALL three services to give comprehensive options!
    
    AVAILABLE SERVICES:
    1. 🚗 רכבים יד ראשונה - רכבים עם היסטוריה מוכחת ומחירים אטרקטיביים
    2. ✨ רכבים זירו ק"מ - רכבים חדשים לגמרי במחירים מיוחדים  
    3. 📊 ליסינג פעולי - גמישות מקסימלית עם תשלומים נמוכים
    
    PRESENTATION GUIDELINES:
    
    START with a warm, conversational opening that acknowledges user preferences:
    "בהתבסס על מה שסיפרת לי על [תקציב/סוג רכב], חיפשתי עבורך והנה מה שמצאתי..."
    
    FOR EACH CAR:
    ### [Car Name]
    
    **מחיר:** [Price] ש"ח
    **מפרט:** [Key technical details]
    
    **למה זה מתאים לצרכים שלך:**
    [Connect car features directly to user's stated needs]
    
    **יתרונות:**
    • [Advantage 1 and why it matters to user]
    • [Advantage 2 and why it matters to user]
    • [Advantage 3 and why it matters to user]
    
    **דברים לשקול:** [Honest considerations if any]
    
    **בקיצור:** [One-line summary connecting car to user preferences]
    
    ---
    
    END with comparison and recommendation:
    
    **השוואה מהירה:**
    [Compare key differences between options]
    
    **המלצתי עבורך:** [Clear recommendation with reasoning based on user needs]
    
    TONE AND STYLE:
    - Warm and conversational, not overly salesy
    - Avoid marketing buzzwords ("מהפכני", "מדהים", "פלא") 
    - Be factual but warm ("בדיוק מה שחיפשת", "זה יתאים לך כי...")
    - Always connect features to user's specific needs
    - Show you listened to their requirements
    
    MANDATORY ELEMENTS:
    - Start with personalized opening acknowledging their preferences
    - Present cars in clean, organized format
    - End with clear comparison table and specific recommendation
    - Include cost-benefit analysis (not just price)
    - Maximum 3-4 cars to avoid overwhelming
    - Connect each feature to user's stated needs
    """)
    
    messages = [system_message] + state["messages"]
    response = llm.bind_tools([
        get_first_hand_models_tool,
        get_zero_km_cars_tool, 
        get_first_hand_car_details_tool,
        get_zero_km_car_details_tool,
        get_leasing_cars_tool,
        get_leasing_car_details_tool,
        compare_and_recommend_tool
    ]).invoke(messages)
    
    # Check if we now have complete user preferences
    preferences_complete = has_user_preferences(state["messages"] + [response])
    
    return {
        "messages": response,
        "user_preferences_complete": preferences_complete
    }

# Tool execution node
def tool_executor(state: CarSalesState):
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
            
            if tool_name == "get_first_hand_models":
                result = get_first_hand_models_tool.invoke(tool_args)
            elif tool_name == "get_zero_km_cars":
                result = get_zero_km_cars_tool.invoke(tool_args)
            elif tool_name == "get_first_hand_car_details":
                result = get_first_hand_car_details_tool.invoke(tool_args)
            elif tool_name == "get_zero_km_car_details":
                result = get_zero_km_car_details_tool.invoke(tool_args)
            elif tool_name == "get_leasing_cars":
                result = get_leasing_cars_tool.invoke(tool_args)
            elif tool_name == "get_leasing_car_details":
                result = get_leasing_car_details_tool.invoke(tool_args)
            elif tool_name == "compare_and_recommend":
                result = compare_and_recommend_tool.invoke(tool_args)
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
graph_builder = StateGraph(CarSalesState)

# Add nodes
graph_builder.add_node("sales_assistant", sales_assistant)
graph_builder.add_node("tool_executor", tool_executor)

# Add edges
graph_builder.add_edge(START, "sales_assistant")

# Routing logic
def main_router(state: CarSalesState) -> str:
    """Main routing logic"""
    if not state["messages"]:
        return "END"
        
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool_executor"
    
    if isinstance(last_message, ToolMessage):
        return "sales_assistant"
    
    return "END"

graph_builder.add_conditional_edges(
    "sales_assistant",
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
        "sales_assistant": "sales_assistant",
        "tool_executor": "tool_executor",
        "END": END
    }
)

# Compile the graph
graph = graph_builder.compile()
