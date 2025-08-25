from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import httpx
import os
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv


load_dotenv()

@tool("search_available_cars")
def search_available_cars_tool(
    agreement: str,
    fromDate: str,
    fromTime: str,
    toDate: str,
    toTime: str,
    pickupBranch: int,
    returnBranch: int,
    isTourist: bool,
    product: int
) -> dict:
    """Search for available cars using Shlomo SIXT's rental API."""
    url = "https://backend-prod.shlomo.co.il/api/v1/rent/all-groups"
    payload = {
        "agreement": agreement,
        "fromDate": fromDate,
        "fromTime": fromTime,
        "toDate": toDate,
        "toTime": toTime,
        "pickupBranch": pickupBranch,
        "returnBranch": returnBranch,
        "isTourist": isTourist,
        "product": product
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

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

tools = [search_available_cars_tool, get_branches_tool]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Test car search
print("=== TESTING CAR SEARCH ===")
result = search_available_cars_tool.invoke({
"fromDate":"28/08/2025","toDate":"30/08/2025","fromTime":"02:01","toTime":"02:01","pickupBranch":49,"returnBranch":49,"isTourist":False,"agreement":"121845","product":9807
})
print(result)

print("\n" + "="*50 + "\n")

# Test branches search
print("=== TESTING BRANCHES SEARCH ===")
branches_result = get_branches_tool.invoke({})
print(branches_result)


