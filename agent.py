import os
import re
import json
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict, NotRequired
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import (
    ToolMessage, 
    AIMessage, 
    HumanMessage, 
    SystemMessage, 
    convert_to_messages, 
    AnyMessage, 
    RemoveMessage
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableConfig

load_dotenv()

# Disable LangSmith tracing for production optimization - we use langfuse instead
os.environ["LANGSMITH_TRACING"] = "false"

# 1. Model & Database Initialization
groq_model = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

DB_URI = os.getenv("DATABASE_URL")
if not DB_URI:
    raise ValueError("CRITICAL ERROR: DATABASE_URL is missing from the environment variables!")

db = SQLDatabase.from_uri(
    DB_URI,
    include_tables=["goods"],
    max_string_length=1500
)

# 2. Tools & Tool Binding Configuration
toolkit = SQLDatabaseToolkit(db=db, llm=groq_model)
tools = toolkit.get_tools()

gemini_with_tools = gemini_model.bind_tools(tools)
groq_with_tools = groq_model.bind_tools(tools)

# Router Fallback Strategy: Groq as primary, Gemini as backup
model = groq_with_tools.with_fallbacks([gemini_with_tools])

run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
run_query_tool.description = (
    "Execute a SQL query against the database and return the result. "
    "Pass the exact SQL string into the query argument."
)

# 3. Graph State Definition
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    raw_db_data: NotRequired[Any]  # Raw structured data passed directly to the Android UI

GOODS_SCHEMA = """
Table: goods
Columns:
- id: BIGINT (Primary Key)
- product_name: TEXT (e.g., 'رب گوجه قوطی')
- brand: TEXT (e.g., 'مکنزی', 'مهرام')
- weight: INTEGER (e.g., 800)
- price: INTEGER
- price_after_off: INTEGER (Use this for sorting cheapest items)
- off_percent: INTEGER
- llm_guide: TEXT ('recommended', 'best quality', or null)
- image: TEXT (URL string)
"""

# 4. Prompts Definition
GENERATE_QUERY_SYSTEM_PROMPT = """
You are an AI agent designed to translate natural-language requests into syntactically correct {dialect} SELECT queries for a grocery application based on this exact schema:
{GOODS_SCHEMA}.
Your primary role is to call the appropriate query execution tool with your generated SQL query.

CRITICAL SCHEMA RULES:
- You MUST always use `SELECT *` from the "goods" table.
- Limit results to at most {top_k} rows unless the user explicitly requests a different number. Always append a `LIMIT 5` clause per query segment.
- Use single quotes `''` for string literals (e.g., product_name LIKE '%رب%').
- always put ; in the end of query.

Business Logic & Filtering Rules:
1. **Cheapest items**: If the user asks for the "cheapest" or "lowest price", filter using `product_name LIKE '%X%'` and order by the numeric column `price_after_off` in ASCENDING order (`ORDER BY price_after_off ASC`).
2. **Best Quality**: If the user asks for "best quality" or "highest quality", filter rows using `product_name LIKE '%X%'` and prioritize those with `llm_guide = 'best quality'` using: `ORDER BY (CASE WHEN llm_guide = 'best quality' THEN 0 ELSE 1 END), off_percent DESC LIMIT 5`.
3. **Default / Recommended Products**: If the user asks for a product generally without modifiers, or asks for recommended items, you MUST prioritize recommended items using: `ORDER BY (CASE WHEN llm_guide = 'recommended' THEN 0 ELSE 1 END), off_percent DESC LIMIT 5`.
4. **Farsi Search**: Always write the raw string filter explicitly in the query using wildcards, for example: `WHERE product_name LIKE '%رب گوجه%'`. Do not use parameterized placeholders.
5. **Multi-Product Handling (UNION ALL & Smart Priority)**: If the user asks for two or more different products (e.g., "روغن و رب گوجه میخوام"), you MUST generate a separate SELECT statement for each product and combine them using `UNION ALL`. Every single SELECT statement in the union MUST have its own `LIMIT 5` clause and use the priority sorting from rule 3.

Example Tool Arguments for Multiple Products:
- User: "روغن و رب گوجه و ماکارانی میخوام"
  ➔ Tool Query argument: 
  (SELECT * FROM goods WHERE product_name LIKE '%روغن%' ORDER BY (CASE WHEN llm_guide = 'recommended' THEN 0 ELSE 1 END), off_percent DESC LIMIT 5) 
  UNION ALL 
  (SELECT * FROM goods WHERE product_name LIKE '%رب%' ORDER BY (CASE WHEN llm_guide = 'recommended' THEN 0 ELSE 1 END), off_percent DESC LIMIT 5) 
  UNION ALL 
  (SELECT * FROM goods WHERE product_name LIKE '%ماکارانی%' ORDER BY (CASE WHEN llm_guide = 'recommended' THEN 0 ELSE 1 END), off_percent DESC LIMIT 5);

- User: "ارزان ترین ماکارونی و نوشابه"
  ➔ Tool Query argument: 
  (SELECT * FROM goods WHERE product_name LIKE '%ماکارانی%' ORDER BY price_after_off ASC LIMIT 5) 
  UNION ALL 
  (SELECT * FROM goods WHERE product_name LIKE '%نوشابه%' ORDER BY price_after_off ASC LIMIT 5);

Example Tool Arguments for one product:
- User: "ارزان ترین ماکارانی"
  ➔ Tool Query argument: SELECT * FROM goods WHERE product_name LIKE '%ماکارانی%' ORDER BY price_after_off ASC LIMIT 5;
   
- User: "Show me recommended meat"
  ➔ Tool Query argument: SELECT * FROM goods WHERE product_name LIKE '%گوشت%' ORDER BY (CASE WHEN llm_guide = 'recommended' THEN 0 ELSE 1 END), off_percent DESC LIMIT 5;
""".format(
    dialect=db.dialect,
    GOODS_SCHEMA=GOODS_SCHEMA,
    top_k=5,
)

GENERATE_ANSWER_PROMPT = """
You are a brilliant, friendly, and honest sales assistant for a grocery mobile application. 
Your job is to write a short, engaging, and persuasive response in Persian based strictly on the provided database context.

CRITICAL CONTEXT RULES:
1. **Recommended Products**: If the user asked for recommended items (good price/quality matching) and the database results contain items where `llm_guide` is 'recommended', enthusiastically explain to the user in Persian why these specific products are highly recommended choices.
2. **Missing Recommendations**: If the user asked for a recommended item, but the database results do not have 'recommended' values in the `llm_guide` column, be completely honest! Reassure them that all our store items are high quality, and then smoothly introduce the available options present in the context.
3. **Best Quality**: If the user asked for the absolute best quality and the database results show items with `llm_guide` as 'best quality', highlight their premium value enthusiastically.
4. **Context Constraints**: If the user did not explicitly ask for "recommended" or "best_quality" items, do NOT mention or imply these attributes unless those exact labels ('recommended' or 'best_quality') are present in the provided database context.

GENERAL GENERATION RULES:
- Address the user politely in Persian.
- Keep the answer extremely brief and helpful. Products are already displayed as UI cards on the device, so focus purely on a conversational intro/summary.
- Do not mention or worry about product images; they are automatically processed and displayed by the app interface.
- If the context is empty or you do not know the answer, politely state in Persian that you couldn't find the item right now.
- Never guess or hardcode the numeric placement or index order of items (e.g., do not say 'fourth choice') unless you are explicitly reading it out from an ordered list structure. Simply refer to items by their brand name and characteristics.

Question: {question}
Database Context: {context}

Response (in Persian):
"""

GENERAL_CHAT_PROMPT = """
You are a brilliant, friendly, and honest sales assistant for a grocery mobile application named "افق کوروش". 
The user is asking a general question, greeting you, or asking for a recipe/cooking advice.

CRITICAL RULES FOR GROCERY CONTEXT:
1. **Friendly & Brief**: Speak in a warm, polite, and energetic Persian tone. Keep responses relatively short and sweet since this is a mobile chat.
2. **Recipe / Cooking Requests**: If the user asks for a recipe (e.g., how to make Ghormeh Sabzi, a cake, etc.):
   - Step 1: Briefly list the main ingredients needed.
   - Step 2: Give a super simple, step-by-step cooking guide.
   - Step 3: Enthusiastically remind them that they can buy all these fresh ingredients right now from our "OK Market" app! (e.g., "راستی، می‌تونی همه این مواد اولیه تازه رو همین الان از افق کوروش سفارش بدی!").
3. **Greetings**: If they say hello, welcome them warmly to OK Market and ask what groceries they are looking for today.
4. **No Database assumptions**: Do not mention specific store prices or inventory counts here, as this node does not query the live database. Just focus on being a helpful shopping companion.

Response (in Persian):
"""

# 5. Graph Nodes Definition
async def generate_query(state: AgentState, config: RunnableConfig):
    system_message = SystemMessage(content=GENERATE_QUERY_SYSTEM_PROMPT)
    standard_messages = convert_to_messages(state["messages"])
    messages_to_send = [system_message] + standard_messages
    response = await model.ainvoke(messages_to_send, config=config)
    return {"messages": [response]}

def run_and_parse(sql_query):
    try:
        res = db.run(sql_query, include_columns=True)
        if isinstance(res, str):
            try:
                return json.loads(res)
            except json.JSONDecodeError:
                return eval(res)
        return res
    except Exception as e:
        print(f"Database Error: {e}")
        return []

async def custom_run_query_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]
    query = tool_call["args"]["query"]
    call_id = tool_call["id"]
            
    data = run_and_parse(query)
    clean_data_for_llm = []
    categorized_raw_data = {}

    keywords = re.findall(r"LIKE\s+['\"]%(.*?)%['\"]", query, re.IGNORECASE)

    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                p_name = row.get("product_name", "")
                clean_row = {
                    "نام محصول": p_name,
                    "برند": row.get("brand"),
                    "وزن": row.get("weight"),
                    "درصد تخفیف": row.get("off_percent"),
                    "قیمت نهایی": row.get("price_after_off"),
                    "ویژگی": row.get("llm_guide") if row.get("llm_guide") is not None else "عمومی"
                }
                clean_data_for_llm.append(clean_row)
            else:
                p_name = ""
                clean_data_for_llm.append(row)

            if p_name:
                detected_category = "سایر"
                for kw in keywords:
                    if kw in p_name:
                        detected_category = kw
                        break
                
                if detected_category not in categorized_raw_data:
                    categorized_raw_data[detected_category] = []
                
                if len(categorized_raw_data[detected_category]) < 5:
                    categorized_raw_data[detected_category].append(row)
    else:
        clean_data_for_llm = data
        categorized_raw_data["دیتای عمومی"] = data

    return {
        "messages": [
            ToolMessage(
                content=json.dumps(clean_data_for_llm, ensure_ascii=False), 
                name="sql_db_query", 
                tool_call_id=call_id
            )
        ],
        "raw_db_data": categorized_raw_data  
    }

async def generate_answer(state: AgentState, config: RunnableConfig):
    db_rows_content = None
    messages = state["messages"]
    
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ToolMessage) and msg.name == "sql_db_query":
            db_rows_content = msg.content
            break

    if not db_rows_content:
        return {"messages": [AIMessage(content="هیچ داده‌ای در پایگاه داده یافت نشد.")]}

    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human')]
    question = human_messages[-1].content if human_messages else "درخواست کالا"

    prompt = GENERATE_ANSWER_PROMPT.format(question=question, context=db_rows_content)
    fast_model = groq_model.with_config(config={"max_tokens": 150, "tags": ["final_answer_stream"]})
    response = await fast_model.ainvoke([{"role": "user", "content": prompt}], config=config)
    
    return {"messages": [AIMessage(content=response.content)]}

async def general_answer_node(state: AgentState, config: RunnableConfig):
    messages = state["messages"]
    last_ai_message = messages[-1]
    clean_history = messages[:-1]
    system_message = SystemMessage(content=GENERAL_CHAT_PROMPT)
    messages_to_send = [system_message] + convert_to_messages(clean_history)
    
    fast_model = model.with_config(config={"max_tokens": 300, "tags": ["final_answer_stream"]})
    response = await fast_model.ainvoke(messages_to_send, config=config)
    
    return {
        "messages": [
            RemoveMessage(id=last_ai_message.id),
            AIMessage(content=response.content)
        ]
    }
    
def should_continue(state: AgentState) -> Literal["general_answer", "run_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "general_answer"
    return "run_query"

# 6. StateGraph Compilation
builder = StateGraph(AgentState)
builder.add_node("generate_query", generate_query)
builder.add_node("run_query", custom_run_query_node)
builder.add_node("generate_answer", generate_answer)
builder.add_node("general_answer", general_answer_node)

builder.add_edge(START, "generate_query")
builder.add_conditional_edges("generate_query", should_continue)
builder.add_edge("run_query", "generate_answer")
builder.add_edge("generate_answer", END)
builder.add_edge("general_answer", END)

