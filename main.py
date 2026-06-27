import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from langserve import add_routes
from agent import builder
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

# Initialize Langfuse tracking
langfuse = Langfuse()

# Define the pool
pool = AsyncConnectionPool(
    conninfo=os.getenv("DATABASE_URL"),
    max_size=20,
    kwargs={"autocommit": True},
    open=False # Don't open connections until lifespan runs
)

async def langfuse_config_modifier(config: dict, request: Request) -> dict:
    """
    Intercepts incoming HTTP requests to extract the thread_id without 
    consuming the request body stream permanently, ensuring state persistence.
    """
    try:
        body_bytes = await request.body()
        if body_bytes:
            # Reset the receive channel to allow LangServe to read the body stream again
            async def re_receive():
                return {"type": "http.request", "body": body_bytes}
            request._receive = re_receive
            
            body = json.loads(body_bytes)
            
            # Extract thread_id supporting both LangChain standard and custom flat layouts
            raw_thread_id = (
                body.get("config", {}).get("configurable", {}).get("thread_id") 
                or body.get("configurable", {}).get("thread_id")
            )
            
            if raw_thread_id:
                if "configurable" not in config:
                    config["configurable"] = {}
                config["configurable"]["thread_id"] = str(raw_thread_id)
                print(f"🔗 Injected thread_id into Config: {raw_thread_id}")
    except Exception as e:
        print(f"⚠️ Failed to parse raw request body: {e}")

    # Fallback mechanism to safeguard against Graph initialization crashes
    if "configurable" not in config:
        config["configurable"] = {}
    if "thread_id" not in config["configurable"] or not config["configurable"]["thread_id"]:
        config["configurable"]["thread_id"] = "default_playground_session"
        print("🔗 Fallback triggered: Injected 'default_playground_session'")

    # Inject Langfuse Callback Handler for LangChain/LangGraph Tracing
    langfuse_handler = CallbackHandler()
    if "callbacks" not in config:
        config["callbacks"] = []
    config["callbacks"].append(langfuse_handler)
    
    return config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown lifecycles of the FastAPI application.
    Handles database connection pooling and LangGraph checkpointer initialization.
    """
    # Startup Logic
    print("Opening Async Postgres Connection Pool...")
    await pool.open()
    
    print("Setting up Async Postgres Checkpointer...")
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()

    print("Compiling Agent with Live Checkpointer...")
    agent = builder.compile(checkpointer=checkpointer)

    print("Registering LangServe Routes dynamically...")
    add_routes(
        app, 
        agent, 
        path="/shopping-agent", 
        config_keys=["configurable"], 
        per_req_config_modifier=langfuse_config_modifier,
    )
    
    print("Database, Async Checkpointer, and Graph Tracing are fully ready!")
    yield
    
    # Shutdown Logic
    print("Closing Async Postgres Connection Pool...")
    await pool.close()
    langfuse.flush()
    print("Database Pool disconnected cleanly.")

# Initialize FastAPI App with Lifespan Context Manager
app = FastAPI(
    title="Grocery Shopping Agent Backend",
    version="1.0",
    description="LangGraph backend server for Android client",
    lifespan=lifespan
)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
