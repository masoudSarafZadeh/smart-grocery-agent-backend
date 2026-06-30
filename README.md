# smart-grocery-agent-backend
# Autonomous Grocery Shopping Agent Backend (LangGraph + FastAPI)

A production-grade, state-driven backend server hosting an intelligent Text-to-SQL and conversational shopping assistant built for the **OK Market** mobile application ecosystem. 

This repository leverages **LangGraph** for resilient multi-model workflow orchestration, **FastAPI + LangServe** for streaming execution contexts, and **PostgreSQL** for multi-turn state checkpointing persistence and enterprise application indexing.

---

## System Architecture & Conversational Flow

The core system logic evaluates user intents, extracts structured data patterns, validates transactional safety, and generates streaming humanized Persian dialogue.


### Core Workflow Graph Components:
1. **Query Generation (`generate_query`)**: Uses a high-performance **Groq (`openai/gpt-oss-20b`)** LLM layer to transform natural language Farsi into raw Postgres dialect queries. If Groq encounters rate-limiting or downtime, an automated fallback layer shifts the computation to **Google Gemini 2.5 Flash** without user interruption.
2. **Conditional Routing (`should_continue`)**: Inspects downstream tool invocations. If the request requires structured product retrieval, the graph transitions to database execution. If the intent is conversational (e.g., greetings, product cooking recipes), it branches to a isolated general dialogue frame.
3. **Resilient Data Execution (`run_query`)**: Executes parameterized SQL search patterns exclusively on the `goods` schema. It structures query metrics like sorting by cheapest configurations (`price_after_off`), handling multi-item requests natively using parallelized `UNION ALL` execution windows, and extracting clean payloads.
4. **Persian Response Synthesis (`generate_answer` / `general_answer`)**: Merges the processed database vectors into a concise conversational Persian output optimized for mobile UI cards, omitting presentation layer details while preserving critical inventory context rules.

---

## Advanced Production Features

### Async State Persistence & Session Checkpointing
Using the `AsyncPostgresSaver` driver, the agent maintains fully persistent historical context loops across disconnected client requests. Session tracks are bound directly via an incoming database connection pool (`AsyncConnectionPool`) enabling instant recovery of long-running user transactions.

### Custom LangServe Middleware Interceptor
To isolate and persist mobile platform states, a specialized lifecycle modifier intercepts incoming raw HTTP streaming request bodies inside `main.py`. It extracts the `thread_id` parameter without exhausting the internal network stream buffer, dynamically mounting historical timelines and appending **Langfuse** telemetry handlers to the live execution context.

### Enterprise Grade Containerization
The integrated multi-stage `Dockerfile` conforms strictly to rootless execution standards. By shifting tracking runtimes away from standard root accounts to non-privileged access profiles (`USER 1000`), the container natively fits strict security compliance baselines required by cloud targets like Hugging Face Spaces or enterprise Kubernetes clusters.

---

## Repository Directory Structure

```text
├── download.png           # Visual runtime execution state graph layout
├── Dockerfile             # Hardened non-root multi-stage production container configuration
├── .dockerignore          # Optimization patterns excluding local data from builds
├── .gitignore             # Standard exclusion engine tracking for Python runtimes
├── LICENSE                # Open-source distribution permissions engine (MIT License)
├── README.md              # Global repository registry documentation (This file)
├── requirements.txt       # Production pinned application dependency bundle
├── agent.py               # Core LangGraph graph topology and LLM prompt definitions
├── main.py                # FastAPI lifecycle engine and LangServe streaming routes
└── example.env            # Encrypted placeholder file for required API credentials
```

## Local Installation & Execution Guide
Follow these sequential parameters to deploy and run the system locally for sandbox validation:

### 1. Provision Your Environment
Clone the repository and verify your local runtime shell matches a Python 3.11 base environment. Then change directory into the root path:

```bash
cd smart-grocery-agent-backend
```
### 2. Install Dependencies
Run a silent module setup installation using python package manager tracking:

```bash
pip install --no-cache-dir -r requirements.txt
```

3. Setup Private Vault Configurations
Initialize your production variable file by safely copying the repository distribution asset:

```bash
cp example.env .env
```
Open your newly created .env file and supply your valid cloud tokens:

```plaintext
DATABASE_URL=postgresql://<user>:<password>@<host>:<port>/postgres
GROQ_API_KEY=gsk_your_production_groq_credential_string
GOOGLE_API_KEY=AIzaSy_your_production_gemini_credential_string
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=[https://cloud.langfuse.com](https://cloud.langfuse.com)
```

4. Execute the Application Instance
Initialize the live streaming backend server deployment locally:

```bash
python main.py
```
Once initialized, the service mounts a default web application layer accessible at http://localhost:7860/shopping-agent/playground.

🐳 Docker Deployment Pipeline
To containerize, optimize, and launch the application using insulated container primitives:

Build the Image
```bash
docker build -t smart-grocery-agent-backend:1.0 .
```
Run the Container
Map the internal informational container runtime port (7860) seamlessly to your local target machine port interface:

```bash
docker run -d \
  -p 7860:7860 \
  --env-file .env \
  --name smart-grocery-agent-container \
  smart-grocery-agent-backend:1.0
```
## Mobile Client Integration Payload Standard
All active downstream responses passing through the graph payload yield an explicitly structured dual-layered tracking context. The state separates natural language processing boundaries (llm_response) from raw tabular records arrays (raw_db_data), sending them as indexed map blocks to ensure the Android mobile client can instantly render responsive native interface elements:

```JSON
{
  "messages": [
    {
      "type": "ai",
      "content": "سلام! خوش آمدید به اوکی مارکت. برای پخت یک ماکارونی خوشمزه، مواد لازم شامل ماکارونی تازه و رب گوجه فرنگی درجه یک را می‌توانید با تخفیف‌های ویژه همین حالا از اپلیکیشن ما سفارش دهید!"
    }
  ],
  "raw_db_data": {
    "ماکارونی": [
      {
        "id": 104,
        "product_name": "ماکارونی ۷۰۰ گرمی زر ماکارون",
        "brand": "زر ماکارون",
        "weight": 700,
        "price": 34000,
        "price_after_off": 29000,
        "off_percent": 15,
        "llm_guide": "recommended",
        "image": "[https://supabase-storage-url.com/zar_macaroni.png](https://supabase-storage-url.com/zar_macaroni.png)"
      }
    ]
  }
}
```
