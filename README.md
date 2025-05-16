# 🧠 AI Agent for Medical Data

## 🩺 A natural language interface for querying medical practice data using conversational language instead of SQL.

### 🚀 Motivation

This application allows **medical staff** to easily access information from the practice database **without requiring technical SQL knowledge**.

## 🧱 Technical Architecture

The application follows a modular architecture with these key components:

- **Frontend Layer**: Streamlit-based reactive UI with dynamic component rendering  
- **Middleware Layer**: Query processing pipeline with NL2SQL transformation  
- **Database Layer**: SQLite-based persistence with optimized query execution paths  
- **AI Integration Layer**: LLM-powered query understanding and transformation  


## ✨ Advanced Features

- 🔍 **LLM-Powered Natural Language Processing**: Transforms plain English to optimized SQL using Groq's Meta-Llama/Llama-4-Scout-17B-16e-Instruct model  
- ⚛️ **Interactive ReactFlow UI**: Streamlit-based dynamic interface with real-time query processing  
- 🧠 **Automated Schema Introspection**: SQLite PRAGMA-based schema discovery with relationship mapping  
- 🚀 **Semantic Query Optimization**: Automatic transformation of queries with duplicate prevention algorithms  
- 🧩 **LRU Cache Mechanism**: Optimized query caching using MD5 hash-based caching strategy  
- 📊 **Data Quality Analytics**: Automatic anomaly detection in query result sets  
- 🔍 **Regular Expression Pattern Matching**: Enhanced string comparison for case-insensitive queries  
- 🔄 **Thread-safe Database Connection Pool**: Efficient connection management for concurrent queries  


agno >= 0.5.1  
groq >= 0.4.0  
python-dotenv >= 1.0.0  
pandas >= 2.0.0  
sqlalchemy >= 2.0.0  
fastapi >= 0.103.0  
uvicorn >= 0.23.0  

GROQ_API_KEY=your_groq_api_key
AGNO_API_KEY=your_agno_api_key  
DATABASE_URL=sqlite:///medical_practice.db
LOG_LEVEL=INFO

##**Database Schema**

The application works with a medical practice database that contains the following tables:

-Patients
-Appointments
-Billing
-Insurance
-Procedures
-Doctors/Staff
-Financial data
-Supply inventory

## 📚 References

- https://github.com/infiniflow/ragflow  
- https://youtu.be/wdHlKXFPqro?si=O8J8TtlHicoJAZ8S  
- https://github.com/venugopal-adep/agno-agents/blob/main/agno_agent_advanced.ipynb  
- https://console.groq.com/docs/integrations  
- https://blog.futuresmart.ai/mastering-natural-language-to-sql-with-langchain-nl2sql#heading-building-a-basic-nl2sql-model  
- https://github.com/peremartra/Large-Language-Model-Notebooks-Course/blob/main/P1-NL2SQL/nl2sql_prompt_OpenAI.ipynb  
- https://youtu.be/SH3R8ryfR04?si=PN86H9NTOkJTnv4z
- https://console.groq.com/docs/openai
- https://docs.agno.com/introduction
- https://github.com/infiniflow/ragflow
- https://github.com/dharsandip/agno_ai_agent_sql_tools

## 🛠️ System Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/medical-sql-assistant.git
cd medical-sql-assistant

| Package               | Purpose                            |
| --------------------- | ---------------------------------- |
| `streamlit`           | Web application framework          |
| `pandas`              | Data manipulation and display      |
| `sqlite3`             | Database connection                |
| `dotenv`              | Environment variable management    |
| `agno`                | Natural language to SQL conversion |
| `groq`                | LLM API integration                |
| `PIL`                 | Image processing for logo          |
| `logging`             | Application logging                |
| `hashlib`             | Performance optimization (hashing) |
| `functools.lru_cache` | Caching optimized queries          |


