import os
import sqlite3
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.sql import SQLTools
import re
import logging
import time
from functools import lru_cache
import hashlib
from PIL import Image


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

agno_api_key = os.getenv("AGNO_API_KEY", "api key")
os.environ["AGNO_API_KEY"] = agno_api_key

db_url = os.getenv("DATABASE_URL", "sqlite:///medical_practice.db")
db_file = db_url.replace("sqlite:///", "")

st.set_page_config(
    page_title="Medical Practice SQL Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def validate_database():
    """Check if the database file exists and can be connected to with detailed diagnostics"""
    if not os.path.exists(db_file):
        st.error(f"Database file not found: {db_file}")
        st.info("Please run the initialization script first to create the database.")
        logger.error(f"Database file not found: {db_file}")
        return False
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            st.warning("Database exists but contains no tables. Please run the initialization script.")
            logger.warning(f"Database {db_file} exists but contains no tables")
            conn.close()
            return False
            
        table_counts = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            table_counts[table_name] = count
            
        conn.close()
        
        empty_tables = [name for name, count in table_counts.items() if count == 0]
        if empty_tables:
            st.warning(f"The following tables exist but contain no data: {', '.join(empty_tables)}")
            logger.warning(f"Empty tables detected: {', '.join(empty_tables)}")
        
        logger.info(f"Database validation successful. Found {len(tables)} tables.")
        return True
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        logger.error(f"Database connection error: {str(e)}")
        return False

sql_tool = None
if validate_database():
    try:
        sql_tool = SQLTools(db_url=db_url)
        logger.info("SQLTools initialized successfully")
    except Exception as e:
        st.error(f"Failed to initialize SQLTools: {str(e)}")
        logger.error(f"SQLTools initialization error: {str(e)}")

@lru_cache(maxsize=32)
def get_table_schema(cache_key=None):
    """Get the schema of all tables in the database with caching"""
    if not sql_tool:
        logger.warning("Cannot get schema: SQLTools not initialized")
        return {}
        
    try:
        start_time = time.time()
        tables = sql_tool.run_sql("SELECT name FROM sqlite_master WHERE type='table';")
        
        schema_info = {}
        for table in tables:
            table_name = table['name']
            columns = sql_tool.run_sql(f"PRAGMA table_info({table_name})")
            
            foreign_keys = sql_tool.run_sql(f"PRAGMA foreign_key_list({table_name})")
            
            schema_info[table_name] = {
                "columns": [
                    {
                        "name": col["name"], 
                        "type": col["type"], 
                        "primary_key": bool(col["pk"]),
                        "nullable": not bool(col["notnull"]),
                        "default_value": col["dflt_value"]
                    }
                    for col in columns
                ],
                "foreign_keys": [
                    {
                        "from": fk["from"], 
                        "to_table": fk["table"], 
                        "to_column": fk["to"]
                    } 
                    for fk in foreign_keys
                ],
                "row_count": get_table_row_count(table_name)
            }
        
        # Get some sample data for each table
        for table_name in schema_info:
            schema_info[table_name]["sample_data"] = get_table_sample_data(table_name)
            
        logger.info(f"Schema fetched in {time.time() - start_time:.2f} seconds")
        return schema_info
    except Exception as e:
        logger.error(f"Error fetching schema: {str(e)}")
        return {}

def get_table_row_count(table_name):
    """Get the number of rows in a table"""
    try:
        # Use direct SQLite connection instead of sql_tool for count queries
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.error(f"Error counting rows in {table_name}: {str(e)}")
        return 0

def get_table_sample_data(table_name, limit=3):
    """Get sample data from a table"""
    try:
        return sql_tool.run_sql(f"SELECT * FROM {table_name} LIMIT {limit}")
    except Exception as e:
        logger.error(f"Error fetching sample data from {table_name}: {str(e)}")
        return []

def detect_and_handle_duplicates(query_text):
    """Add DISTINCT to queries when appropriate to prevent duplicates"""
    # Check if the query is likely to return duplicates (e.g., has JOINs without GROUP BY)
    has_joins = re.search(r'\bjoin\b', query_text, re.IGNORECASE)
    has_group_by = re.search(r'\bgroup\s+by\b', query_text, re.IGNORECASE)
    has_distinct = re.search(r'\bdistinct\b', query_text, re.IGNORECASE)
    
    # If query has joins but no GROUP BY or DISTINCT, suggest adding DISTINCT
    if has_joins and not has_group_by and not has_distinct:
        # Find the SELECT part and add DISTINCT
        modified_query = re.sub(
            r'(SELECT\s+)', 
            r'\1DISTINCT ', 
            query_text, 
            flags=re.IGNORECASE, 
            count=1
        )
        return modified_query, "Added DISTINCT to prevent duplicate rows from joins"
        
    return query_text, None

def improve_string_matching(query_text):
    """Improve string matching to be case-insensitive and handle partial matches"""
    # Convert exact string comparisons to case-insensitive comparisons
    modified_query = re.sub(
        r'(\w+)\s*=\s*([\'"])(.*?)(\2)',
        r'LOWER(\1) = LOWER(\2\3\4)',
        query_text
    )
    
    # If query contains string comparisons that might benefit from partial matching
    if "LIKE" not in modified_query.upper() and re.search(r'\w+\s*=\s*[\'"]', modified_query):
        modified_query = re.sub(
            r'LOWER\((\w+)\)\s*=\s*LOWER\([\'"])(.*?)([\'"])\)',
            r'LOWER(\1) LIKE LOWER(\2%\3)', 
            modified_query
        )
        return modified_query, "Modified for case-insensitive and partial string matching"
        
    return modified_query, None

def preprocess_user_query(query):
    """Preprocess and sanitize user input query"""
    query = query.strip()
    
    query = re.sub(r'\s+', ' ', query)
    
    query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
    query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
    
    return query

def generate_query_hash(query):
    """Generate a hash of the query for caching purposes"""
    return hashlib.md5(query.encode()).hexdigest()

# LRU cache for query results to improve performance
@lru_cache(maxsize=100)
def cached_sql_results(query_hash, sql_query):
    """Cache SQL query results to improve performance for repeated queries"""
    try:
        return sql_tool.run_sql(sql_query)
    except Exception as e:
        logger.error(f"SQL execution error: {str(e)}")
        return None

def sql_agent(query):
    """Process natural language query using Agno SQL Agent with improved error handling"""
    if not sql_tool:
        return "Database connection failed. Please check that the database exists and is properly initialized."
    
    # Preprocess user query
    query = preprocess_user_query(query)
    
    # Refresh schema information to ensure it's current
    schema_info = get_table_schema(time.time() // (60 * 5))  # Cache for 5 minutes
    if not schema_info:
        return "Failed to retrieve database schema. Check database connection and initialization."
    
    # Build comprehensive schema prompt with relationship information and sample data
    schema_prompt = "# Database Schema\n\n"
    for table_name, table_info in schema_info.items():
        schema_prompt += f"## Table: {table_name} ({table_info['row_count']} rows)\n"
        
        schema_prompt += "### Columns:\n"
        for col in table_info["columns"]:
            pk_marker = " (PK)" if col["primary_key"] else ""
            nullable = " NULL" if col["nullable"] else " NOT NULL"
            default = f" DEFAULT {col['default_value']}" if col["default_value"] else ""
            schema_prompt += f"- {col['name']}: {col['type']}{pk_marker}{nullable}{default}\n"
        
        # Foreign keys section if any exist
        if table_info["foreign_keys"]:
            schema_prompt += "### Relationships:\n"
            for fk in table_info["foreign_keys"]:
                schema_prompt += f"- {fk['from']} → {fk['to_table']}.{fk['to_column']}\n"
        
        # Sample data if available
        if table_info["sample_data"]:
            schema_prompt += "### Sample Data:\n"
            sample = table_info["sample_data"]
            for i, row in enumerate(sample):
                if i == 0:
                    schema_prompt += "```\n"
                schema_prompt += f"{row}\n"
                if i == len(sample) - 1:
                    schema_prompt += "```\n"
        
        schema_prompt += "\n"
    
    # Improve prompt with guidance for common issues
    prompt_guidance = """
# Query Requirements:
1. ALWAYS use DISTINCT when performing JOINs to avoid duplicate rows
2. Use LOWER() function for case-insensitive string comparisons 
3. Use LIKE with wildcards (%) for partial string matching
4. Format dates consistently using strftime() function
5. When appropriate, include GROUP BY for aggregation
6. Use meaningful column aliases for better readability
7. Convert raw data into insights when appropriate
8. Limit result sets to a reasonable size (max 100 rows)
9. Add proper error handling for empty result sets

# Response Format:
0. explanation must should be in normal formate and normal english give space between each word
1. A brief explanation of how you're approaching the question
2. The SQL query (clearly formatted and with comments)
3. The results in a clean, readable format (use markdown tables for structured data)
4. A plain language explanation of what the results mean
5. (If applicable) Data quality issues identified in the results
6. (If applicable) Recommendations based on the data
"""

    combined_query = f"""
{schema_prompt}

{prompt_guidance}

User Query: {query}

Respond with the information requested using the format above. Be thorough but concise. 
Explain medical terminology and SQL concepts in simple terms that non-technical users can understand.
"""
    
    start_time = time.time()
    
    try:
        agent = Agent(
            model=Groq(api_key=groq_api_key, id="meta-llama/llama-4-scout-17b-16e-instruct"),
            description="You are a medical practice database expert who helps non-technical staff understand their practice data.",
            tools=[sql_tool],
            show_tool_calls=True,
            markdown=True
        )
        
        output = agent.run(combined_query, timeout=60)
        
        sql_query_match = re.search(r'```sql\s+(.*?)\s+```', output.content, re.DOTALL)
        if sql_query_match:
            extracted_sql = sql_query_match.group(1).strip()
            
            improved_sql, duplicate_message = detect_and_handle_duplicates(extracted_sql)
            improved_sql, string_match_message = improve_string_matching(improved_sql)
            
            # if improved_sql != extracted_sql:
            #     modification_message = ""
            #     if duplicate_message:
            #         modification_message += f"\n\n> 🔍 Query improved: {duplicate_message}."
            #     if string_match_message:
            #         modification_message += f"\n\n> 🔍 Query improved: {string_match_message}."
                
            #     modified_output = re.sub(
            #         r'(```sql\s+)(.*?)(\s+```)', 
            #         f'\\1{improved_sql}\\3{modification_message}', 
            #         output.content, 
            #         flags=re.DOTALL
            #     )
            #     output.content = modified_output
        
        execution_time = time.time() - start_time
        logger.info(f"Query processed in {execution_time:.2f} seconds")
        
        # Add data quality checks
        output.content = add_data_quality_insights(output.content)
        
        return output.content
    except Exception as e:
        logger.error(f"Error in SQL agent: {str(e)}")
        return f"""
## Error Processing Query

I encountered an issue while processing your question: `{str(e)}`

### Troubleshooting suggestions:
1. Try rephrasing your question with more specific details
2. Check if you're referring to tables or columns that exist in the database
3. If asking about specific values, double-check spellings and formatting
4. For complex questions, try breaking them down into simpler parts

If the problem persists, please contact technical support.
"""

def add_data_quality_insights(content):
    """Add data quality insights to the output"""
    # Check if the output mentions specific issues
    has_empty_results = "no results" in content.lower() or "no rows" in content.lower()
    has_many_results = "many results" in content.lower() or "large number" in content.lower()
    
    insights = ""
    
    if has_empty_results:
        insights += """
### Data Quality Note
No results were found. This could be due to:
- The search criteria being too specific
- Possible data entry inconsistencies in the database
- The information may not be recorded in the system

Consider broadening your search terms or checking alternative spellings.
"""
    elif has_many_results:
        insights += """
### Data Interpretation Note
A large number of results were returned. Consider:
- Adding more specific filters to narrow your search
- Looking for patterns or trends in the data rather than individual records
- Exporting the results for further analysis if needed
"""

    # If we have insights to add, append them to the content
    if insights:
        return content + insights
    
    return content

def display_schema():
    """Display database schema in Streamlit sidebar with improved visualization"""
    schema_info = get_table_schema()
    
    if not schema_info:
        st.sidebar.warning(" Schema information could not be loaded. Please check database connection.")
        return
    
    st.sidebar.header("Database Schema")
    
    # Create expandable sections for each table
    for table_name, table_info in schema_info.items():
        with st.sidebar.expander(f"📋 {table_name} ({table_info['row_count']} rows)"):
            # Create dataframe for columns
            columns_data = []
            for col in table_info["columns"]:
                pk_marker = "✓" if col["primary_key"] else ""
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                columns_data.append([col["name"], col["type"], pk_marker, nullable])
                
            # Display columns
            df_columns = pd.DataFrame(columns_data, columns=["Column", "Type", "PK", "Nullable"])
            st.dataframe(df_columns, use_container_width=True)
            
            # Display relationships if any
            if table_info["foreign_keys"]:
                st.markdown("**Relationships:**")
                for fk in table_info["foreign_keys"]:
                    st.markdown(f"- {fk['from']} → {fk['to_table']}.{fk['to_column']}")

def sanitize_results_for_display(results):
    """Sanitize SQL results for display"""
    if not results:
        return []
    
    # Make a deep copy to avoid modifying original
    clean_results = []
    for row in results:
        clean_row = {}
        for key, value in row.items():
            # Handle None values
            if value is None:
                clean_row[key] = "NULL"
            else:
                clean_row[key] = value
        clean_results.append(clean_row)
    
    return clean_results

def check_sqltools_working():
    """Check if SQLTools is retrieving data correctly with detailed diagnostics"""
    if not sql_tool:
        st.error("SQLTools initialization failed. Check database connection settings.")
        return
    
    st.markdown("### Database Connection Status:")
    
    try:
        # Perform a simple query to check if SQLTools is working
        start_time = time.time()
        result = sql_tool.run_sql("SELECT name FROM sqlite_master WHERE type='table';")
        query_time = time.time() - start_time
        
        st.success(f"✅ Database connection successful (query executed in {query_time:.2f}s)")
        
        # Show tables in the database with row counts
        if result:
            st.markdown("**Tables found in the database:**")
            table_info = []
            for table in result:
                table_name = table["name"]
                
                # Use direct SQLite connection for count queries to avoid issues
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                conn.close()
                
                table_info.append({"Table": table_name, "Row Count": row_count})
            
            st.dataframe(pd.DataFrame(table_info), use_container_width=True)
        else:
            st.warning("No tables found in the database.")
    except Exception as e:
        st.error(f"Error connecting to the database: {str(e)}")
        st.info("Make sure the database file exists and is properly initialized.")
        logger.error(f"SQLTools check failed: {str(e)}")


def suggest_example_queries():
    """Provide example queries organized by category with unique keys for buttons"""
    st.sidebar.header("Example Queries")
    
    categories = {
        "Financial": [
            "Show me all bank statements with deposits greater than $10,000",
            "What was our profit in Q4 2024?",
            "Compare total revenue between Q3 and Q4 2024",
            "Show me our top 5 revenue-generating procedures"
        ],
        "Vendors & Suppliers": [
            "List all purchase orders from Medline Industries",
            "Show me items in the supply catalog with price greater than $1900",
            "What is the payment term for Blue Cross?",
            "List all purchase order items with unit price over $1000"
        ],
        "Practice Management": [
            "Who owns the most equity in the practice?",
            "Find all procedures covered by Aetna",
            "Show me patients with appointments next week",
            "Which doctors have the highest number of patients?"
        ]
    }
    
    selected_queries = []
    # Use a counter for unique key generation
    query_counter = 0
    
    for category, queries in categories.items():
        with st.sidebar.expander(f"📊 {category}"):
            for query in queries:
                query_counter += 1
                if st.button(query, key=f"example_query_{query_counter}"):
                    selected_queries.append(query)
    
    return selected_queries

def display_query_history():
    """Display and allow reuse of query history"""
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    if st.session_state.query_history:
        st.sidebar.header("Recent Queries")
        for i, past_query in enumerate(reversed(st.session_state.query_history[-5:])):
            if st.sidebar.button(f"🔄 {past_query[:40]}...", key=f"history_{i}"):
                return past_query
    
    return None

def main():
    # App header with logo and improved UI
    col1, col2 = st.columns([1, 5])
    #C:\Users\Admin\Downloads\istockphoto-1369900529-612x612.jpg
    with col1:
        st.image(r"C:\Users\Admin\Downloads\istockphoto-1369900529-612x612.jpg", width=80)
    with col2:
        st.title("Medical SQL Assistant")
        st.markdown("Ask Any questions about medical practice data")
    
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    st.info(f"Database path: {db_file}")
    if not os.path.exists(db_file):
        st.error("Database file not found. Please run the initialization script first.")
        st.code("python init_database.py", language="bash")
        return
    
    display_schema()
    
    # Provide connection status and diagnostics information
    with st.expander("Database Connection Status"):
        check_sqltools_working()
    
    example_selection = suggest_example_queries()
    history_selection = display_query_history()
    
    query_input = st.text_area(
        "Please Enter Your Question :", 
        placeholder="Examples:Which procedures generated the most revenue last quarter?",
        height=80,
        key="query_input"
    )
    
    # Use selection from example or history if available
    if example_selection:
        query_input = example_selection[0]  
    elif history_selection:
        query_input = history_selection
    
    # Control buttons with improved layout
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        run_query_button = st.button("🔍 Run Query", use_container_width=True)
    
  
    if run_query_button and query_input:
        if query_input not in st.session_state.query_history:
            st.session_state.query_history.append(query_input)
            # Keep only the last 20 queries
            if len(st.session_state.query_history) > 20:
                st.session_state.query_history.pop(0)
        
        # Process with progress indicator
        with st.spinner("Processing your query..."):
            start_time = time.time()
            results = sql_agent(query_input)
            processing_time = time.time() - start_time
            
            st.markdown("### Results:")
            st.markdown(results)
            
            # Show processing time for transparency
            st.caption(f"Query processed in {processing_time:.2f} seconds")
            
            # # Add option to export results if they contain tables
            # if "table" in results.lower():
            #     st.download_button(
            #         "📥 Export Results",
            #         results,
            #         file_name="query_results.md",
            #         mime="text/markdown"
            #     )
   
    
    # Improved footer with context help
    st.markdown("""
    <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-top:30px;">
        <p style="text-align:center; color:#666; font-size:0.9em">
            <br>Powered by RAGflow, Agno NL2SQL, and Llama4
        </p>
    </div>
    """, unsafe_allow_html = True)

if __name__ == '__main__':
    main()
