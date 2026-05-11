import os
import logging
import json
import time
from agno.models.groq import Groq
import os
from fastapi import HTTPException, Query
import logging
import json
from contextlib import contextmanager
from fastapi import Query
import random
import psycopg2
from psycopg2.extras import DictCursor, RealDictCursor
import re
from datetime import datetime, timedelta
from uuid import uuid4
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.postgres import PostgresMemoryDb
from agno.memory.v2.schema import UserMemory
from agno.agent import Agent
from agno.tools.postgres import PostgresTools
from agno.storage.postgres import PostgresStorage
from contextlib import contextmanager, asynccontextmanager
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import DictCursor
from psycopg2.pool import SimpleConnectionPool
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import socketio
import uvicorn
from textwrap import dedent
from rich.pretty import pprint
import boto3
# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent.env')
if not os.path.exists(env_path):
    logger.error(f"Environment file not found at {env_path}")
    raise FileNotFoundError(f"Environment file not found at {env_path}")
load_dotenv(env_path)

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     logger.error("GROQ_API_KEY is not set in environment variables.")
#     raise ValueError("GROQ_API_KEY is not set in environment variables.")

# Bedrock Configuration
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-2")
BEDROCK_INFERENCE_PROFILE_ARN = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN")
BEDROCK_INFERENCE_PROFILE_ID = os.getenv("BEDROCK_INFERENCE_PROFILE_ID") 

# PostgreSQL connection parameters
DB_NAME = os.getenv("DB_NAME", "dev_database")
DB_USER = os.getenv("DB_USER", "master_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "your_postgres_password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
LOG_USER_ID = os.getenv("LOG_USER_ID")
DB_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Schema cache configuration
SCHEMA_CACHE_FILE = "schema_cache.json"
SCHEMA_CACHE_DURATION = 3600  # 1 hour in seconds

# Global variables
memory_db = None
memory = None
connection_pool = None
knowledge_base = None
session_storage = None
active_sessions = {}  # Track active sessions per user

WORKING_TABLES = [
    "big_sky_admission_billing_schedule",
    "big_sky_anesthesia_statistics",
    "big_sky_contractual_revenue_variance",
    "big_sky_cpt_codes",
    "big_sky_employee_credentials",
    "big_sky_employees_list",
    "big_sky_financial_class_summary",
    "big_sky_item_information",
    "big_sky_payer_list",
    "big_sky_payment_trending",
    "big_sky_physician_case_billings",
    "big_sky_preference_card_list",
    "big_sky_procedure_data_with_turnover",
    "big_sky_procedure_profit_cost",
    "big_sky_procedure_summary",
    "big_sky_procedures",
    "big_sky_staff_master",
    "big_sky_staff_utilization",
    "big_sky_surgeon_master",
    "big_sky_surgery_time_log",
    "big_sky_surgical_clinical_data",
    "big_sky_visit_billing_data"
]


class KnowledgeBase:
    """PostgreSQL-based knowledge base for storing and utilizing previous conversations"""
    
    def __init__(self, db_url=DB_URL):
        self.db_url = db_url
        self.init_db()
    
    def init_db(self):
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_conversations (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        query TEXT NOT NULL,
                        sql_query TEXT,
                        result_count INTEGER,
                        success BOOLEAN,
                        timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                        execution_time REAL,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        response_data JSONB DEFAULT '{}'::jsonb
                        
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS query_patterns (
                        id SERIAL PRIMARY KEY,
                        pattern TEXT NOT NULL,
                        sql_template TEXT NOT NULL,
                        usage_count INTEGER DEFAULT 1,
                        last_used TIMESTAMP NOT NULL DEFAULT NOW(),
                        success_rate REAL DEFAULT 1.0
                    )
                """)
                conn.commit()
                logger.info("Initialized knowledge base database in PostgreSQL")
    
    @contextmanager
    def get_connection(self):
        conn = psycopg2.connect(self.db_url.replace("postgresql+psycopg", "postgresql"))
        try:
            yield conn
        finally:
            conn.close()
    
    def store_conversation(self, user_id, query, sql_query=None, result_count=0, success=True, 
                        execution_time=0, metadata=None, response_data=None):
        """
        Store conversation with robust handling of all response data formats.
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # Prepare insert data
                    insert_data = {
                        'user_id': str(user_id),
                        'query': query.strip(),
                        'sql_query': str(sql_query) if sql_query is not None else None,
                        'result_count': int(result_count),
                        'success': bool(success),
                        'execution_time': float(execution_time),
                    }

                    # Handle metadata
                    if metadata is None:
                        metadata = {}
                    insert_data['metadata'] = json.dumps(metadata, ensure_ascii=False)

                    # Handle response_data - ensure it's properly formatted
                    if response_data is None:
                        response_data = {'data': []}
                    
                    # If response_data is already in our standard format
                    if isinstance(response_data, dict) and 'data' in response_data:
                        prepared_response = response_data
                    else:
                        # Convert to our standard format
                        prepared_response = {'data': []}
                        if isinstance(response_data, list):
                            for item in response_data:
                                if isinstance(item, dict) and 'type' in item and 'content' in item:
                                    prepared_response['data'].append(item)
                                else:
                                    prepared_response['data'].append({
                                        'type': 'text',
                                        'content': {'html': str(item)}
                                    })
                        else:
                            prepared_response['data'].append({
                                'type': 'text',
                                'content': {'html': str(response_data)}
                            })

                    # Validate and normalize each data item
                    for item in prepared_response['data']:
                        if not isinstance(item, dict):
                            item = {'type': 'text', 'content': {'html': str(item)}}
                        if 'type' not in item:
                            item['type'] = 'text'
                        if 'content' not in item:
                            item['content'] = {'html': ''}
                        
                        # Ensure table format is correct
                        if item['type'] == 'table':
                            if 'headers' not in item['content']:
                                item['content']['headers'] = []
                            if 'rows' not in item['content']:
                                item['content']['rows'] = []
                        
                        # Ensure chart format is complete
                        if item['type'] == 'chart':
                            if 'chart_type' not in item['content']:
                                item['content']['chart_type'] = 'bar'
                            if 'data' not in item['content']:
                                item['content']['data'] = {'labels': [], 'datasets': []}
                            if 'options' not in item['content']:
                                item['content']['options'] = {
                                    'responsive': True,
                                    'maintainAspectRatio': False,
                                    'plugins': {
                                        'legend': {'position': 'top'},
                                        'title': {'display': True, 'text': item.get('title', 'Chart')}
                                    }
                                }

                    insert_data['response_data'] = json.dumps(prepared_response, ensure_ascii=False)

                    # Execute the insert
                    cursor.execute("""
                        INSERT INTO user_conversations 
                        (user_id, query, sql_query, result_count, success, 
                        execution_time, metadata, response_data)
                        VALUES (%(user_id)s, %(query)s, %(sql_query)s, %(result_count)s, 
                        %(success)s, %(execution_time)s, %(metadata)s, %(response_data)s)
                        RETURNING id
                    """, insert_data)
                    
                    conn.commit()
                    return cursor.fetchone()[0]

                except Exception as e:
                    conn.rollback()
                    logger.error(f"Failed to store conversation: {str(e)}")
                    raise
    
    def get_similar_queries(self, user_id, query, limit=5):
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT query, sql_query, result_count, success, timestamp
                    FROM user_conversations
                    WHERE user_id = %s AND success = TRUE AND sql_query IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (user_id, limit))
                return cursor.fetchall()
    
    def get_user_conversations(self, user_id, limit=10, offset=0):
        """
        Retrieve paginated conversations with proper handling of response data and chart reconstruction.
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get paginated conversations
                cur.execute("""
                    SELECT id, user_id, query, sql_query, result_count,
                        execution_time, success, timestamp, metadata,
                        response_data
                    FROM user_conversations
                    WHERE user_id = %s
                    ORDER BY timestamp DESC
                    LIMIT %s OFFSET %s
                """, (user_id, limit, offset))
                
                conversations = []
                for row in cur.fetchall():
                    # Parse metadata
                    metadata = {}
                    if row['metadata']:
                        try:
                            metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse metadata: {e}")
                            metadata = {'parse_error': str(e)}
                    
                    # Parse response_data
                    response_data = row.get('response_data', {})
                    if isinstance(response_data, str):
                        try:
                            response_data = json.loads(response_data)
                        except (json.JSONDecodeError, TypeError):
                            response_data = {'data': [{'type': 'text', 'content': {'html': str(response_data)}}]}
                    
                    # Ensure response_data has proper structure
                    if not isinstance(response_data, dict):
                        response_data = {'data': [{'type': 'text', 'content': {'html': str(response_data)}}]}
                    
                    # Handle different response data formats
                    if 'data' not in response_data:
                        # Check if this is the old format with sql/query keys
                        if 'sql' in response_data and 'query' in response_data:
                            # This is the new format but missing 'data' array
                            response_data = {'data': response_data.get('data', [])}
                        else:
                            # This is the old format, convert it
                            response_data = {'data': [{
                                'type': 'text',
                                'content': {'html': f"<p>Query: {row['query']}</p><p>SQL: {row['sql_query']}</p>"}
                            }]}
                    
                    # Build chat history items
                    timestamp = row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp']
                    
                    conversation = {
                        'id': row['id'],
                        'user_id': row['user_id'],
                        'query': row['query'],
                        'sql_query': row['sql_query'],
                        'result_count': row['result_count'],
                        'execution_time': row['execution_time'],
                        'success': row['success'],
                        'timestamp': timestamp,
                        'metadata': metadata,
                        'response_data': response_data,
                        'chat_history': [
                            {
                                "role": "user",
                                "content": row['query'],
                                "timestamp": timestamp,
                                "source": "knowledge_base",
                                "conversation_id": row['id']
                            }
                        ]
                    }
                    
                    # Process each data item in the response
                    html_parts = []
                    for item in response_data['data']:
                        if item['type'] == 'text' and 'content' in item and 'html' in item['content']:
                            html_parts.append(item['content']['html'])
                        elif item['type'] == 'table' and 'content' in item:
                            # Generate HTML table
                            table_content = item['content']
                            if 'headers' in table_content and 'rows' in table_content:
                                table_html = "<table class='result-table' style='width: 100%; border-collapse: collapse; margin: 20px 0;'>"
                                table_html += "<thead><tr style='background-color: #f8f9fa;'>"
                                table_html += "".join(f"<th style='border: 1px solid #dee2e6; padding: 12px; text-align: left; font-weight: bold;'>{h}</th>" for h in table_content['headers'])
                                table_html += "</tr></thead><tbody>"
                                table_html += "".join(
                                    "<tr style='border-bottom: 1px solid #dee2e6;'>" + 
                                    "".join(f"<td style='border: 1px solid #dee2e6; padding: 12px;'>{cell}</td>" for cell in row_data) + 
                                    "</tr>"
                                    for row_data in table_content['rows']
                                )
                                table_html += "</tbody></table>"
                                html_parts.append(table_html)
                        elif item['type'] == 'chart' and 'content' in item:
                            # Generate chart HTML with proper Chart.js integration
                            canvas_id = f"chart_{row['id']}_{len(html_parts)}"
                            chart_content = item.get('content', {})
                            chart_title = item.get('title', 'Chart')
                            
                            # Ensure chart has all required properties
                            chart_config = {
                                'type': chart_content.get('chart_type', 'bar'),
                                'data': chart_content.get('data', {
                                    'labels': [],
                                    'datasets': []
                                }),
                                'options': chart_content.get('options', {
                                    'responsive': True,
                                    'maintainAspectRatio': False,
                                    'plugins': {
                                        'legend': {
                                            'position': 'top',
                                        },
                                        'title': {
                                            'display': True,
                                            'text': chart_title
                                        }
                                    }
                                })
                            }
                            
                            # Fix options to ensure proper structure
                            if 'options' not in chart_config:
                                chart_config['options'] = {}
                            if 'plugins' not in chart_config['options']:
                                chart_config['options']['plugins'] = {}
                            if 'title' not in chart_config['options']['plugins']:
                                chart_config['options']['plugins']['title'] = {
                                    'display': True,
                                    'text': chart_title
                                }
                            
                            # Convert chart config to JSON string
                            chart_config_json = json.dumps(chart_config)
                            
                            chart_html = f"""
                                <div class='chart-container' style='position: relative; height: 400px; width: 100%; margin: 20px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafafa;'>
                                    <h4 style='margin: 0 0 15px 0; color: #333; text-align: center;'>{chart_title}</h4>
                                    <div style='position: relative; height: 350px; width: 100%;'>
                                        <canvas id='{canvas_id}' style='max-height: 350px;'></canvas>
                                    </div>
                                    <script>
                                        (function() {{
                                            function initChart() {{
                                                const canvas = document.getElementById('{canvas_id}');
                                                if (canvas && typeof Chart !== 'undefined') {{
                                                    const ctx = canvas.getContext('2d');
                                                    try {{
                                                        new Chart(ctx, {chart_config_json});
                                                    }} catch (error) {{
                                                        console.error('Chart creation error:', error);
                                                        canvas.parentElement.innerHTML = '<p style="text-align: center; color: #666; padding: 20px;">Chart could not be displayed</p>';
                                                    }}
                                                }} else if (canvas) {{
                                                    // Chart.js not loaded yet, try again later
                                                    setTimeout(initChart, 100);
                                                }}
                                            }}
                                            
                                            // Try to initialize immediately
                                            if (document.readyState === 'complete') {{
                                                initChart();
                                            }} else {{
                                                document.addEventListener('DOMContentLoaded', initChart);
                                                // Fallback for cases where DOMContentLoaded already fired
                                                setTimeout(initChart, 100);
                                            }}
                                        }})();
                                    </script>
                                </div>
                            """
                            html_parts.append(chart_html)
                    
                    # Add assistant response
                    conversation['chat_history'].append({
                        "role": "assistant",
                        "content": "\n".join(html_parts) if html_parts else "<p>No response content available</p>",
                        "timestamp": timestamp,
                        "source": "knowledge_base",
                        "conversation_id": row['id'],
                        "sql_query": row['sql_query'],
                        "result_count": row['result_count'],
                        "success": row['success'],
                        "metadata": metadata
                    })
                    
                    conversations.append(conversation)
                
                return conversations
def get_connection_pool():
    global connection_pool
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            connect_timeout=60,
            application_name="MedicalSQLAssistant"
        )
        logger.info("Connection pool created successfully")
        return connection_pool
    except Exception as e:
        logger.error(f"Connection pool creation failed: {e}")
        raise ValueError(f"Failed to create connection pool: {e}")

def get_postgres_connection():
    global connection_pool
    try:
        if not connection_pool:
            get_connection_pool()
        conn = connection_pool.getconn()
        with conn.cursor() as test_cursor:
            test_cursor.execute("SELECT 1")
        logger.debug("Retrieved and tested connection from pool")
        return conn
    except Exception as e:
        logger.error(f"Connection attempt failed: {e}")
        raise

def close_connection(conn):
    global connection_pool
    try:
        if connection_pool and conn:
            connection_pool.putconn(conn)
            logger.debug("Returned connection to pool")
        elif conn:
            conn.close()
            logger.debug("Closed connection")
    except Exception as e:
        logger.warning(f"Error closing connection: {e}")

def execute_sql(query, params=None):
    start_time = time.time()
    logger.debug(f"Executing SQL query: {query[:200]}...")
    conn = None
    try:
        query_upper = query.strip().upper()
        allowed_system_queries = [
            "SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN"
        ]
        
        if "PG_CONSTRAINT" in query_upper or "INFORMATION_SCHEMA" in query_upper:
            pass
        elif not any(query_upper.startswith(allowed) for allowed in allowed_system_queries):
            raise ValueError("Only SELECT and system catalog queries are allowed")
        
        conn = get_postgres_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        cursor.execute(query, params or ())
        results = [dict(row) for row in cursor.fetchall()] if cursor.description else []
        conn.commit()
        logger.debug(f"SQL query executed in {time.time() - start_time:.2f} seconds, returned {len(results)} rows")
        return results
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            close_connection(conn)

def load_schema_cache():
    try:
        if os.path.exists(SCHEMA_CACHE_FILE):
            with open(SCHEMA_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                
                if not isinstance(cache_data, dict) or 'timestamp' not in cache_data or 'schema' not in cache_data:
                    logger.warning("Invalid cache file structure, removing cache")
                    try:
                        os.remove(SCHEMA_CACHE_FILE)
                    except OSError as e:
                        logger.warning(f"Failed to remove invalid schema cache file: {e}")
                    return None
                
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                if datetime.now() - cache_time < timedelta(seconds=SCHEMA_CACHE_DURATION):
                    logger.info("Using cached schema data")
                    return cache_data['schema']
                else:
                    logger.info("Schema cache expired, will refresh")
                    try:
                        os.remove(SCHEMA_CACHE_FILE)
                    except OSError as e:
                        logger.warning(f"Failed to remove expired schema cache file: {e}")
                    return None
    except Exception as e:
        logger.warning(f"Error loading schema cache: {e}")
        try:
            if os.path.exists(SCHEMA_CACHE_FILE):
                os.remove(SCHEMA_CACHE_FILE)
        except OSError as e:
            logger.warning(f"Failed to remove schema cache file: {e}")
        return None
    return None

def save_schema_cache(schema_info):
    try:
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'schema': schema_info,
            'version': '1.0'
        }
        with open(SCHEMA_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        logger.info("Schema cached successfully")
    except Exception as e:
        logger.warning(f"Error saving schema cache: {e}")

def get_table_schema():
    cached_schema = load_schema_cache()
    if cached_schema:
        return cached_schema
    
    schema_info = {}
    conn = None
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor(cursor_factory=DictCursor)
        
        for table_name in WORKING_TABLES:
            cursor.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (table_name,))
            columns = cursor.fetchall()
            
            if not columns:
                logger.warning(f"No columns found for table: {table_name}")
                continue
            
            cursor.execute("""
                SELECT column_name
                FROM information_schema.key_column_usage k
                JOIN information_schema.table_constraints t
                ON k.constraint_name = t.constraint_name
                WHERE t.table_name = %s AND t.constraint_type = 'PRIMARY KEY'
            """, (table_name,))
            primary_keys = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("""
                SELECT kcu.column_name as from_column, ccu.table_name as to_table, ccu.column_name as to_column
                FROM information_schema.key_column_usage kcu
                JOIN information_schema.referential_constraints rc 
                    ON kcu.constraint_name = rc.constraint_name
                JOIN information_schema.constraint_column_usage ccu 
                    ON rc.unique_constraint_name = ccu.constraint_name
                WHERE kcu.table_name = %s
            """, (table_name,))
            foreign_keys = cursor.fetchall()
            
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            
            schema_info[table_name] = {
                "columns": [
                    {
                        "name": col["column_name"],
                        "type": col["data_type"],
                        "nullable": col["is_nullable"] == "YES",
                        "default_value": col["column_default"],
                        "primary_key": col["column_name"] in primary_keys
                    } for col in columns
                ],
                "foreign_keys": [
                    {
                        "from": fk["from_column"],
                        "to_table": fk["to_table"],
                        "to_column": fk["to_column"]
                    } for fk in foreign_keys
                ],
                "row_count": row_count
            }
        
        save_schema_cache(schema_info)
        logger.info(f"Schema loaded successfully for {len(schema_info)} tables")
        return schema_info
        
    except Exception as e:
        logger.error(f"Error getting table schema: {str(e)}")
        return {}
    finally:
        if conn:
            close_connection(conn)
# # Validate Bedrock configuration
if not BEDROCK_INFERENCE_PROFILE_ARN and not BEDROCK_INFERENCE_PROFILE_ID:
    logger.warning("Neither BEDROCK_INFERENCE_PROFILE_ARN nor BEDROCK_INFERENCE_PROFILE_ID is set. Using base model ID.")


class ClaudeSonnetModel:
    def __init__(self):
        self.model_id = 'anthropic.claude-3-7-sonnet-20250219-v1:0'  # Updated Claude Sonnet 3.7 model ID
        self.id = self.model_id
        self.provider = 'bedrock'
        self._response = None  # Store last response internally

    def generate(self, prompt: str) -> str:
        import json
        import boto3
        from botocore.config import Config
        import botocore.exceptions
        # Set a timeout (connect: 10s, read: 30s) and retry policy
        boto_config = Config(connect_timeout=10, read_timeout=30, retries={'max_attempts': 2})
        client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION, config=boto_config)
        try:
            body = {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "temperature": 0.2
            }

            # Determine model identifier: use inference profile ARN/ID if provided, else base model ID
            model_identifier = (
                BEDROCK_INFERENCE_PROFILE_ARN or BEDROCK_INFERENCE_PROFILE_ID or self.model_id
            )

            response = client.invoke_model(
                body=json.dumps(body).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
                modelId=model_identifier,
            )
            raw = response['body'].read().decode('utf-8')
            try:
                payload = json.loads(raw)
                # Anthropic over Bedrock returns text in payload["content"][0]["text"]
                text = ""
                if isinstance(payload, dict):
                    if "content" in payload and isinstance(payload["content"], list) and payload["content"]:
                        first = payload["content"][0]
                        if isinstance(first, dict) and first.get("type") == "text":
                            text = first.get("text", "")
                    elif "outputText" in payload:  # some runtimes
                        text = payload.get("outputText", "")
                    else:
                        text = raw
                else:
                    text = raw
            except Exception:
                text = raw

            self._response = text
            return text
        except botocore.exceptions.ClientError as e:
            self._response = None
            msg = str(e)
            if "on-demand throughput isn't supported" in msg or "on-demand throughput isn't supported" in msg:
                return (
                    "[ERROR] Bedrock model requires an inference profile. Set BEDROCK_INFERENCE_PROFILE_ARN or "
                    "BEDROCK_INFERENCE_PROFILE_ID in agent.env and restart the server."
                )
            return f"[ERROR] Bedrock model call failed: {msg}"
        except botocore.exceptions.ConnectTimeoutError:
            self._response = None
            return '[ERROR] Bedrock model connection timed out.'
        except botocore.exceptions.ReadTimeoutError:
            self._response = None
            return '[ERROR] Bedrock model read timed out.'
        except Exception as e:
            self._response = None
            return f'[ERROR] Bedrock model call failed: {str(e)}'

    def get_instructions_for_model(self, *args, **kwargs):
        return ""

    def get_system_message_for_model(self, *args, **kwargs):
        return ""

    def response(self, *args, **kwargs):
        return self._response or ""
    
def determine_chart_type(query, results):
    """Enhanced inference based on query content and data structure"""
    query_lower = query.lower()
    
    if not results or len(results) == 0:
        return 'bar'
    
    # Analyze data structure
    headers = list(results[0].keys()) if results else []
    numeric_cols = []
    categorical_cols = []
    date_cols = []
    
    # Enhanced column classification
    for col in headers:
        col_lower = col.lower()
        sample_values = [row.get(col) for row in results[:10] if row.get(col) is not None]
        
        if not sample_values:
            continue
            
        # Check for date columns
        if any(term in col_lower for term in ['date', 'time', 'year', 'month', 'day']):
            date_cols.append(col)
        else:
            # Check if numeric
            try:
                numeric_samples = [float(str(val)) for val in sample_values if val is not None]
                if len(numeric_samples) > len(sample_values) * 0.7:  # 70% numeric threshold
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except (ValueError, TypeError):
                categorical_cols.append(col)
    
    # Smart chart type selection based on content and structure
    if len(numeric_cols) >= 2:
        # Two or more numeric columns suggest scatter plot for correlation
        if any(term in query_lower for term in ['correlation', 'relationship', 'vs', 'against', 'compare']):
            return 'scatter'
    
    if date_cols or any(term in query_lower for term in ['trend', 'over time', 'timeline', 'progression', 'monthly', 'yearly', 'daily']):
        return 'line'
    
    if any(term in query_lower for term in ['distribution', 'breakdown', 'composition', 'percentage', 'proportion']) and len(results) <= 10:
        return 'pie'
    
    if any(term in query_lower for term in ['compare', 'comparison', 'versus', 'ranking']):
        return 'bar'
    
    # Default based on data size
    if len(results) <= 5:
        return 'pie'
    elif len(results) > 20:
        return 'line'
    else:
        return 'bar'

def generate_colors(count):
    """Generate attractive colors for charts with better distribution"""
    base_colors = [
        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
        '#FF9F40', '#8AC24A', '#7E57C2', '#EF5350', '#29B6F6',
        '#66BB6A', '#FFA726', '#5C6BC0', '#26C6DA', '#D4E157'
    ]
    
    if count <= len(base_colors):
        return base_colors[:count]
    
    # Generate additional colors using HSL color space for better distribution
    additional_colors = []
    hue_step = 360 / (count - len(base_colors))
    
    for i in range(count - len(base_colors)):
        hue = int(i * hue_step) % 360
        saturation = 70 + random.randint(-10, 10)
        lightness = 50 + random.randint(-10, 10)
        additional_colors.append(f'hsl({hue}, {saturation}%, {lightness}%)')
    
    return base_colors + additional_colors

def clean_label(text):
    """Convert labels to clean, readable text with better formatting"""
    if not text:
        return ""
    
    # Replace special characters with spaces
    text = re.sub(r'[_\-]+', ' ', str(text))
    
    # Handle camelCase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Convert to title case but preserve acronyms
    words = []
    for word in text.split():
        if word.isupper() and len(word) > 1:
            words.append(word)
        else:
            words.append(word.title())
    
    # Join and clean up
    text = ' '.join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def format_time_value(value):
    """Format time values with proper units and smart rounding"""
    try:
        num_value = float(value)
        if num_value >= 1440:  # Convert to days if >= 24 hours
            days = num_value / 1440
            return f"{days:.1f} days"
        elif num_value >= 60:  # Convert to hours if >= 60 minutes
            hours = num_value / 60
            return f"{hours:.1f} hrs"
        elif num_value >= 1:  # Show as minutes if >= 1 minute
            return f"{num_value:.1f} min"
        else:  # Show as seconds if < 1 minute
            seconds = num_value * 60
            return f"{seconds:.0f} sec"
    except (ValueError, TypeError):
        return str(value)
    
def format_financial_value_clean(value):
    """Format financial values with dollar sign and 2 decimal places"""
    try:
        # Remove existing formatting
        if isinstance(value, str):
            clean_val = value.replace('$', '').replace(',', '')
        else:
            clean_val = str(value)
        
        num = float(clean_val)
        return f"${num:.2f}"
    except (ValueError, TypeError):
        return str(value)
    
def is_financial_column(column_name):
    """Check if column contains financial data with better precision"""
    financial_keywords = [
        'amount', 'revenue', 'payment', 'billing', 'cost', 'price', 
        'profit', 'loss', 'fee', 'income', 'expense', 'charge',
        'bill', 'total', 'variance', 'margin', 'discount', 'tax',
        'wage', 'bonus', 'allowance', 'deduction', 'premium',
        'refund', 'settlement', 'valuation', 'balance', 'adjustment'
    ]
    
    # Explicit non-financial columns (even if they contain financial keywords)
    non_financial_columns = [
        'claim number', 'account number', 'visit number', 
        'payer id', 'id', 'reference', 'code'
    ]
    
    col_lower = str(column_name).lower().strip()
    
    # First check if it's explicitly a non-financial column
    if any(non_financial in col_lower for non_financial in non_financial_columns):
        return False
        
    # Then check for financial indicators
    return (
        any(keyword in col_lower for keyword in financial_keywords) or
        col_lower.endswith('amount') or
        col_lower.startswith('amount') or
        col_lower.endswith('value') or
        col_lower.startswith('value')
    )


def is_time_column(column_name):
    """Check if column contains time data with more indicators"""
    time_keywords = [
        'time', 'duration', 'minutes', 'hours', 'seconds', 'days',
        'turnover', 'wait', 'delay', 'processing', 'cycle', 'lead',
        'response', 'throughput', 'latency', 'interval', 'period',
        'schedule', 'arrival', 'departure', 'start', 'end', 'eta'
    ]
    col_lower = str(column_name).lower()
    return any(keyword in col_lower for keyword in time_keywords)

    
def format_number_with_units(value, column_name=None):
    """Format numbers with proper units and 2 decimal places"""
    if value is None:
        return "0"
    
    try:
        num_value = float(value)
    except (ValueError, TypeError):
        return str(value)
    
    # Check if this is a time-related column
    col_lower = str(column_name).lower() if column_name else ""
    is_time = any(term in col_lower for term in ['time', 'duration', 'minutes', 'hours', 'seconds'])
    
    if is_time:
        # Format time values
        if num_value >= 60:  # Convert to hours if >= 60 minutes
            hours = num_value / 60
            return f"{hours:.2f} hrs"
        else:
            return f"{num_value:.2f} min"
    else:
        # Regular number formatting
        return f"{num_value:.2f}"


def format_month_label(value):
    """Convert month numbers or abbreviations to full month names"""
    month_map = {
    '1': 'January', '2': 'February', '3': 'March', '4': 'April',
    '5': 'May', '6': 'June', '7': 'July', '8': 'August',
    '9': 'September', '10': 'October', '11': 'November', '12': 'December',
    '01': 'January', '02': 'February', '03': 'March', '04': 'April',
    '05': 'May', '06': 'June', '07': 'July', '08': 'August',
    '09': 'September', '10': 'October', '11': 'November', '12': 'December',
    'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April',
    'may': 'May', 'jun': 'June', 'jul': 'July', 'aug': 'August',
    'sep': 'September', 'oct': 'October', 'nov': 'November', 'dec': 'December',
    'january': 'January', 'february': 'February', 'march': 'March', 
    'april': 'April', 'june': 'June', 'july': 'July', 'august': 'August',
    'september': 'September', 'october': 'October', 'november': 'November', 'december': 'December'
    }
    
    value_str = str(value).lower().strip()
    return month_map.get(value_str, value)


def create_financial_tooltip_callback(column_name, financial_columns):
    """Create proper tooltip callback for financial data"""
    if financial_columns and column_name in financial_columns:
        return "function(context) { return context.label + ': $' + context.parsed.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}); }"
    else:
        return "function(context) { return context.label + ': ' + context.parsed.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}); }"


def create_axis_tick_callback(column_name, financial_columns, is_time_column=False):
    """Create proper axis tick callback with financial formatting and units"""
    if financial_columns and column_name in financial_columns:
        return "function(value) { return '$' + value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}); }"
    elif is_time_column:
        return "function(value) { return value >= 60 ? (value/60).toFixed(2) + ' hrs' : value.toFixed(2) + ' min'; }"
    else:
        return "function(value) { return value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}); }"


def prepare_scatter_data(results, numeric_cols, query, financial_columns=None):
    """Prepare scatter plot data with enhanced financial formatting and clean labels"""
    if len(numeric_cols) < 2:
        return None
    
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]
    
    # Check if columns are time-related
    x_is_time = any(term in x_col.lower() for term in ['time', 'duration', 'minutes', 'hours'])
    y_is_time = any(term in y_col.lower() for term in ['time', 'duration', 'minutes', 'hours'])
    
    # Try to carry a category label for each point (e.g., Financial Class)
    cat_col = None
    try:
        headers = list(results[0].keys()) if results and isinstance(results[0], dict) else []
        # Assuming categorical_cols is passed from prepare_chart_data
        categorical_cols = [col for col in headers if col not in numeric_cols and not any(term in col.lower() for term in ['date', 'time', 'year', 'month', 'day'])]
        if categorical_cols:
            cat_col = categorical_cols[0]
        if not cat_col:
            cat_col = choose_dimension_column(headers)  # Assuming choose_dimension_column is defined
            # Avoid picking numeric/date columns accidentally
            if cat_col in (x_col, y_col):
                cat_col = None
    except Exception:
        cat_col = None
    
    data_points = []
    for row in results:
        try:
            x_val_raw = str(row.get(x_col, 0))
            y_val_raw = str(row.get(y_col, 0))
            
            if financial_columns and x_col in financial_columns:
                x_val_raw = x_val_raw.replace('$', '').replace(',', '')
            if financial_columns and y_col in financial_columns:
                y_val_raw = y_val_raw.replace('$', '').replace(',', '')
                
            x_val = round(float(x_val_raw), 2)
            y_val = round(float(y_val_raw), 2)
            point = {'x': x_val, 'y': y_val}
            if cat_col:
                try:
                    point['category'] = str(row.get(cat_col, ''))
                except Exception:
                    point['category'] = ''
            data_points.append(point)
        except (ValueError, TypeError):
            continue
    
    if not data_points:
        return None
    
    # Clean labels with units
    x_label = clean_label(x_col)  # Assuming clean_label is defined
    y_label = clean_label(y_col)
    if x_is_time:
        x_label = f"{x_label} (min)"
    elif financial_columns and x_col in financial_columns:
        x_label = f"{x_label} ($)"
    if y_is_time:
        y_label = f"{y_label} (min)"
    elif financial_columns and y_col in financial_columns:
        y_label = f"{y_label} ($)"
    
    return {
        'chart_type': 'scatter',
        'data': {
            'datasets': [{
                'label': f'{x_label} vs {y_label}',
                'data': data_points,
                'backgroundColor': '#36A2EB',
                'borderColor': '#36A2EB',
                'borderWidth': 1,
                'pointRadius': 5,
                'pointHoverRadius': 8
            }]
        },
        'options': {
            'responsive': True,
            'maintainAspectRatio': False,
            'plugins': {
                'title': {
                    'display': True,
                    'text': f'Scatter Analysis: {x_label} vs {y_label}',
                    'font': {'size': 16, 'weight': 'bold'}
                },
                'legend': {
                    'display': True,
                    'position': 'top'
                },
                'tooltip': {
                    'callbacks': {
                        'label': "function(context) { return context.dataset.label + ': (' + context.parsed.x + ', ' + context.parsed.y + ')'; }",
                        'afterLabel': "function(context) { var d=context.raw||{}; return d.category ? ' ' + d.category : ''; }"
                    }
                }
            },
            'scales': {
                'x': {
                    'title': {
                        'display': True,
                        'text': x_label,
                        'font': {'size': 14}
                    },
                    'ticks': {
                        'callback': create_axis_tick_callback(x_col, financial_columns, x_is_time)  # Assuming create_axis_tick_callback is defined
                    }
                },
                'y': {
                    'title': {
                        'display': True,
                        'text': y_label,
                        'font': {'size': 14}
                    },
                    'ticks': {
                        'callback': create_axis_tick_callback(y_col, financial_columns, y_is_time)
                    },
                    'beginAtZero': True
                }
            }
        },
        'title': f'{x_label} vs {y_label} Analysis'
    }


def prepare_line_data(results, date_cols, categorical_cols, numeric_cols, chart_type, query, financial_columns=None):
    """Prepare line/area chart data with proper formatting"""
    if not numeric_cols:
        return None
    
    # Choose x-axis column
    x_col = None
    if date_cols:
        x_col = date_cols[0]
    elif categorical_cols:
        x_col = categorical_cols[0]
    elif results and len(results) > 0 and isinstance(results[0], dict):
        x_col = list(results[0].keys())[0]
    else:
        return None
    
    y_col = numeric_cols[0]
    
    # Check column types more accurately
    is_financial = False
    is_time = False
    is_percentage = False
    
    # Check if column is actually financial
    if financial_columns and y_col in financial_columns:
        is_financial = True
    elif is_financial_column(y_col):
        is_financial = True
    
    # Check for time columns
    if is_time_column(y_col):
        is_time = True
    
    # Check for percentage columns
    y_col_lower = y_col.lower()
    if any(keyword in y_col_lower for keyword in ['percent', 'percentage', '%', 'rate']):
        is_percentage = True
    
    # Extract and process data
    chart_data = []
    sample_values = []
    
    for row in results:
        try:
            x_val = str(row.get(x_col, ''))
            y_val_raw = str(row.get(y_col, 0))
            
            # Clean financial formatting
            if '$' in y_val_raw:
                y_val_raw = y_val_raw.replace('$', '').replace(',', '')
                is_financial = True  # If we see $ signs, treat as financial
            
            # Clean percentage formatting
            if '%' in y_val_raw:
                y_val_raw = y_val_raw.replace('%', '')
                is_percentage = True
            
            y_val = round(float(y_val_raw), 2)
            sample_values.append(y_val)
            
            # Format month labels
            if 'month' in x_col.lower():
                month_map = {
                    '1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr',
                    '5': 'May', '6': 'Jun', '7': 'Jul', '8': 'Aug',
                    '9': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'
                }
                x_val = month_map.get(x_val, x_val)
            
            chart_data.append((x_val, y_val))
        except (ValueError, TypeError):
            continue
    
    if not chart_data:
        return None
    
    # Sort data
    try:
        chart_data.sort(key=lambda x: float(x[0]))
    except (ValueError, TypeError):
        chart_data.sort(key=lambda x: x[0])
    
    labels = [item[0] for item in chart_data]
    values = [item[1] for item in chart_data]
    
    # Clean labels and prepare axis titles
    x_label = clean_label(x_col)
    y_label = clean_label(y_col)
    
    # Analyze data for appropriate step size
    max_value = max(sample_values) if sample_values else 0
    value_range = max(values) - min(values) if values else 0
    
    if value_range <= 10:
        step_size = 1
    elif value_range <= 50:
        step_size = 5
    elif value_range <= 100:
        step_size = 10
    elif value_range <= 500:
        step_size = 50
    elif value_range <= 1000:
        step_size = 100
    else:
        step_size = round(value_range / 10, -1)
    
    # Create appropriate formatting based on data type
    if is_financial:
        tooltip_callback = "function(context) { return context.dataset.label + ': $' + context.parsed.y.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}); }"
        y_tick_callback = "function(value) { return '$' + value.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0}); }"
        formatted_y_title = f"{y_label} ($)"
    elif is_time:
        tooltip_callback = "function(context) { var val = context.parsed.y; return context.dataset.label + ': ' + (val >= 60 ? (val/60).toFixed(1) + ' hrs' : val.toFixed(1) + ' min'); }"
        y_tick_callback = "function(value) { return value >= 60 ? (value/60).toFixed(1) + ' hrs' : value.toFixed(1) + ' min'; }"
        formatted_y_title = f"{y_label} (time)"
    elif is_percentage:
        tooltip_callback = "function(context) { return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%'; }"
        y_tick_callback = "function(value) { return value.toFixed(1) + '%'; }"
        formatted_y_title = f"{y_label} (%)"
    else:
        # Default formatting for regular numbers
        if max_value >= 1000:
            tooltip_callback = "function(context) { return context.dataset.label + ': ' + context.parsed.y.toLocaleString(); }"
            y_tick_callback = "function(value) { return value.toLocaleString(); }"
        else:
            tooltip_callback = "function(context) { return context.dataset.label + ': ' + context.parsed.y.toFixed(1); }"
            y_tick_callback = "function(value) { return value.toFixed(1); }"
        formatted_y_title = y_label
    
    dataset_config = {
        'label': y_label,
        'data': values,
        'borderColor': '#36A2EB',
        'backgroundColor': 'rgba(54, 162, 235, 0.2)' if chart_type == 'area' else '#36A2EB',
        'borderWidth': 2,
        'tension': 0.4,
        'pointRadius': 4,
        'pointHoverRadius': 6
    }
    
    if chart_type == 'area':
        dataset_config['fill'] = True
    
    return {
        'chart_type': chart_type,
        'data': {
            'labels': labels,
            'datasets': [dataset_config]
        },
        'options': {
            'responsive': True,
            'maintainAspectRatio': False,
            'plugins': {
                'title': {
                    'display': True,
                    'text': f'{y_label} by {x_label}',
                    'font': {'size': 16, 'weight': 'bold'}
                },
                'legend': {
                    'display': True,
                    'position': 'top'
                },
                'tooltip': {
                    'callbacks': {
                        'label': tooltip_callback
                    }
                }
            },
            'scales': {
                'x': {
                    'title': {
                        'display': True,
                        'text': x_label,
                        'font': {'size': 14}
                    }
                },
                'y': {
                    'title': {
                        'display': True,
                        'text': formatted_y_title,
                        'font': {'size': 14}
                    },
                    'beginAtZero': True,
                    'ticks': {
                        'callback': y_tick_callback,
                        'stepSize': step_size,
                        'maxTicksLimit': 8,
                        'precision': 0 if is_financial or max_value >= 100 else 1
                    },
                    'grid': {
                        'display': True,
                        'color': 'rgba(0,0,0,0.1)'
                    }
                }
            }
        },
        'title': f'{y_label} Trend Analysis'
    }


def prepare_pie_data(results, categorical_cols, numeric_cols, query, financial_columns=None):
    """Prepare pie chart data with proper formatting"""
    if not categorical_cols or not numeric_cols:
        return None
    
    cat_col = categorical_cols[0]
    num_col = numeric_cols[0]
    
    # Check if numeric column is financial or time
    is_financial = is_financial_column(num_col) or (financial_columns and num_col in financial_columns)
    is_time = is_time_column(num_col)
    
    # Aggregate data
    data_map = {}
    for row in results:
        try:
            category = str(row.get(cat_col, 'Unknown'))
            value_raw = str(row.get(num_col, 0))
            
            # Clean financial formatting
            if '$' in value_raw:
                value_raw = value_raw.replace('$', '').replace(',', '')
            
            value = round(float(value_raw), 2)
            data_map[category] = round(data_map.get(category, 0) + value, 2)
        except (ValueError, TypeError):
            continue
    
    if not data_map:
        return None
    
    # Sort by value descending and limit to top 10
    sorted_data = sorted(data_map.items(), key=lambda x: x[1], reverse=True)[:10]
    
    raw_labels = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]
    total = sum(values) if values else 0
    colors = generate_colors(len(raw_labels))
    
    # Clean labels and build tooltip
    labels = [clean_label(item[0]) for item in sorted_data]
    
    # Format values for display
    formatted_values = []
    for v in values:
        if is_financial:
            formatted_values.append(f"${v:,.2f}")
        elif is_time:
            formatted_values.append(f"{v:,.2f} min")
        else:
            formatted_values.append(f"{v:,.2f}")
    
    # Build labels with value and percent
    display_labels = []
    for i, lbl in enumerate(labels):
        v = values[i]
        pct = (v / total * 100.0) if total else 0
        display_labels.append(f"{lbl} ({formatted_values[i]}, {pct:.1f}%)")
    
    # Tooltip callback
    if is_financial:
        tooltip_callback = "function(context) { var total = context.dataset.data.reduce((a,b) => a + b, 0); var percentage = ((context.parsed / total) * 100).toFixed(1); return context.label + ': $' + context.parsed.toFixed(2) + ' (' + percentage + '%)'; }"
    elif is_time:
        tooltip_callback = "function(context) { var val = context.parsed; var total = context.dataset.data.reduce((a,b) => a + b, 0); var percentage = ((val / total) * 100).toFixed(1); return context.label + ': ' + (val >= 60 ? (val/60).toFixed(2) + ' hrs' : val.toFixed(2) + ' min') + ' (' + percentage + '%)'; }"
    else:
        tooltip_callback = "function(context) { var total = context.dataset.data.reduce((a,b) => a + b, 0); var percentage = ((context.parsed / total) * 100).toFixed(1); return context.label + ': ' + context.parsed.toFixed(2) + ' (' + percentage + '%)'; }"

    # Clean column names for display
    cat_label = clean_label(cat_col)
    num_label = clean_label(num_col)
    
    return {
        'chart_type': 'pie',
        'data': {
            'labels': display_labels,
            'datasets': [{
                'data': values,
                'backgroundColor': colors,
                'borderColor': '#ffffff',
                'borderWidth': 2
            }]
        },
        'options': {
            'responsive': True,
            'maintainAspectRatio': False,
            'plugins': {
                'title': {
                    'display': True,
                    'text': f'{num_label} by {cat_label}',
                    'font': {'size': 16, 'weight': 'bold'}
                },
                'legend': {
                    'display': True,
                    'position': 'right'
                },
                'tooltip': {
                    'callbacks': {
                        'label': tooltip_callback
                    }
                }
            },
            'cutout': '50%'
        },
        'title': f'{num_label} Distribution'
    }
def format_table_data(formatted_results, financial_columns=None):
    """Format table data with clean headers and proper formatting"""
    if not formatted_results:
        return [], []
    
    # Clean headers
    original_headers = list(formatted_results[0].keys())
    clean_headers = [clean_label(header) for header in original_headers]
    
    # Identify ID columns (columns containing 'id' in their name) and other non-financial numeric columns
    id_columns = [col for col in original_headers if 'id' in str(col).lower()]
    non_financial_numeric_columns = [col for col in original_headers if 
                                    not is_financial_column(col) and 
                                    any(keyword in str(col).lower() for keyword in 
                                        ['number', 'reference', 'code'])]
    
    rows = []
    for row in formatted_results:
        formatted_row = []
        for i, col in enumerate(original_headers):
            value = row.get(col, '')
            
            if value is None or value == '':
                formatted_row.append('')
            elif col in id_columns or col in non_financial_numeric_columns:
                # Format ID and other non-financial numeric columns as plain numbers
                try:
                    num = float(str(value))
                    formatted_row.append(f"{int(num)}" if num.is_integer() else str(value))
                except (ValueError, TypeError):
                    formatted_row.append(str(value))
            elif is_financial_column(col) or (financial_columns and col in financial_columns):
                formatted_row.append(format_financial_value_clean(value))
            elif is_time_column(col):
                formatted_row.append(format_time_value(value))
            else:
                # Handle other numeric values
                try:
                    num = float(str(value))
                    if num.is_integer():
                        formatted_row.append(f"{int(num):,}")
                    else:
                        formatted_num = f"{num:,.2f}"
                        if '.' in formatted_num:
                            formatted_num = formatted_num.rstrip('0').rstrip('.')
                        formatted_row.append(formatted_num)
                except (ValueError, TypeError):
                    formatted_row.append(str(value))
            
        rows.append(formatted_row)
    
    return clean_headers, rows

def prepare_bar_data(results, categorical_cols, numeric_cols, query, financial_columns=None):
    """Prepare bar chart data with proper tooltip and formatting"""
    if not categorical_cols or not numeric_cols:
        return None
    
    cat_col = categorical_cols[0]
    num_col = numeric_cols[0]
    
    # Check column types - make this more accurate
    is_financial = False
    is_time = False
    is_percentage = False
    
    # Check if column is actually financial
    if financial_columns and num_col in financial_columns:
        is_financial = True
    elif is_financial_column(num_col):
        is_financial = True
    
    # Check for time columns
    if is_time_column(num_col):
        is_time = True
    
    # Check for percentage columns
    num_col_lower = num_col.lower()
    if any(keyword in num_col_lower for keyword in ['percent', 'percentage', '%', 'rate']):
        is_percentage = True
    
    # Aggregate data
    data_map = {}
    sample_values = []
    
    for row in results:
        try:
            category = str(row.get(cat_col, 'Unknown'))
            value_raw = str(row.get(num_col, 0))
            
            # Clean financial formatting for calculation
            if '$' in value_raw:
                value_raw = value_raw.replace('$', '').replace(',', '')
                is_financial = True  # If we see $ signs, treat as financial
            
            # Clean percentage formatting
            if '%' in value_raw:
                value_raw = value_raw.replace('%', '')
                is_percentage = True
            
            value = round(float(value_raw), 2)
            data_map[category] = round(data_map.get(category, 0) + value, 2)
            sample_values.append(value)
        except (ValueError, TypeError):
            continue
    
    if not data_map:
        return None
    
    # Analyze data to determine appropriate formatting
    max_value = max(sample_values) if sample_values else 0
    min_value = min(sample_values) if sample_values else 0
    avg_value = sum(sample_values) / len(sample_values) if sample_values else 0
    
    # Sort and limit data
    sorted_data = sorted(data_map.items(), key=lambda x: x[1], reverse=True)[:15]
    
    labels = [clean_label(item[0]) for item in sorted_data]
    values = [item[1] for item in sorted_data]
    colors = generate_colors(len(labels))
    
    # Clean column labels
    cat_label = clean_label(cat_col)
    num_label = clean_label(num_col)
    
    # Determine step size for y-axis based on data range
    value_range = max(values) - min(values) if values else 0
    if value_range <= 10:
        step_size = 1
    elif value_range <= 50:
        step_size = 5
    elif value_range <= 100:
        step_size = 10
    elif value_range <= 500:
        step_size = 50
    elif value_range <= 1000:
        step_size = 100
    else:
        step_size = round(value_range / 10, -1)  # Round to nearest 10
    
    # Create appropriate formatting based on data type
    if is_financial:
        tooltip_callback = "function(context) { return context.dataset.label + ': $' + context.parsed.y.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}); }"
        y_tick_callback = f"function(value) {{ return '$' + value.toLocaleString(undefined, {{minimumFractionDigits: 0, maximumFractionDigits: 0}}); }}"
        y_axis_title = f"{num_label} ($)"
        datalabels_formatter = "function(value) { return '$' + value.toLocaleString(undefined, {minimumFractionDigits: 0, maximumFractionDigits: 0}); }"
    elif is_time:
        tooltip_callback = "function(context) { var val = context.parsed.y; return context.dataset.label + ': ' + (val >= 60 ? (val/60).toFixed(1) + ' hrs' : val.toFixed(1) + ' min'); }"
        y_tick_callback = "function(value) { return value >= 60 ? (value/60).toFixed(1) + ' hrs' : value.toFixed(1) + ' min'; }"
        y_axis_title = f"{num_label} (time)"
        datalabels_formatter = "function(value) { return value >= 60 ? (value/60).toFixed(1) + 'h' : value.toFixed(1) + 'm'; }"
    elif is_percentage:
        tooltip_callback = "function(context) { return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%'; }"
        y_tick_callback = "function(value) { return value.toFixed(1) + '%'; }"
        y_axis_title = f"{num_label} (%)"
        datalabels_formatter = "function(value) { return value.toFixed(1) + '%'; }"
    else:
        # Default formatting for regular numbers
        if max_value >= 1000:
            tooltip_callback = "function(context) { return context.dataset.label + ': ' + context.parsed.y.toLocaleString(); }"
            y_tick_callback = "function(value) { return value.toLocaleString(); }"
            datalabels_formatter = "function(value) { return value.toLocaleString(); }"
        else:
            tooltip_callback = "function(context) { return context.dataset.label + ': ' + context.parsed.y.toFixed(1); }"
            y_tick_callback = "function(value) { return value.toFixed(1); }"
            datalabels_formatter = "function(value) { return value.toFixed(1); }"
        y_axis_title = num_label
    
    return {
        'chart_type': 'bar',
        'data': {
            'labels': labels,
            'datasets': [{
                'label': num_label,
                'data': values,
                'backgroundColor': colors,
                'borderColor': colors,
                'borderWidth': 1
            }]
        },
        'options': {
            'responsive': True,
            'maintainAspectRatio': False,
            'plugins': {
                'title': {
                    'display': True,
                    'text': f'{num_label} by {cat_label}',
                    'font': {'size': 16, 'weight': 'bold'}
                },
                'legend': {
                    'display': True,
                    'position': 'top'
                },
                'tooltip': {
                    'callbacks': {
                        'label': tooltip_callback
                    }
                },
                'datalabels': {
                    'display': True,
                    'anchor': 'end',
                    'align': 'top',
                    'formatter': datalabels_formatter,
                    'font': {
                        'size': 11,
                        'weight': 'bold'
                    },
                    'color': '#333'
                }
            },
            'scales': {
                'x': {
                    'title': {
                        'display': True,
                        'text': cat_label,
                        'font': {'size': 14}
                    }
                },
                'y': {
                    'title': {
                        'display': True,
                        'text': y_axis_title,
                        'font': {'size': 14}
                    },
                    'beginAtZero': True,
                    'ticks': {
                        'callback': y_tick_callback,
                        'stepSize': step_size,
                        'maxTicksLimit': 8,
                        'precision': 0 if is_financial or max_value >= 100 else 1
                    },
                    'grid': {
                        'display': True,
                        'color': 'rgba(0,0,0,0.1)'
                    }
                }
            }
        },
        'title': f'{num_label} Analysis'
    }
# Dimension detection
DIMENSION_MARKERS = [
    'surgeon', 'physician', 'doctor', 'provider', 'specialty', 'cpt', 'procedure',
    'payer', 'financial_class', 'or_room', 'room', 'anesthesia_type', 'item', 'category', 'type', 'name'
]

COST_COMPONENT_COLUMNS = [
    'direct_cost', 'indirect_cost', 'total_cost', 'supply_cost', 'center_supply_cost', 'implant_cost',
    'staff_cost', 'labor_cost', 'anesthesia_cost'
]


def is_dimension_field(col: str) -> bool:
    cl = str(col).lower()
    return any(tok in cl for tok in DIMENSION_MARKERS)


def choose_dimension_column(headers):
    # Prefer explicit dimension fields
    for h in headers:
        if is_dimension_field(h):
            return h
    # Fallback to first non-numeric-looking column
    return headers[0] if headers else None


def has_enough_categories(labels, min_unique):
    return len({l for l in labels if l is not None}) >= min_unique
def display_name(column_name: str) -> str:
    """
    Convert snake_case column names to Title Case with proper spacing and acronym handling.
    
    Examples:
    - or_room → OR Room
    - avg_turnover_minutes → Avg Turnover Minutes  
    - case_count → Case Count
    - cpt_code → CPT Code
    - total_payments → Total Payments
    """
    if not column_name:
        return ""
    
    # Handle common acronyms that should stay uppercase
    acronyms = {
        'or': 'OR',
        'cpt': 'CPT', 
        'npi': 'NPI',
        'dos': 'DOS',
        'pt': 'PT',
        'phys': 'Phys',
        'srgy': 'Surgery',
        'rr': 'RR',
        'pacu': 'PACU',
        'anes': 'Anesthesia',
        'preop': 'Pre-op',
        'postop': 'Post-op',
        'hcpcs': 'HCPCS',
        'ftes': 'FTEs',
        'tat': 'TAT',
        'bcbs': 'BCBS',
        'medicare': 'Medicare',
        'medicaid': 'Medicaid',
        'commercial': 'Commercial'
    }
    
    # Split by underscore and process each part
    parts = column_name.lower().split('_')
    processed_parts = []
    
    for part in parts:
        if part in acronyms:
            processed_parts.append(acronyms[part])
        else:
            # Title case the part, handling special cases
            if part in ['avg', 'min', 'max', 'sum', 'count', 'total']:
                # Common aggregations
                processed_parts.append(part.title())
            elif part in ['id', 'num', 'amt', 'cost', 'time', 'date']:
                # Common abbreviations
                processed_parts.append(part.upper())
            else:
                # Regular title case
                processed_parts.append(part.title())
    
    return ' '.join(processed_parts)

def prepare_chart_data(results, query):
    """Prepare chart data with currency formatting preservation and dimension-aware selection"""
    if not results or len(results) == 0:
        return None
    
    if not isinstance(results[0], dict):
        return None
    
    # Drop total rows
    def _is_total_row(row):
        for k, v in row.items():
            try:
                s = str(v).strip().lower()
                if s in ("total", "totals", "grand total", "grand totals"):
                    return True
            except Exception:
                continue
        return False
    filtered_results = [r for r in results if isinstance(r, dict) and not _is_total_row(r)]
    results = filtered_results or results
    
    # Detect financial columns first
    financial_columns = detect_financial_columns(results, query)
    formatted_results = format_results_with_currency(results, financial_columns)
    
    if not formatted_results or len(formatted_results) == 0:
        return None
    
    headers = list(formatted_results[0].keys())
    # Suppress charts when only 1 or 2 columns are present
    if len(headers) <= 2:
        return None
    
    # Classify columns
    numeric_cols = []
    categorical_cols = []
    date_cols = []
    
    for col in headers:
        col_lower = col.lower()
        sample_values = [row.get(col) for row in formatted_results[:10] if row.get(col) is not None]
        
        if not sample_values:
            continue
            
        # Check for date columns
        if any(term in col_lower for term in ['date', 'time', 'year', 'month', 'day']):
            date_cols.append(col)
        else:
            # Check if numeric (handle currency formatting)
            try:
                numeric_samples = []
                for val in sample_values:
                    if val is not None:
                        val_str = str(val).replace('$', '').replace(',', '')
                        numeric_samples.append(float(val_str))
                
                if len(numeric_samples) > len(sample_values) * 0.7:  # 70% numeric threshold
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except (ValueError, TypeError):
                categorical_cols.append(col)
    
    # Pick dimension column (assuming choose_dimension_column is defined elsewhere)
    dim_col = choose_dimension_column(headers)
    if dim_col and dim_col not in categorical_cols and dim_col not in date_cols:
        categorical_cols.insert(0, dim_col)
    
    # If there are multiple cost components present, consider stacked bar
    present_cost_metrics = [c for c in COST_COMPONENT_COLUMNS if c in numeric_cols]  # Assuming COST_COMPONENT_COLUMNS is defined
    if dim_col and present_cost_metrics and len(present_cost_metrics) >= 2:
        # Build stacked bar
        labels = []
        series_map = {m: [] for m in present_cost_metrics}
        # Aggregate by dimension
        agg = {}
        for row in formatted_results:
            key = str(row.get(dim_col, 'Unknown'))
            if key not in agg:
                agg[key] = {m: 0.0 for m in present_cost_metrics}
            for m in present_cost_metrics:
                try:
                    val_str = str(row.get(m, 0)).replace('$', '').replace(',', '')
                    agg[key][m] += float(val_str)
                except (ValueError, TypeError):
                    pass
        # Sort by total descending and limit to top 12
        sorted_items = sorted(agg.items(), key=lambda kv: sum(kv[1].values()), reverse=True)[:12]
        labels = [k for k, _ in sorted_items]
        if has_enough_categories(labels, 3):  # Assuming has_enough_categories is defined
            for m in present_cost_metrics:
                series_map[m] = [v[m] for _, v in sorted_items]
            colors = generate_colors(len(present_cost_metrics))  # Assuming generate_colors is defined
            datasets = []
            for idx, m in enumerate(present_cost_metrics):
                pretty_m = display_name(m)  # Assuming display_name is defined
                if m in financial_columns:
                    pretty_m = f"{pretty_m} ($)"
                datasets.append({
                    'label': pretty_m,
                    'data': series_map[m],
                    'backgroundColor': colors[idx % len(colors)],
                    'stack': 'cost'
                })
            return {
                'chart_type': 'bar',
                'data': {'labels': labels, 'datasets': datasets},
                'options': {
                    'responsive': True,
                    'maintainAspectRatio': False,
                    'plugins': {
                        'title': {'display': True, 'text': f'Cost Components by {display_name(dim_col)}', 'font': {'size': 16, 'weight': 'bold'}},
                        'legend': {'display': True, 'position': 'top'}
                    },
                    'scales': {
                        'x': {'stacked': True, 'title': {'display': True, 'text': display_name(dim_col), 'font': {'size': 14}}},
                        'y': {'stacked': True, 'title': {'display': True, 'text': 'Cost ($)', 'font': {'size': 14}}, 'beginAtZero': True}
                    }
                },
                'title': f'Cost Composition by {display_name(dim_col)}'
            }
    
    chart_type = determine_chart_type(query, formatted_results)
    
    # Guard: require sufficient categories/points
    if chart_type in ['pie', 'bar']:
        cat = dim_col or (categorical_cols[0] if categorical_cols else None)
        num = numeric_cols[0] if numeric_cols else None
        if not cat or not num:
            return None
        # Aggregate by category
        data_map = {}
        for row in formatted_results:
            try:
                key = str(row.get(cat, 'Unknown'))
                val_str = str(row.get(num, 0)).replace('$', '').replace(',', '')
                val = float(val_str)
                data_map[key] = data_map.get(key, 0) + val
            except (ValueError, TypeError):
                continue
        labels = [k for k, _ in sorted(data_map.items(), key=lambda kv: kv[1], reverse=True)]
        if not has_enough_categories(labels, 3):
            return None
        # Limit and build
        trimmed = labels[:10]
        values = [data_map[k] for k in trimmed]
        if chart_type == 'pie':
            return prepare_pie_data([{cat: k, num: data_map[k]} for k in trimmed], [cat], [num], query, financial_columns)
        else:
            return prepare_bar_data([{cat: k, num: data_map[k]} for k in trimmed], [cat], [num], query, financial_columns)
    
    if chart_type in ['line', 'area']:
        if not date_cols:
            return None
        conf = prepare_line_data(formatted_results, date_cols, categorical_cols, numeric_cols, chart_type, query, financial_columns)
        if conf and len(conf.get('data', {}).get('labels', [])) >= 4:
            return conf
        return None
    
    if chart_type == 'scatter':
        conf = prepare_scatter_data(formatted_results, numeric_cols, query, financial_columns)
        if conf and len(conf.get('data', {}).get('datasets', [{}])[0].get('data', [])) >= 3:
            return conf
        return None
    
    # Default: try bar with safeguards
    cat = dim_col or (categorical_cols[0] if categorical_cols else None)
    num = numeric_cols[0] if numeric_cols else None
    if not cat or not num:
        return None
    return prepare_bar_data(formatted_results, [cat], [num], query, financial_columns)

def prepare_chart_data(results, query):
    """Prepare chart data with currency formatting preservation"""
    # Detect financial columns first
    financial_columns = detect_financial_columns(results, query)
    formatted_results = format_results_with_currency(results, financial_columns)
    
    if not formatted_results or len(formatted_results) == 0:
        return None
    
    headers = list(formatted_results[0].keys())
    if len(headers) < 2:
        return None
    
    # Classify columns
    numeric_cols = []
    categorical_cols = []
    date_cols = []
    
    for col in headers:
        col_lower = col.lower()
        sample_values = [row.get(col) for row in formatted_results[:10] if row.get(col) is not None]
        
        if not sample_values:
            continue
            
        # Check for date columns
        if any(term in col_lower for term in ['date', 'time', 'year', 'month', 'day']):
            date_cols.append(col)
        else:
            # Check if numeric (handle currency formatting)
            try:
                numeric_samples = []
                for val in sample_values:
                    if val is not None:
                        val_str = str(val).replace('$', '').replace(',', '')
                        numeric_samples.append(float(val_str))
                
                if len(numeric_samples) > len(sample_values) * 0.7:  # 70% numeric threshold
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            except (ValueError, TypeError):
                categorical_cols.append(col)
    
    chart_type = determine_chart_type(query, formatted_results)
    
    # Pass financial_columns to all chart preparation functions
    if chart_type == 'scatter':
        return prepare_scatter_data(formatted_results, numeric_cols, query, financial_columns)
    elif chart_type in ['line', 'area']:
        return prepare_line_data(formatted_results, date_cols, categorical_cols, numeric_cols, chart_type, query, financial_columns)
    elif chart_type == 'pie':
        return prepare_pie_data(formatted_results, categorical_cols, numeric_cols, query, financial_columns)
    else:  # bar chart
        return prepare_bar_data(formatted_results, categorical_cols, numeric_cols, query, financial_columns)



# Add these helper functions if they don't exist in your original code
def detect_financial_columns(results, query):
    """Detect columns that contain financial data"""
    if not results or len(results) == 0:
        return set()
    
    financial_keywords = ['price', 'financial_columns', 'cost', 'amount', 'fee', 'payment', 'salary', 'revenue', 'income', 'expense', 'charge', 'bill', 'total']
    headers = list(results[0].keys())
    financial_columns = set()
    
    for header in headers:
        header_lower = header.lower()
        
        # Check if header contains financial keywords
        if any(keyword in header_lower for keyword in financial_keywords):
            financial_columns.add(header)
            continue
        
        # Check if values look like currency (contain $ or are large numbers)
        sample_values = [row.get(header) for row in results[:5] if row.get(header) is not None]
        for val in sample_values:
            val_str = str(val)
            if '$' in val_str or (val_str.replace(',', '').replace('.', '').isdigit() and float(val_str.replace(',', '')) > 100):
                financial_columns.add(header)
                break
    
    return financial_columns


def format_financial_value(value):
    """Format a value as currency if it's numeric"""
    try:
        # Remove existing formatting
        clean_val = str(value).replace('$', '').replace(',', '')
        num = float(clean_val)
        return f"${num:,.2f}"
    except (ValueError, TypeError):
        return str(value)


# Replace your format_results_with_currency function
def format_results_with_currency(results, financial_columns):
    """Format financial columns and time columns with proper formatting"""
    if not results:
        return results
    
    formatted_results = []
    for row in results:
        formatted_row = {}
        for k, v in row.items():
            if v is None:
                formatted_row[k] = v
            elif is_financial_column(k) or k in (financial_columns or []):
                formatted_row[k] = format_financial_value_clean(v)
            elif is_time_column(k):
                formatted_row[k] = format_time_value(v)
            else:
                # Round other numeric values to 2 decimal places
                try:
                    num_val = float(v)
                    formatted_row[k] = f"{num_val:.2f}"
                except (ValueError, TypeError):
                    formatted_row[k] = v
        formatted_results.append(formatted_row)
    
    return formatted_results

async def generate_ai_analysis(query, results, sql_query, execution_time, schema_info):
        """Generate comprehensive AI analysis using Agno agent"""
        try:
            # Prepare data summary for the AI agent
            data_summary = ""
            if results:
                result_count = len(results)
                headers = list(results[0].keys())
                
                # Sample data for AI analysis
                sample_data = results[:5] if len(results) > 5 else results
                
                # Numeric analysis
                numeric_stats = {}
                for header in headers:
                    numeric_values = []
                    for row in results:
                        try:
                            val = float(str(row.get(header, 0)))
                            if val != 0:
                                numeric_values.append(val)
                        except (ValueError, TypeError):
                            continue
                    
                    if len(numeric_values) > 0:
                        numeric_stats[header] = {
                            'count': len(numeric_values),
                            'total': sum(numeric_values),
                            'average': sum(numeric_values) / len(numeric_values),
                            'min': min(numeric_values),
                            'max': max(numeric_values)
                        }
                
                data_summary = f"""
    QUERY ANALYSIS CONTEXT:
    - Original Query: {query}
    - SQL Generated: {sql_query}
    - Records Found: {result_count}
    - Execution Time: {execution_time:.2f} seconds
    - Data Columns: {', '.join(headers)}

    SAMPLE DATA (first 3 records):
    {json.dumps(sample_data[:3], indent=2, default=str)}

    NUMERIC STATISTICS:
    {json.dumps(numeric_stats, indent=2, default=str) if numeric_stats else 'No numeric data found'}
    """
            else:
                data_summary = f"""
    QUERY ANALYSIS CONTEXT:
    - Original Query: {query}
    - SQL Generated: {sql_query}
    - Records Found: 0
    - Execution Time: {execution_time:.2f} seconds
    - Result: No data matched the query criteria
    """

            # Enhanced AI analysis prompt
            analysis_prompt = f"""
    You are an expert medical practice data analyst and AI assistant. Analyze the following query results and provide a comprehensive, friendly, and insightful response that makes users feel they're interacting with an intelligent ChatGPT-like assistant.

    {data_summary}

    Please provide a comprehensive analysis in the following JSON format:

    {{
        "executive_summary": "A friendly, conversational 2-3 sentence overview of what the data shows",
        "key_insights": [
            "Insight 1 with specific numbers and context",
            "Insight 2 with clinical or business implications",
            "Insight 3 with comparative analysis"
        ],
        "parameter_explanations": [
            {{
                "parameter": "column_name",
                "explanation": "What this parameter means in simple terms",
                "clinical_significance": "Why this matters for medical practice",
                "sample_interpretation": "What the actual values in this data mean"
            }}
        ],
        "deeper_analysis": {{
            "patterns_discovered": "What patterns or trends you notice",
            "performance_indicators": "How the practice is performing based on this data",
            "benchmark_context": "How these numbers compare to typical medical practice standards"
        }},
        "actionable_recommendations": [
            "Specific action item 1 with rationale",
            "Strategic recommendation 2 with expected impact",
            "Operational improvement 3 with implementation steps"
        ],
        "follow_up_questions": [
            "Thoughtful question 1 to explore related analysis",
            "Strategic question 2 to dive deeper into insights",
            "Comparative question 3 to expand understanding"
        ],
        "data_quality_assessment": {{
            "completeness": "Assessment of data completeness",
            "reliability": "How reliable these insights are",
            "limitations": "What limitations to consider"
        }}
    }}

    Guidelines for your analysis:
    1. Be conversational and friendly, like ChatGPT
    2. Use specific numbers and percentages from the actual data
    3. Provide medical practice context and industry insights
    4. Make complex data accessible to non-technical users
    5. Include actionable recommendations
    6. Suggest meaningful follow-up questions
    7. Be encouraging and highlight positive findings when appropriate
    8. If no data found, provide helpful suggestions for alternative queries

    Focus on making the user feel this is a sophisticated AI that understands their medical practice data deeply.
    """

            # Initialize analysis agent
            analysis_agent = Agent(
                name="Medical Data Analysis Expert",
                model=ClaudeSonnetModel(),
                description="Expert medical practice data analyst providing comprehensive insights",
                instructions="You are a friendly, knowledgeable AI assistant specializing in medical practice data analysis. Provide detailed, actionable insights that help medical professionals make better decisions."
            )

            # Generate AI analysis
            analysis_result = analysis_agent.run(analysis_prompt, timeout=45)
            
            # Parse the JSON response
            try:
                analysis_data = json.loads(analysis_result.content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                analysis_data = {
                    "executive_summary": "I've analyzed your query results and found valuable insights in your medical practice data.",
                    "key_insights": ["Data analysis completed successfully"],
                    "parameter_explanations": [],
                    "deeper_analysis": {"patterns_discovered": "Analysis completed", "performance_indicators": "Data processed", "benchmark_context": "Results available"},
                    "actionable_recommendations": ["Review the detailed results below"],
                    "follow_up_questions": ["Would you like to explore any specific aspect of this data?"],
                    "data_quality_assessment": {"completeness": "Good", "reliability": "High", "limitations": "Standard data limitations apply"}
                }
                
            return analysis_data
            
        except Exception as e:
            logger.error(f"AI analysis generation failed: {str(e)}")
            # Return a basic analysis structure
            return {
                "executive_summary": f"I've processed your query and found {len(results) if results else 0} records that match your criteria.",
                "key_insights": ["Query executed successfully", "Data retrieved from medical practice database"],
                "parameter_explanations": [],
                "deeper_analysis": {"patterns_discovered": "Analysis in progress", "performance_indicators": "Data available for review", "benchmark_context": "Context analysis available"},
                "actionable_recommendations": ["Review the detailed results in the table below"],
                "follow_up_questions": ["What specific aspect would you like to explore further?"],
                "data_quality_assessment": {"completeness": "Available", "reliability": "Standard", "limitations": "Review data context"}
            }
        
def format_response_as_objects(query, sql_query, results, execution_time, explanation=""):
    """Enhanced response formatting with AI-generated comprehensive analysis and financial support"""
    # Detect financial columns first
    financial_columns = detect_financial_columns(results, query)
    formatted_results = format_results_with_currency(results, financial_columns)
    objects = []
    
    # Handle no results case
    if not results:
        objects.append({
            "type": "text",
            "title": "No Results",
            "content": {
                "html": """
                    <div style="text-align: center; padding: 30px; background: #fff3cd; border-radius: 8px; border: 1px solid #ffeaa7;">
                        <h4 style="color: #856404; margin-top: 0;">⚠️ No Data Found</h4>
                        <p style="color: #856404;">No records match your query criteria. Try adjusting your search parameters.</p>
                    </div>
                """
            }
        })
        return objects
    
    try:
        # Generate AI-powered comprehensive analysis with error handling
        try:
            ai_analysis = generate_ai_analysis(query, formatted_results, sql_query, execution_time, financial_columns)
            
            # Ensure ai_analysis is a dictionary
            if isinstance(ai_analysis, str):
                try:
                    ai_analysis = json.loads(ai_analysis)
                except json.JSONDecodeError:
                    ai_analysis = {
                        'executive_summary': ai_analysis,
                        'key_insights': [],
                        'recommendations': []
                    }
        except Exception as e:
            logger.error(f"Error generating AI analysis: {str(e)}")
            ai_analysis = {
                'executive_summary': f"Analysis of {len(results)} records",
                'key_insights': [],
                'recommendations': []
            }

        # Determine response format and create visualizations
        response_format = determine_response_format(query, formatted_results)
        
        # Create chart if applicable
        chart_data = prepare_chart_data(formatted_results, query)
        if chart_data and response_format in ['chart', 'both']:
            full_chart_config = {
                "type": "chart",
                "title": chart_data.get('title', 'Data Visualization'),
                "content": {
                    "chart_type": chart_data['chart_type'],
                    "data": chart_data['data'],
                    "options": chart_data['options']
                }
            }
            objects.append(full_chart_config)
        
        # Add table if requested or as fallback
        if response_format in ['table', 'both'] or (response_format == 'chart' and not chart_data):
            clean_headers, formatted_rows = format_table_data(formatted_results, financial_columns)
            
            objects.append({
                "type": "table",
                "title": f"Data Table ({len(results)} records)",
                "content": {
                    "headers": clean_headers,
                    "rows": formatted_rows,
                    "financial_columns": list(financial_columns) if financial_columns else []
                }
            })
        
        # Add AI-generated explanation
        if ai_analysis.get('explanation') or ai_analysis.get('executive_summary'):
            explanation_content = ai_analysis.get('explanation') or ai_analysis.get('executive_summary', 'Analysis completed successfully.')
            explanation_html = f"""
            <div style='background: #f8f9ff; padding: 25px; border-radius: 12px; border-left: 5px solid #4CAF50; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h3 style='color: #2E7D32; margin-top: 0; display: flex; align-items: center; font-size: 18px;'>
                    <span style='margin-right: 12px;'>🔍</span> Data Analysis & Insights
                </h3>
                <div style='line-height: 1.7; font-size: 15px; color: #444;'>
                    {explanation_content}
                </div>
            </div>
            """
            objects.append({
                "type": "text",
                "title": "AI Analysis",
                "content": {"html": explanation_html}
            })
        
        # Add detailed breakdown of results
        if ai_analysis.get('breakdown') or ai_analysis.get('detailed_breakdown'):
            breakdown_content = ai_analysis.get('breakdown') or ai_analysis.get('detailed_breakdown', 'Detailed analysis in progress.')
            breakdown_html = f"""
            <div style='background: #e8f5e8; padding: 25px; border-radius: 12px; border-left: 5px solid #66BB6A; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h3 style='color: #2E7D32; margin-top: 0; display: flex; align-items: center; font-size: 18px;'>
                    <span style='margin-right: 12px;'>📊</span> Detailed Results Breakdown
                </h3>
                <div style='line-height: 1.7; font-size: 15px; color: #444;'>
                    {breakdown_content}
                </div>
            </div>
            """
            objects.append({
                "type": "text",
                "title": "Results Breakdown",
                "content": {"html": breakdown_html}
            })
        
        # Add key insights and patterns
        if ai_analysis.get('insights') or ai_analysis.get('key_insights'):
            insights_content = ai_analysis.get('insights') or ai_analysis.get('key_insights', 'Key insights are being analyzed.')
            
            if isinstance(insights_content, list):
                insights_text = '<ul style="margin: 0; padding-left: 20px;">'
                for insight in insights_content:
                    insights_text += f'<li style="margin-bottom: 10px;">{insight}</li>'
                insights_text += '</ul>'
            else:
                insights_text = str(insights_content)
                
            insights_html = f"""
            <div style='background: #fff3e0; padding: 25px; border-radius: 12px; border-left: 5px solid #FF9800; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h3 style='color: #E65100; margin-top: 0; display: flex; align-items: center; font-size: 18px;'>
                    <span style='margin-right: 12px;'>💡</span> Key Insights & Patterns
                </h3>
                <div style='line-height: 1.7; font-size: 15px; color: #444;'>
                    {insights_text}
                </div>
            </div>
            """
            objects.append({
                "type": "text",
                "title": "Key Insights",
                "content": {"html": insights_html}
            })
        
        # Add recommendations and next steps
        if ai_analysis.get('recommendations') or ai_analysis.get('actionable_recommendations'):
            recommendations_content = ai_analysis.get('recommendations') or ai_analysis.get('actionable_recommendations', 'Recommendations are being generated.')
            
            if isinstance(recommendations_content, list):
                recommendations_text = '<ul style="margin: 0; padding-left: 20px;">'
                for rec in recommendations_content:
                    recommendations_text += f'<li style="margin-bottom: 10px;">{rec}</li>'
                recommendations_text += '</ul>'
            else:
                recommendations_text = str(recommendations_content)
                
            recommendations_html = f"""
            <div style='background: #f3e5f5; padding: 25px; border-radius: 12px; border-left: 5px solid #9C27B0; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h3 style='color: #6A1B9A; margin-top: 0; display: flex; align-items: center; font-size: 18px;'>
                    <span style='margin-right: 12px;'>🎯</span> Recommendations & Action Items
                </h3>
                <div style='line-height: 1.7; font-size: 15px; color: #444;'>
                    {recommendations_text}
                </div>
            </div>
            """
            objects.append({
                "type": "text",
                "title": "Recommendations",
                "content": {"html": recommendations_html}
            })
        
        # Add follow-up questions
        if ai_analysis.get('follow_up_questions'):
            follow_up_content = ai_analysis.get('follow_up_questions', [])
            
            if isinstance(follow_up_content, list):
                follow_up_text = '<ul style="margin: 0; padding-left: 20px;">'
                for question in follow_up_content:
                    follow_up_text += f'<li style="margin-bottom: 10px;">{question}</li>'
                follow_up_text += '</ul>'
            else:
                follow_up_text = str(follow_up_content)
                
            follow_up_html = f"""
            <div style='background: #e3f2fd; padding: 25px; border-radius: 12px; border-left: 5px solid #2196F3; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h3 style='color: #1976D2; margin-top: 0; display: flex; align-items: center; font-size: 18px;'>
                    <span style='margin-right: 12px;'>❓</span> Suggested Follow-up Questions
                </h3>
                <div style='line-height: 1.7; font-size: 15px; color: #444;'>
                    {follow_up_text}
                </div>
                <div style='margin-top: 15px; padding: 15px; background: rgba(33, 150, 243, 0.1); border-radius: 8px; font-style: italic; font-size: 14px; color: #1976D2;'>
                    💬 Feel free to ask any of these questions to dive deeper into your data!
                </div>
            </div>
            """
            objects.append({
                "type": "text",
                "title": "Follow-up Questions",
                "content": {"html": follow_up_html}
            })
        
        # Add technical details (SQL explanation)
        if explanation or sql_query:
            technical_html = """
            <div style='background: #fafafa; padding: 25px; border-radius: 12px; border-left: 5px solid #607D8B; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h3 style='color: #37474F; margin-top: 0; display: flex; align-items: center; font-size: 18px;'>
                    <span style='margin-right: 12px;'>🔧</span> Technical Implementation
                </h3>
                <div style='margin: 15px 0;'>
            """
            
            if explanation:
                formatted_explanation = explanation.replace('\n', '<br>')
                technical_html += f"""
                    <div style='line-height: 1.6; font-size: 14px; color: #555; margin-bottom: 20px;'>
                        <h4 style='color: #37474F; margin-bottom: 10px;'>Query Processing Steps:</h4>
                        {formatted_explanation}
                    </div>
                """
            
            if sql_query:
                formatted_sql = sql_query.strip()
                if formatted_sql.startswith('```sql'):
                    formatted_sql = formatted_sql[6:]
                if formatted_sql.endswith('```'):
                    formatted_sql = formatted_sql[:-3]
                formatted_sql = formatted_sql.strip()
                
                technical_html += f"""
                    <details style='margin-top: 15px; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;'>
                        <summary style='cursor: pointer; font-weight: bold; color: #37474F; padding: 12px 15px; background: #f5f5f5; user-select: none; transition: background-color 0.2s;'>
                            <span style='margin-right: 8px;'>📝</span> SQL Query Used
                        </summary>
                        <div style='background: #1e1e1e; color: #ffffff; padding: 20px; font-family: "Courier New", Monaco, monospace; font-size: 13px; line-height: 1.4; overflow-x: auto;'>
                            <code style='color: #ffffff; white-space: pre;'>{formatted_sql}</code>
                        </div>
                    </details>
                """

            technical_html += "</div></div>"
            objects.append({
                "type": "text",
                "title": "Technical Details",
                "content": {"html": technical_html}
            })

        # Add enhanced summary with AI insights
        summary_stats = ai_analysis.get('summary_stats', {})
        
        # Calculate financial summary if financial columns detected
        financial_summary = ""
        if financial_columns:
            try:
                total_financial = 0
                financial_count = 0
                for col in financial_columns:
                    for row in results:
                        val = row.get(col)
                        if val is not None:
                            try:
                                clean_val = str(val).replace('$', '').replace(',', '')
                                total_financial += float(clean_val)
                                financial_count += 1
                            except (ValueError, TypeError):
                                continue
                
                if financial_count > 0:
                    avg_financial = total_financial / financial_count
                    financial_summary = f"""
                    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; backdrop-filter: blur(10px);'>
                        <div style='font-size: 24px; font-weight: bold; margin-bottom: 5px;'>${total_financial:,.2f}</div>
                        <div style='font-size: 14px; opacity: 0.9;'>Total Financial Value</div>
                    </div>
                    """
            except Exception as e:
                logger.error(f"Financial summary calculation failed: {str(e)}")
        
        summary_html = f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 12px; margin-top: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
            <h3 style='color: white; margin-top: 0; display: flex; align-items: center; font-size: 18px;'>
                <span style='margin-right: 12px;'>📈</span> Executive Summary
            </h3>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;'>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; backdrop-filter: blur(10px);'>
                    <div style='font-size: 24px; font-weight: bold; margin-bottom: 5px;'>{len(results):,}</div>
                    <div style='font-size: 14px; opacity: 0.9;'>Records Found</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; backdrop-filter: blur(10px);'>
                    <div style='font-size: 24px; font-weight: bold; margin-bottom: 5px;'>{execution_time:.2f}s</div>
                    <div style='font-size: 14px; opacity: 0.9;'>Query Time</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; backdrop-filter: blur(10px);'>
                    <div style='font-size: 24px; font-weight: bold; margin-bottom: 5px;'>{chart_data['chart_type'].title() if chart_data else 'Table'}</div>
                    <div style='font-size: 14px; opacity: 0.9;'>Visualization</div>
                </div>
                <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; backdrop-filter: blur(10px);'>
                    <div style='font-size: 24px; font-weight: bold; margin-bottom: 5px;'>{summary_stats.get("confidence_score", "High")}</div>
                    <div style='font-size: 14px; opacity: 0.9;'>Data Quality</div>
                </div>
                {financial_summary}
            </div>
            {f'<div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px; font-size: 15px; line-height: 1.6;">{summary_stats.get("key_takeaway", "Analysis completed successfully.")}</div>' if summary_stats.get("key_takeaway") else ''}
        </div>
        """
        
        objects.append({
            "type": "text",
            "title": "Summary",
            "content": {"html": summary_html}
        })
        
    except Exception as e:
        logger.error(f"Error formatting response objects: {str(e)}")
   
    return objects


def prepare_data_summary(results):
    """Prepare a concise summary of the data for AI analysis"""
    if not results:
        return "No data available"
    
    headers = list(results[0].keys())
    numeric_cols = []
    categorical_cols = []
    
    # Categorize columns
    for header in headers:
        sample_values = [row.get(header) for row in results[:5]]
        if any(isinstance(val, (int, float)) for val in sample_values):
            numeric_cols.append(header)
        else:
            categorical_cols.append(header)
    
    summary = f"Dataset with {len(results)} records and {len(headers)} columns.\n"
    
    if numeric_cols:
        summary += f"Numeric columns: {', '.join(numeric_cols)}\n"
        
        # Add basic stats for first numeric column
        first_numeric = numeric_cols[0]
        values = [float(row.get(first_numeric, 0)) for row in results if row.get(first_numeric) is not None]
        if values:
            summary += f"{first_numeric} range: {min(values):,.2f} to {max(values):,.2f}, avg: {sum(values)/len(values):,.2f}\n"
    
    if categorical_cols:
        summary += f"Categorical columns: {', '.join(categorical_cols)}\n"
        
        # Add unique count for first categorical column
        if categorical_cols:
            first_cat = categorical_cols[0]
            unique_values = set(str(row.get(first_cat, '')) for row in results)
            summary += f"{first_cat} has {len(unique_values)} unique values\n"
    
    return summary

# UPDATED: generate_ai_analysis function
def generate_ai_analysis(query, results, sql_query, execution_time, financial_columns=None, chart_data=None):
    """Generate comprehensive AI analysis using Agno AI agent"""
    try:
        # Prepare data summary for AI analysis
        data_summary = prepare_data_summary(results)
        chart_info = f"Chart Type: {chart_data['chart_type']}" if chart_data else "No visualization"
        
        # Enhanced analysis prompt that ensures proper section formatting
        analysis_prompt = f"""
        You are an expert medical practice data analyst. Analyze the following query results and provide a comprehensive analysis.

        **Query Context:**
        - Original Query: {query}
        - Results Count: {len(results) if results else 0}
        - Execution Time: {execution_time:.2f} seconds
        - Visualization: {chart_info}
        """
        
        # Add sample data for context
        if results and len(results) > 0:
            sample_size = min(3, len(results))
            analysis_prompt += f"\n**Sample Data (first {sample_size} records):**\n"
            for i, row in enumerate(results[:sample_size]):
                analysis_prompt += f"Record {i+1}: {dict(row)}\n"
        
        # Add financial context
        if financial_columns:
            analysis_prompt += f"\n**Financial Columns Detected:** {', '.join(financial_columns)}"
        
        analysis_prompt += """
        
        Please provide your analysis in EXACTLY this format with these section headers:

        ## EXECUTIVE_SUMMARY
        Provide a clear, conversational 2-3 sentence overview of what the data shows. Be friendly and engaging.
        ## PARAMETER_EXPLANATIONS
        For each data column, explain:
        - What this parameter means in medical practice terms
        - Why it's important for healthcare operations
        - How to interpret the values
        - Normal ranges or benchmarks when applicable
        - keep informative
        
        ## DETAILED_BREAKDOWN
        Explain the key patterns, trends, and important details in the data. Include specific numbers and percentages where relevant.

        ## KEY_INSIGHTS
        • List 3-5 key insights as bullet points
        • Each insight should be specific and actionable
        • Include numerical data where possible
        • Focus on business implications
        • dont tell about execution time
        

        ## RECOMMENDATIONS
        • Provide 3-5 specific, actionable recommendations
        • Focus on practical steps they can take
        • Prioritize high-impact actions
        • Consider both short-term and strategic goals

        ## FOLLOW_UP_QUESTIONS
        • Suggest 4-5 intelligent follow-up questions
        • Make them specific to this data
        • Encourage deeper exploration
        • Consider different analytical angles

        ## SUMMARY_STATS
        confidence_score: High
        key_takeaway: One powerful sentence summarizing the most important insight from this data.

        Make your response engaging, specific, and actionable. Use actual numbers from the data when possible.
        """

        # Initialize AI agent
        analysis_agent = Agent(
            name="Medical Data Analyst",
            model=ClaudeSonnetModel(),
            description="Expert medical practice data analyst providing comprehensive insights",
            instructions="""You are an expert data analyst specializing in medical practice operations. 
            Provide detailed, actionable insights in a friendly, conversational tone. 
            Always use the exact section headers requested and format responses clearly."""
        )
        
        # Generate AI analysis
        result = analysis_agent.run(analysis_prompt, timeout=45)
        
        # Parse the AI response into structured format
        analysis_sections = parse_ai_analysis_response(result.content)
        
        return analysis_sections
        
    except Exception as e:
        logger.error(f"Error generating AI analysis: {str(e)}")
        # Return comprehensive fallback
        return {
            'executive_summary': f"I've analyzed your query results and found {len(results) if results else 0} records with valuable insights for your medical practice.",
            'detailed_breakdown': "This dataset contains important information that can help inform your operational decisions. The data shows clear patterns that warrant further investigation.",
            'key_insights': [
                f"Query successfully retrieved {len(results) if results else 0} records from your database",
                "Data contains both financial and operational metrics",
                "Results show patterns that can guide business decisions",
                "Multiple data points available for comparative analysis"
            ],
            'recommendations': [
                "Review the detailed data table for specific values and trends",
                "Consider comparing this data with historical periods",
                "Look for outliers or unusual patterns that may need attention",
                "Use this analysis as a baseline for future performance tracking"
            ],
            'follow_up_questions': [
                "How does this data compare to the same period last year?",
                "What are the trends over the last 6 months for these metrics?",
                "Can you show me the top performers in this category?",
                "Are there any seasonal patterns I should be aware of?"
            ],
            'summary_stats': {
                'confidence_score': 'High',
                'key_takeaway': 'Your data analysis reveals actionable insights that can drive informed decision-making for your medical practice.'
            }
        }
        
# FIXED FUNCTION: parse_ai_analysis_response
def parse_ai_analysis_response(ai_response):
    """Parse the AI analysis response into structured sections"""
    try:
        analysis_dict = {}
        
        # Split response by section headers
        sections = {
            'executive_summary': ['## EXECUTIVE_SUMMARY', 'executive_summary', 'explanation'],
            'detailed_breakdown': ['## DETAILED_BREAKDOWN', 'detailed_breakdown', 'breakdown'],
            'key_insights': ['## KEY_INSIGHTS', 'key_insights', 'insights'],
            'recommendations': ['## RECOMMENDATIONS', 'actionable_recommendations', 'recommendations'],
            'follow_up_questions': ['## FOLLOW_UP_QUESTIONS', 'follow_up_questions'],
            'summary_stats': ['## SUMMARY_STATS', 'summary_stats']
        }
        
        # Parse each section
        for section_key, section_markers in sections.items():
            content = None
            
            # Try to find content by section markers
            for marker in section_markers:
                if marker.upper() in ai_response.upper():
                    # Find content between this marker and next section
                    start_idx = ai_response.upper().find(marker.upper())
                    if start_idx != -1:
                        start_idx += len(marker)
                        # Find next section or end
                        next_section_idx = len(ai_response)
                        for next_marker in ['## ', '\n##']:
                            temp_idx = ai_response.find(next_marker, start_idx + 10)
                            if temp_idx != -1 and temp_idx < next_section_idx:
                                next_section_idx = temp_idx
                        
                        content = ai_response[start_idx:next_section_idx].strip()
                        break
            
            # Clean up content
            if content:
                content = content.replace('##', '').strip()
                # Convert bullet points to list if needed
                if section_key in ['key_insights', 'recommendations', 'follow_up_questions']:
                    if '•' in content or '*' in content or '-' in content:
                        lines = content.split('\n')
                        bullet_points = []
                        for line in lines:
                            line = line.strip()
                            if line and (line.startswith('•') or line.startswith('*') or line.startswith('-')):
                                bullet_points.append(line[1:].strip())
                            elif line and not line.startswith('#'):
                                bullet_points.append(line)
                        if bullet_points:
                            content = bullet_points
                
                analysis_dict[section_key] = content
        
        # Parse summary stats specially
        if 'summary_stats' not in analysis_dict:
            analysis_dict['summary_stats'] = {
                'confidence_score': 'High',
                'key_takeaway': 'Analysis completed successfully with valuable insights.'
            }
        
        return analysis_dict
        
    except Exception as e:
        logger.error(f"Error parsing AI analysis response: {str(e)}")
        return {
            'executive_summary': 'Analysis completed successfully.',
            'detailed_breakdown': 'Detailed breakdown available in results.',
            'key_insights': ['Analysis provides valuable insights into your data.'],
            'recommendations': ['Review the data patterns and consider follow-up analysis.'],
            'follow_up_questions': ['What specific trends would you like to explore further?'],
            'summary_stats': {'confidence_score': 'High', 'key_takeaway': 'Data analysis completed.'}
        }


def format_ai_text_to_html(text):
    """Convert AI-generated text to properly formatted HTML"""
    if not text:
        return ""
    
    # Convert markdown-style formatting to HTML
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'^\* ', r'• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\. ', lambda m: f'<strong>{m.group().strip()}</strong> ', text, flags=re.MULTILINE)
    
    # Convert bullet points to proper list
    if '•' in text:
        lines = text.split('\n')
        formatted_lines = []
        in_list = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('•'):
                if not in_list:
                    formatted_lines.append('<ul style="margin: 10px 0; padding-left: 20px;">')
                    in_list = True
                formatted_lines.append(f'<li style="margin-bottom: 8px;">{line[1:].strip()}</li>')
            else:
                if in_list:
                    formatted_lines.append('</ul>')
                    in_list = False
                if line:
                    formatted_lines.append(f'<p style="margin-bottom: 12px;">{line}</p>')
        
        if in_list:
            formatted_lines.append('</ul>')
        
        text = '\n'.join(formatted_lines)
    else:
        # Convert paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        text = '\n'.join(f'<p style="margin-bottom: 12px;">{p}</p>' for p in paragraphs)
    
    return text


def format_follow_up_questions(text):
    """Format follow-up questions into a nice list"""
    if not text:
        return ""
    
    # Extract questions (lines that end with ?)
    lines = text.split('\n')
    questions = []
    
    for line in lines:
        line = line.strip()
        if line and (line.endswith('?') or any(line.startswith(prefix) for prefix in ['- ', '* ', '1.', '2.', '3.', '4.', '5.'])):
            # Clean up the line
            line = re.sub(r'^[-*\d.]\s*', '', line).strip()
            if line and not line.endswith('?'):
                line += '?'
            questions.append(line)
    
    if not questions:
        # Fallback questions
        questions = [
            "Can you show me trends over time for this data?",
            "How does this compare to previous periods?", 
            "What are the top performers in this category?",
            "Are there any unusual patterns I should investigate?"
        ]
    
    # Format as HTML list
    question_html = '<ul style="margin: 10px 0; padding-left: 20px;">'
    for question in questions[:5]:  # Limit to 5 questions
        question_html += f'<li style="margin-bottom: 10px; cursor: pointer; padding: 8px; background: rgba(33, 150, 243, 0.05); border-radius: 6px; transition: all 0.2s;" onmouseover="this.style.background=\'rgba(33, 150, 243, 0.1)\'" onmouseout="this.style.background=\'rgba(33, 150, 243, 0.05)\'">{question}</li>'
    question_html += '</ul>'
    
    return question_html

def determine_response_format(query, results):
    """Determine the best response format"""
    query_lower = query.lower()
    
    # Explicit format requests
    if any(term in query_lower for term in ['table', 'tabular', 'list']):
        return 'table'
    if any(term in query_lower for term in ['chart', 'graph', 'plot', 'visualize', 'visualization']):
        return 'chart'
    
    # Implicit format detection
    if len(results) > 50:
        return 'table'  # Too many records for effective visualization
    
    if len(results) <= 20 and len(results) > 1:
        return 'chart'  # Good size for visualization
    
    return 'both'  # Show both chart and table

def preprocess_user_query(query):
    if not query or not query.strip():
        return query.strip() if query else ""
    
    query = re.sub(r'\s+', ' ', query)
    query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
    query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
    return query

async def sql_agent(query, user_id=None):
    """Process natural language query using a unified Agno SQL Agent for validation and SQL generation"""
    start_time = time.time()
    logger.info(f"Processing query for user {user_id}: {query}")

    try:
        if not query or not query.strip():
            error_response = {
                "success": False,
                "error": "Empty query provided",
                "data": [
                    {
                        "type": "text",
                        "title": "Error",
                        "content": {"html": "<p>Please provide a valid query.</p>"}
                    }
                ]
            }
            if user_id:
                knowledge_base.store_conversation(
                    user_id=user_id,
                    query=query,
                    success=False,
                    metadata={"error": "Empty query"},
                    response_data=error_response
                )
            return error_response

        # Handle memory clearing
        if "clear my memories" in query.lower() or "remove all memories" in query.lower():
            if user_id:        
                memory.clear()
                success_response = {
                    "success": True,
                    "query": query,
                    "data": [
                        {
                            "type": "text",
                            "title": "Memory Cleared",
                            "content": {"html": "<p>All your memories have been cleared.</p>"}
                        }
                    ]
                }
                knowledge_base.store_conversation(
                    user_id=user_id,
                    query=query,
                    success=True,
                    metadata={"action": "clear_memories"},
                    response_data=success_response
                )
                return success_response
            else:
                error_response = {
                    "success": False,
                    "error": "User ID required to clear memories",
                    "data": [
                        {
                            "type": "text",
                            "title": "Error",
                            "content": {"html": "<p>Please provide a user ID to clear memories.</p>"}
                        }
                    ]
                }
                return error_response

        # Get schema
        schema_info = get_table_schema()
        if not schema_info:
            error_msg = "Failed to retrieve database schema"
            error_response = {
                "success": False,
                "error": error_msg,
                "data": [
                    {
                        "type": "text",
                        "title": "Error",
                        "content": {"html": "<p>Database schema retrieval failed. Check database connection.</p>"}
                    }
                ]
            }
            if user_id:
                knowledge_base.store_conversation(
                    user_id=user_id,
                    query=query,
                    success=False,
                    metadata={"error": error_msg},
                    response_data=error_response
                )
            return error_response

        # Get user preferences
        
        # Build schema prompt
        schema_prompt = "# Medical Practice Database Schema\n\n"
        for table_name, table_info in schema_info.items():
            schema_prompt += f"## Table: {table_name} ({table_info['row_count']:,} rows)\n"
            schema_prompt += "### Columns:\n"
            for col in table_info["columns"]:
                pk_marker = " (PK)" if col["primary_key"] else ""
                nullable = " NULL" if col["nullable"] else " NOT NULL"
                schema_prompt += f"- {col['name']}: {col['type']}{pk_marker}{nullable}\n"
            if table_info["foreign_keys"]:
                schema_prompt += "### Relationships:\n"
                for fk in table_info["foreign_keys"]:
                    schema_prompt += f"- {fk['from']} → {fk['to_table']}.{fk['to_column']}\n"
            schema_prompt += "\n"
            
        # Initialize PostgresTools
        postgres_tools = PostgresTools(
            host=DB_HOST,
            port=int(DB_PORT),
            db_name=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )


        # Enhanced instructions for unified validation and SQL generation
        enhanced_instructions = dedent("""\
            You are an expert SQL generator and validator for a medical practice database. Your role is to:
            1. Understand the user's query intent, understand semantic meaning and context of input
            2. Map it to the provided schema
            3. Generate a PostgreSQL SELECT query
            
            RESPONSE FORMAT REQUIREMENTS:
            Your response must contain these sections in markdown format:

            ### Explanation
            Provide only the relevant steps used to generate this query in clear, numbered points:
            [Only include steps that were actually needed for designing from following general template that sql_query dont explain steps not needed for design]

            Provide a clear, step-by-step explanation in natural language of how the query works:
            1. First, identify what key information the query needs to find.
            (Use this step whenever you need to clarify the main goal.)

            2. Next, map this information to the correct tables and columns in the database.
            (Do this when you want to show where the data comes from.)

            3. Join the necessary tables together using the right conditions or keys.
            (Include joins only if you need to combine data from multiple tables.)

            4. Apply filters to include only the rows that meet certain criteria.
            (Use filters when you want to narrow down the results.)

            5. Group and sort the results in a way that makes the output meaningful.
            (Use grouping or sorting when you need organized or summarized data.)

            6. Finally, select and display the specific columns you want in the result.
            (Do this step every time, since your query needs an output.)
            

            ### SQL Query
            ```sql
            [Generated SQL Query]
            ```

            MANDATORY VALIDATION PROTOCOL:
            1. **Understand Query Intent and Map to Schema**:
            - Analyze the query to identify the user's intent by extracting key terms (e.g., 'income', 'earnings', 'salary', 'procedure_date').
            - Map these terms to relevant schema columns using semantic understanding:
                - Financial terms → revenue/income/cost columns
                - Time terms → date/time/duration columns
                - Doctor terms → surgeon_master table
            - If no direct match, select the most relevant column(s) based on context move from table to column to entries
            - If the query cannot be mapped, return:
                ```
                INFORMATION_NOT_AVAILABLE: for this query '[query]' information is not present in the database schema. Available information includes: [list relevant table.column pairs]
                ```

            2. **Check for Specific Doctor Names**:
            - Detect specific doctor names using patterns: 'dr. <name>', 'doctor <name>', etc.
            - For each detected name, verify its existence in surgeon_master
            - If no records found, return:
                ```
                INFORMATION_NOT_AVAILABLE: Specific doctor name(s) mentioned: [names]. No matching doctors found.
                ```

            SCHEMA:
            The schema is provided in the prompt. Use only the listed tables and columns.
        """)

        table_prompt = dedent("""\
            1. PATIENT & VISIT MANAGEMENT
            big_sky_admission_billing_schedule
            Purpose: Tracks patient appointments and billing status
            patient_id (PK): Unique patient identifier
            dos (PK): Date of service - the actual appointment/surgery date
            appointment_time: Scheduled time for the appointment
            patient_name: Patient's full name
            physician: Attending physician name
            procedure: Medical procedure to be performed
            auth_dos: Authorization days of service approved by insurance
            primary_payer: Primary insurance company/payer
            verification_status: Boolean - whether insurance verification is complete
            appointment_status: Current status (scheduled, completed, cancelled, etc.)
            amount_due: Total amount patient owes
            balance_due: Outstanding balance after payments
            is_total: Boolean flag for summary rows

            big_sky_visit_billing_data
            Purpose: Comprehensive billing and clinical data per patient visit

            acct_num (PK): Account number for billing
            ptid_vst_num (PK): Patient ID + Visit number combination
            patient_name: Patient's full name
            dos: Date of service
            date_billed: When the claim was submitted
            date_paid: When payment was received
            phys_id: Physician unique identifier
            phys_name: Physician name
            cpt_code1-10: Up to 10 CPT procedure codes billed
            desc1-10: Descriptions for each CPT code
            cpt1-10_modifiers: Medical coding modifiers
            pre_op_min: Pre-operative time in minutes
            or_min: Operating room time in minutes
            srgy_min: Actual surgery time in minutes
            rr_min: Recovery room time in minutes
            supply_cost: Cost of medical supplies used
            staff_cost: Labor cost for staff time
            billed_amt: Total amount billed to insurance
            payments: Total payments received
            balance_due: Outstanding balance

            2. SURGICAL OPERATIONS & SCHEDULING
            big_sky_procedure_data_with_turnover
            Purpose: Detailed surgical case timing and turnover metrics

            patient_id (PK): Patient identifier
            visit_number (PK): Visit sequence number
            or_room: Operating room identifier
            date_of_service: Surgery date
            scheduled_start_time: Planned surgery start time
            anesthesia_start_time: When anesthesia began
            anesthesia_end_time: When anesthesia ended
            surgery_start_time: Actual surgery start time
            surgery_end_time: Actual surgery end time
            or_start_time: OR room occupation start
            or_end_time: OR room occupation end
            or_turnover_minutes: Time between cases for room preparation
            procedure_description: Surgical procedure performed
            surgeon_name: Primary surgeon
            anesthesia_type: Type of anesthesia used
            anesthesiologist_name: Anesthesia provider

            big_sky_surgery_time_log
            Purpose: Detailed timestamps for each phase of surgical care

            pt_id_visit (PK): Patient + visit identifier
            dos: Date of service
            sched_time: Scheduled appointment time
            pt_arr: Patient arrival time
            phys_arr: Physician arrival time
            registr_start/min: Registration phase timing
            preop_start/min: Pre-operative preparation timing
            anes_start/min: Anesthesia administration timing
            or_start/min: Operating room time
            surgery_start/min: Actual surgical procedure timing
            pacu_phase1_start/min: Post-anesthesia care unit phase 1
            postop_phase2_start/min: Post-operative phase 2 recovery
            extended_stay_start/min: Extended recovery if needed
            antibiotic_start/min: Antibiotic prophylaxis timing

            big_sky_surgical_clinical_data
            Purpose: Comprehensive clinical and operational data per case (currently empty but structured)

            Contains detailed clinical metrics, staff assignments, timing data
            Multiple staff positions (staff1-10) with roles and time allocation
            Temperature, vital signs, and clinical indicators
            Equipment and scope usage tracking

            3. FINANCIAL MANAGEMENT
            big_sky_financial_class_summary
            Purpose: Payment performance metrics by insurance type

            financial_class (PK): Insurance category (Medicare, Commercial, Self-pay, etc.)
            total_cases: Number of cases in this category
            total_billed: Total charges submitted
            balance_due: Outstanding receivables
            primary_payments: Payments from primary insurance
            secondary_payments: Payments from secondary insurance
            self_pay_payments: Patient payments
            contract_writeoffs: Contractual adjustments
            bad_debt_writeoffs: Uncollectable amounts
            percent_primary_pay_billed: Primary payer reimbursement rate
            avg_payment_days_primary: Days to receive primary payment
            avg_billed_per_case: Average charges per case
            avg_payment_per_case: Average payment per case

            big_sky_contractual_revenue_variance
            Purpose: Analysis of expected vs actual contract performance

            account_number (PK): Patient account
            visit_number (PK): Visit identifier
            payer_id: Insurance company identifier
            contract_id: Specific contract terms
            billed_amount: Amount charged
            contract_estimated: Expected reimbursement per contract
            contract_actual: Actual payment received
            contract_variance: Difference between expected and actual
            revenue_estimated: Projected revenue
            revenue_actual: Actual revenue received
            revenue_variance: Revenue performance variance

            big_sky_payment_trending
            Purpose: Payment timing analysis by aging buckets

            claim_number (PK): Insurance claim identifier
            payer_id: Insurance company
            gross_billing: Total charges
            expected_revenue: Projected collections
            payment_0_30_days: Payments received within 30 days
            payment_31_60_days: Payments received 31-60 days
            payment_61_90_days: Payments received 61-90 days
            payment_91_120_days: Payments received 91-120 days
            payment_121_plus_days: Payments received after 120 days
            Each aging bucket has corresponding percentage fields

            4. COST ANALYSIS & PROFITABILITY
            big_sky_procedure_profit_cost
            Purpose: Profitability analysis by procedure and surgeon

            procedure_name (PK): Surgical procedure type
            surgeon_id (PK): Surgeon identifier
            case_count: Number of cases performed
            gross_billing: Total charges for these cases
            expected_revenue: Projected collections
            actual_payment: Actual payments received
            direct_cost: Direct costs (supplies, implants)
            indirect_cost: Indirect costs (overhead, admin)
            total_cost: Combined direct and indirect costs
            net_profit: Revenue minus total costs
            profit_margin_expected: Expected profit percentage
            profit_margin_actual: Actual profit percentage

            big_sky_procedure_summary
            Purpose: Case-level cost and timing summary

            pt_id_visit (PK): Patient + visit identifier
            procedure: Surgical procedure performed
            physician: Performing surgeon
            primary_payer: Insurance company
            srgy_min: Surgery duration in minutes
            staff_cost: Labor costs for the case
            center_supply_cost: Medical supply costs
            billed_amount: Total charges

            5. PHYSICIAN & STAFF MANAGEMENT
            big_sky_physician_case_billings
            Purpose: Monthly billing performance by physician

            physician_name (PK): Doctor's name
            specialty: Medical specialty
            jan_25_cases through dec_25_cases: Monthly case volumes
            jan_25_billings through dec_25_billings: Monthly billing amounts
            total_cases: Annual case volume
            total_billings: Annual billing total

            big_sky_surgeon_master
            Purpose: Physician directory and credentials

            physician_id (PK): Unique doctor identifier
            physician_name: Doctor's full name
            specialty: Medical specialty
            npi: National Provider Identifier
            state_license: Medical license number
            status: Active/inactive status

            big_sky_staff_master
            Purpose: Employee directory

            employee_id (PK): Unique staff identifier
            employee_name: Staff member's name
            hire_date: Employment start date
            department: Work department
            employee_type: Job classification

            big_sky_staff_utilization
            Purpose: Staff time tracking per surgical case

            staff_name: Employee name
            role: Job function during case
            scheduled_duration: Planned time allocation
            or_duration: Actual OR time
            time_on_patient: Direct patient care time

            big_sky_employee_credentials
            Purpose: Professional licensing and certification tracking

            employee_id: Staff identifier
            credential_type: Type of license/certification
            expiration_date: When credential expires
            credential_status: Current status (active, expired, pending)
            license_number: Official credential number

            6. CLINICAL QUALITY & ANESTHESIA
            big_sky_anesthesia_statistics
            Purpose: Anesthesia performance and safety metrics

            anesthesia_type (PK): Type of anesthesia (General, Regional, Local, etc.)
            number_of_cases_performed: Case volume
            number_of_complications: Adverse events
            total_anesthesia_time_minutes: Aggregate anesthesia time

            7. SUPPLY CHAIN & INVENTORY
            big_sky_item_information
            Purpose: Medical supply and equipment catalog

            item_id (PK): Inventory item identifier
            item_description: Product name/description
            mfg_cat_no: Manufacturer catalog number
            hcpcs_code: Healthcare billing code
            stock_level: Current inventory quantity
            cost_price: Purchase cost
            markup_price: Internal pricing with markup
            current_price: Current selling price

            big_sky_preference_card_list
            Purpose: Surgeon-specific supply preferences

            phys_id (PK): Physician identifier
            pref_card_description (PK): Preference card name
            procedure_description: Associated procedure
            cost: Total cost of preference card supplies

            8. REFERENCE DATA
            big_sky_procedures
            Purpose: Master procedure catalog

            procedure (PK): Procedure code/name
            description: Detailed procedure description
            primary_cpt: Main billing code
            specialty: Medical specialty
            duration: Expected procedure time
            require_laser: Whether laser equipment needed

            big_sky_cpt_codes
            Purpose: Medical billing code reference

            cpt_code (PK): Current Procedural Terminology code
            description: Procedure description
            specialty: Associated medical specialty
            medicare_fee: Medicare reimbursement rate
            revenue_code: Hospital revenue code category

            big_sky_payer_list
            Purpose: Insurance company directory

            payer_id (PK): Insurance company identifier
            payer_name: Insurance company name
            financial_class: Category (Commercial, Medicare, Medicaid, etc.)
            payer_type: Insurance type classification
            claim_format: Electronic format for claims
            status: Active/inactive status
        """) 
        Formula_prompt = dedent("""\
            First Case On-Time Start Accuracy = (On-time first cases ÷ Total first cases) * 100
            Block Utilization (Method 1) = (Total Case Time ÷ Total Allocated Block Time) * 100
            Block Utilization (Method 2) = ((Last Wheels Out - First Wheels In) ÷ Total Allocated Block Time) *100
            OR Utilization (People Count) = (Minutes with People Present ÷ Total Available Minutes) * 100
            OR Utilization (Case Duration) = (Σ(Wheels Out - Wheels In) ÷ Total Available OR Time) * 100
            OR Utilization (With Turnover) = ((Total Time - Time with Zero People Count) ÷ Total Available Time) * 100
            TAT (>1 hour apart) = T(room_empty) - T(wheels_out)
            TAT (≤1 hour apart) = T(next_wheels_in) - T(previous_wheels_out)
            Procedure Length = Cut Time - Close Time
            Scheduling Duration Accuracy = (Actual - Scheduled) ÷ Scheduled * 100
            Block Time Utilization = (Total Actual Surgery Time ÷ Allocated Block Time) * 100
            Anesthesia Booking Efficiency = (Total Anesthesia Hours Used ÷ Contracted Anesthesia Hours) * 100
            Total Case Cost = Direct Labor + OR Time Cost + Supply & Implant + Medication + Equipment/Depreciation + Allocated Overhead + Indirect Labor
            Profit Margin % = (Revenue - Total Case Cost) ÷ Revenue * 100
            Contribution Margin = Revenue - Variable Costs
            Contribution Margin % = (Revenue - Variable Costs) ÷ Revenue * 100
            FTEs per Case = Count of staff (by role) present during case duration
            On-Time Start Flag = IF(actual_start_time ≤ scheduled_start_time, "On-Time", "Delayed")
            Turnover Time = Next Case WHEELS_IN - Current Case WHEELS_OUT
            Contribution Margin per Surgeon = SUM(Contribution Margin per Case where surgeon = [Surgeon Name])
            Revenue per CPT = SUM(Revenue for cases with specific CPT code)
            Cost per CPT = SUM(Total Case Cost for cases with specific CPT code)
            Margin per CPT = Revenue per CPT - Cost per CPT
            Margin % per CPT = (Margin per CPT ÷ Revenue per CPT) * 100
            Supply Cost per Case = SUM(Cost of supplies/implants used in case)
            Variance from Preference Card = Actual Supply Used - Preference Card Supply
            Margin Impact = Contribution Margin with Actual Supplies - Contribution Margin with Standard Supplies
        """)

        # Get or create session for user
        session_id = active_sessions.get(user_id, str(uuid4()))
        if user_id and user_id not in active_sessions:
            active_sessions[user_id] = session_id
        postgres_tools = PostgresTools(
            host=DB_HOST,
            port=int(DB_PORT),
            db_name=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        # Initialize the unified agent
        agent = Agent(
            name="Medical SQL Validator and Generator",
            model=ClaudeSonnetModel(),
            tools=[postgres_tools],
            memory=memory,
            storage=session_storage,
            enable_agentic_memory=True,
            enable_user_memories=True,
            description="You are an expert at generating and validating SQL code for PostgreSQL database",
            instructions=enhanced_instructions
        )

        # Add context from similar queries
        context_info = ""
        if user_id:
            similar_queries = knowledge_base.get_similar_queries(user_id, query, 3)
            if similar_queries:
                context_info = "\n# Previous Similar Queries Context:\n"
                for sq in similar_queries:
                    context_info += f"- Query: {sq[0]}\n  SQL: {sq[1]}\n"

        processed_query = preprocess_user_query(query)
        combined_query = f"{schema_prompt}\n{table_prompt}\n{Formula_prompt}\n{context_info}\n\nUser Query: {processed_query}\n"

        # Run the agent
        result = agent.run(combined_query, timeout=60, user_id=user_id, session_id=session_id)

        # Check if the agent indicates information is not available
        if "INFORMATION_NOT_AVAILABLE:" in result.content:
            info_not_available_msg = result.content.split("INFORMATION_NOT_AVAILABLE:")[1].strip()
            if user_id:
                knowledge_base.store_conversation(user_id, query, success=True, metadata={"info_not_available": True, "agent_response": info_not_available_msg})
            
            return {
                "success": True,
                "query": query,
                "info_not_available": True,
                "data": [
                    {
                        "type": "text",
                        "title": "Information Not Available",
                        "content": {"html": f"<div style='background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;'><h4 style='color: #856404; margin-top: 0;'>📋 Database Information</h4><p>{info_not_available_msg}</p></div>"}
                    }
                ]
            }

        # Extract explanation and SQL query from response
        explanation = ""
        sql_query = ""

        # Extract explanation section
        explanation_match = re.search(r'### Explanation\n(.*?)(?=\n###)', result.content, re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()

        # Extract SQL query
        sql_query_match = re.search(r'```sql\n(.*?)\n```', result.content, re.DOTALL)
        if sql_query_match:
            sql_query = sql_query_match.group(1).strip()

        if not sql_query and result.content.strip():
            if user_id:
                knowledge_base.store_conversation(user_id, query, success=True, metadata={"no_sql_generated": True, "agent_response": result.content})
            return {
                "success": True,
                "query": query,
                "no_sql_generated": True,
                "data": [
                    {
                        "type": "text",
                        "title": "Agent Response",
                        "content": {"html": f"<div style='background: #e7f3ff; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;'><h4 style='color: #004085; margin-top: 0;'>🤖 Agent Response</h4><p>{result.content}</p></div>"}
                    }
                ]
            }
            if user_id:
                knowledge_base.store_conversation(
                    user_id=user_id,
                    query=query,
                    success=True,
                    metadata={
                        "no_sql_generated": True,
                        "agent_response": result.content
                    },
                    response_data=no_sql_response
                )
            return no_sql_response


        # Validate SQL query safety
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE', 'GRANT', 'REVOKE']
        if any(keyword in sql_query.upper() for keyword in dangerous_keywords):
            raise ValueError("Generated query contains potentially dangerous operations")

        # Execute SQL query
        results = execute_sql(sql_query)
        execution_time = time.time() - start_time

        # Prepare the complete response data
        response_data = {
            "success": True,
            "query": query,
            "sql": sql_query,
            "execution_time": execution_time,
            "result_count": len(results),
            "data": format_response_as_objects(query, sql_query, results, execution_time, explanation)
        }

        # Store conversation with full response data
        if user_id:
            response_format = determine_response_format(query, results)
            knowledge_base.store_conversation(
                user_id=user_id,
                query=query,
                sql_query=sql_query,
                result_count=len(results),
                success=True,
                execution_time=execution_time,
                metadata={
                    "explanation": explanation,
                    "query_type": response_format
                },
                response_data=response_data
            )

        logger.info(f"Query processed in {execution_time:.2f}s - {len(results)} rows")
        return response_data

    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"Query processing failed after {error_time:.2f}s: {str(e)}")
        
        error_response = {
            "success": False,
            "error": str(e),
            "execution_time": error_time,
            "data": [
                {
                    "type": "text",
                    "title": "Error",
                    "content": {"html": f"<p><strong>Error:</strong> {str(e)}<br><strong>Query:</strong> {query}</p>"}
                }
            ]
        }
        
        if user_id:
            knowledge_base.store_conversation(
                user_id=user_id,
                query=query,
                success=False,
                metadata={"error": str(e)},
                response_data=error_response
            )
        
        return error_response

def log_user_conversations(user_id):
    if not user_id or not memory:
        logger.warning("No user_id provided or memory not initialized")
        return
    
    memories = memory.get_user_memories(user_id=user_id)
    if not memories:
        logger.info(f"No conversations found for user {user_id}")
        return
    
    logger.info(f"--- Conversations for user {user_id} ---")
    for i, mem in enumerate(memories, 1):
        logger.info(f"Conversation {i}:")
        logger.info(f"  Memory: {mem.memory}")
        logger.info(f"  Topics: {mem.topics}")
        logger.info("-" * 40)
    logger.info(f"Total conversations logged for user {user_id}: {len(memories)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory_db, memory, connection_pool, knowledge_base, session_storage
    try:
        # Initialize PostgreSQL memory database
        memory_db = PostgresMemoryDb(
            table_name="user_memories",
            db_url=DB_URL
        )
        memory = Memory(db=memory_db)
        
        knowledge_base = KnowledgeBase(db_url=DB_URL)
        session_storage = PostgresStorage(
            table_name="agent_sessions",
            db_url=DB_URL
        )
        get_connection_pool()
        logger.info("Application startup completed")
        yield
    finally:
        if LOG_USER_ID:
            log_user_conversations(LOG_USER_ID)
        if connection_pool:
            connection_pool.closeall()
            logger.info("Connection pool closed")
        logger.info("Application shutdown completed")

app = FastAPI(
    title="Medical SQL Assistant",
    description="Natural Language to SQL Query System for Medical Practice Data",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/agent/api")
async def read_agent_api():
    return {
        "message": "Agent API is working",
        "endpoints": {
            "query": "/agent/api/query (POST)",
            "memories": "/agent/api/memories/{user_id} (GET)",
            "chat_history": "/agent/api/chat-history/{user_id} (GET)",
            "user_memories": "/agent/api/user-memories/{user_id} (GET)",
            "clear_session": "/agent/api/clear-session/{user_id} (DELETE)"
        },
        "status": "healthy"
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory=os.path.dirname(os.path.abspath(__file__))), name="static")

sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True
)
socket_app = socketio.ASGIApp(sio, app)

@sio.event
async def connect(sid, environ):
    logger.info(f"Socket connected: {sid}")
    await sio.emit('connection_success', {'message': 'Connected to SQL Agent'}, to=sid)

@sio.event
async def disconnect(sid):
    logger.info(f"Socket disconnected: {sid}")

@sio.event
async def query(sid, data):
    try:
        query = data.get('query')
        user_id = data.get('user_id')
        if not query:
            await sio.emit('query_error', {'error': 'Query is required'}, to=sid)
            return
        result = await sql_agent(query, user_id)
        await sio.emit('query_result', result, to=sid)
    except Exception as e:
        logger.error(f"Socket query error: {str(e)}")
        await sio.emit('query_error', {'error': str(e)}, to=sid)
        
def create_chart_html(item, unique_id, conversation_id):
    """Create robust chart HTML with enhanced error handling"""
    chart_content = item.get('content', {})
    chart_title = item.get('title', 'Data Visualization')
    
    # Validate chart configuration
    if not chart_content or 'chart_type' not in chart_content:
        return f"""
        <div style='text-align: center; color: #666; padding: 40px; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;'>
            <h4 style='margin: 0; color: #6c757d;'> Chart data unavailable</h4>
            <p style='margin: 10px 0 0 0; font-size: 14px;'>The chart configuration could not be loaded.</p>
        </div>
        """
    
    # Build comprehensive chart configuration
    chart_config = {
        'type': chart_content.get('chart_type', 'bar'),
        'data': chart_content.get('data', {'labels': [], 'datasets': []}),
        'options': {
            'responsive': True,
            'maintainAspectRatio': False,
            'plugins': {
                'legend': {
                    'display': True,
                    'position': 'top'
                },
                'title': {
                    'display': True,
                    'text': chart_title,
                    'font': {'size': 16, 'weight': 'bold'}
                },
                'tooltip': {
                    'enabled': True,
                    'mode': 'index',
                    'intersect': False
                }
            },
            'interaction': {
                'mode': 'nearest',
                'axis': 'x',
                'intersect': False
            }
        }
    }
    
    # Merge existing options
    if 'options' in chart_content:
        merge_options(chart_config['options'], chart_content['options'])
    
    # Add scales for appropriate chart types
    if chart_config['type'] in ['bar', 'line', 'area']:
        if 'scales' not in chart_config['options']:
            chart_config['options']['scales'] = {
                'x': {
                    'beginAtZero': True,
                    'grid': {
                        'display': True,
                        'color': 'rgba(0, 0, 0, 0.1)'
                    }
                },
                'y': {
                    'beginAtZero': True,
                    'grid': {
                        'display': True,
                        'color': 'rgba(0, 0, 0, 0.1)'
                    }
                }
            }
    
    # Convert chart config to JSON string with error handling
    try:
        chart_config_json = json.dumps(chart_config, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.error(f"Chart config serialization error: {e}")
        return f"""
        <div style='text-align: center; color: #dc3545; padding: 40px; background: #f8d7da; border-radius: 8px; border: 1px solid #f5c6cb;'>
            <h4 style='margin: 0;'> Chart Configuration Error</h4>
            <p style='margin: 10px 0 0 0; font-size: 14px;'>Unable to serialize chart configuration.</p>
        </div>
        """
    
    # Generate the complete chart HTML
    chart_html = f"""
    <div class='chart-container' style='position: relative; width: 100%; margin: 20px 0; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); border: 1px solid #e9ecef;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #f8f9fa;'>
            <h4 style='margin: 0; color: #495057; font-size: 18px; font-weight: 600;'>
                 {chart_title}
            </h4>
            <div style='display: flex; gap: 10px; align-items: center;'>
                <span style='font-size: 12px; color: #6c757d; background: #f8f9fa; padding: 4px 8px; border-radius: 4px;'>
                    ID: {conversation_id}
                </span>
            </div>
        </div>
        
        <div style='position: relative; height: 400px; width: 100%;'>
            <canvas id='{unique_id}' style='max-height: 400px; width: 100%;'></canvas>
            <div id='{unique_id}_loading' style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: #6c757d;'>
                <div style='font-size: 14px;'>Loading chart...</div>
                <div style='margin-top: 10px; font-size: 24px;'></div>
            </div>
            <div id='{unique_id}_error' style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: #dc3545; display: none;'>
                <div style='font-size: 14px; font-weight: 500;'> Chart Load Error</div>
                <div style='margin-top: 5px; font-size: 12px;'>Please refresh the page to retry</div>
            </div>
        </div>
        
        <script>
            (function() {{
                const canvasId = '{unique_id}';
                const loadingId = '{unique_id}_loading';
                const errorId = '{unique_id}_error';
                let retryCount = 0;
                const maxRetries = 5;
                
                function hideLoading() {{
                    const loading = document.getElementById(loadingId);
                    if (loading) loading.style.display = 'none';
                }}
                
                function showError() {{
                    hideLoading();
                    const error = document.getElementById(errorId);
                    if (error) error.style.display = 'block';
                }}
                
                function initChart() {{
                    const canvas = document.getElementById(canvasId);
                    if (!canvas) {{
                        console.warn('Canvas not found:', canvasId);
                        return;
                    }}
                    
                    if (typeof Chart === 'undefined') {{
                        if (retryCount < maxRetries) {{
                            retryCount++;
                            console.log(`Chart.js not loaded yet, retry ${{retryCount}}/${{maxRetries}}`);
                            setTimeout(initChart, 200 * retryCount);
                            return;
                        }} else {{
                            console.error('Chart.js failed to load after', maxRetries, 'retries');
                            showError();
                            return;
                        }}
                    }}
                    
                    try {{
                        const ctx = canvas.getContext('2d');
                        const config = {chart_config_json};
                        
                        // Validate configuration
                        if (!config.data || !config.data.labels || !config.data.datasets) {{
                            throw new Error('Invalid chart configuration: missing data structure');
                        }}
                        
                        // Create the chart
                        const chart = new Chart(ctx, config);
                        
                        // Hide loading indicator on success
                        hideLoading();
                        
                        console.log('Chart initialized successfully:', canvasId);
                        
                    }} catch (error) {{
                        console.error('Chart creation error for', canvasId, ':', error);
                        showError();
                    }}
                }}
                
                // Multiple initialization strategies
                if (document.readyState === 'complete') {{
                    initChart();
                }} else {{
                    document.addEventListener('DOMContentLoaded', initChart);
                    // Fallback for cases where DOMContentLoaded already fired
                    setTimeout(initChart, 100);
                }}
                
                // Additional fallback for dynamic content loading
                setTimeout(initChart, 500);
            }})();
        </script>
    </div>
    """
    
    return chart_html


def merge_options(target, source):
    """Recursively merge chart options"""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            merge_options(target[key], value)
        else:
            target[key] = value


def create_fallback_content(conv):
    """Create fallback content for conversation display with enhanced styling"""
    query = conv.get('query', '')
    sql_query = conv.get('sql_query', '')
    result_count = conv.get('result_count', 0)
    
    html = f'''
    <div class="conversation-item" style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #6c757d; margin: 10px 0;">
        <h4 style="color: #495057; margin-top: 0; font-size: 16px;">Query Response</h4>
        <p style="margin: 10px 0;"><strong>Query:</strong> {query}</p>
    '''
    
    if sql_query:
        html += f'<p style="margin: 10px 0;"><strong>SQL:</strong> <code style="background: #e9ecef; padding: 2px 4px; border-radius: 3px;">{sql_query}</code></p>'
    
    html += f'<p style="margin: 10px 0 0 0;"><strong>Results:</strong> {result_count} rows</p>'
    html += '</div>'
    
    return html

@sio.event
async def get_chat_history(sid, data):
    """Enhanced socket handler for chat history with robust chart rendering"""
    try:
        user_id = data.get('user_id')
        page = int(data.get('page', 1))
        per_page = int(data.get('per_page', 20))
        
        if not user_id:
            await sio.emit('chat_history_error', {'error': 'User ID is required'}, to=sid)
            return
        
        # Get total count
        with knowledge_base.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM user_conversations WHERE user_id = %s", (user_id,))
                total_count = cursor.fetchone()[0]
        
        # Calculate pagination
        total_pages = (total_count + per_page - 1) // per_page
        offset = (page - 1) * per_page
        
        # Get conversations
        kb_conversations = knowledge_base.get_user_conversations(user_id, limit=per_page, offset=offset)
        
        chat_history = []
        chart_count = 0
        
        for conv in kb_conversations:
            # Add user message
            chat_history.append({
                "role": "user",
                "content": conv['query'],
                "timestamp": conv['timestamp'],
                "source": "knowledge_base",
                "conversation_id": conv['id']
            })
            
            # Process response data with enhanced chart handling
            response_data = conv.get('response_data', {})
            
            # Ensure proper structure
            if isinstance(response_data, str):
                try:
                    response_data = json.loads(response_data)
                except (json.JSONDecodeError, TypeError):
                    response_data = {'data': []}
            
            if not isinstance(response_data, dict) or 'data' not in response_data:
                response_data = {'data': []}
            
            # Process each data item
            html_parts = []
            
            for item_index, item in enumerate(response_data.get('data', [])):
                if item.get('type') == 'text' and 'content' in item and 'html' in item['content']:
                    html_parts.append(item['content']['html'])
                
                elif item.get('type') == 'table' and 'content' in item:
                    table_content = item['content']
                    if 'headers' in table_content and 'rows' in table_content:
                        table_html = create_table_html(table_content['headers'], table_content['rows'])
                        html_parts.append(table_html)
                
                elif item.get('type') == 'chart' and 'content' in item:
                    chart_count += 1
                    unique_id = f"chat_chart_{conv['id']}_{item_index}_{int(time.time() * 1000)}"
                    chart_html = create_chart_html(item, unique_id, conv['id'])
                    html_parts.append(chart_html)
            
            # Add assistant response
            assistant_content = "\n".join(html_parts) if html_parts else create_fallback_content(conv)
            
            chat_history.append({
                "role": "assistant",
                "content": assistant_content,
                "timestamp": conv['timestamp'],
                "source": "knowledge_base",
                "conversation_id": conv['id'],
                "sql_query": conv.get('sql_query'),
                "result_count": conv.get('result_count'),
                "success": conv.get('success', False),
                "metadata": conv.get('metadata', {}),
                "has_charts": any(item.get('type') == 'chart' for item in response_data.get('data', []))
            })
        
        # Sort by timestamp (newest first)
        chat_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        await sio.emit('chat_history_result', {
            'user_id': user_id,
            'session_id': active_sessions.get(user_id),
            'chat_history': chat_history,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'total_items': total_count
            },
            'message': f'Retrieved {len(chat_history)} messages with {chart_count} charts',
            'chart_count': chart_count
        }, to=sid)
        
    except Exception as e:
        logger.error(f"Socket chat history error: {str(e)}", exc_info=True)
        await sio.emit('chat_history_error', {
            'error': str(e),
            'details': 'Failed to retrieve chat history. Please try again.'
        }, to=sid)



def create_table_html(headers, rows, financial_columns=None):
    """Create styled HTML table with currency formatting"""
    if financial_columns is None:
        financial_columns = set()
    
    table_html = """
    <div style='overflow-x: auto; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
        <table style='width: 100%; border-collapse: collapse; background: white;'>
            <thead>
                <tr style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;'>
    """
    
    for header in headers:
        table_html += f"<th style='padding: 15px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #dee2e6;'>{header}</th>"
    
    table_html += "</tr></thead><tbody>"
    
    for i, row in enumerate(rows):
        bg_color = '#f8f9fa' if i % 2 == 0 else 'white'
        table_html += f"<tr style='background: {bg_color}; transition: background-color 0.2s;' onmouseover='this.style.background=\"#e3f2fd\"' onmouseout='this.style.background=\"{bg_color}\"'>"
        
        for col, cell in zip(headers, row):
            if col in financial_columns:
                formatted_cell = format_financial_value(cell)
                table_html += f"<td style='padding: 12px; border-bottom: 1px solid #dee2e6; color: #495057; text-align: right;'>{formatted_cell}</td>"
            else:
                table_html += f"<td style='padding: 12px; border-bottom: 1px solid #dee2e6; color: #495057;'>{cell}</td>"
        
        table_html += "</tr>"
    
    table_html += "</tbody></table></div>"
    return table_html
        
@sio.event
async def clear_session(sid, data):
    try:
        user_id = data.get('user_id')
        if not user_id:
            await sio.emit('clear_session_error', {'error': 'User ID is required'}, to=sid)
            return
        
        session_id = active_sessions.get(user_id)
        if session_id:
            if session_id in memory.runs:
                del memory.runs[session_id]
            del active_sessions[user_id]
            memory.clear()


        
        await sio.emit('clear_session_result', {
            'success': True,
            'message': f'Session cleared for user {user_id}'
        }, to=sid)
        
    except Exception as e:
        logger.error(f"Socket clear session error: {str(e)}")
        await sio.emit('clear_session_error', {'error': str(e)}, to=sid)

@app.get("/")
async def root():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"))

@app.get("/favicon.ico")
async def favicon():
    favicon_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <rect width="100" height="100" fill="#007bff"/>
        <text x="50" y="65" font-family="Arial, sans-serif" font-size="50" text-anchor="middle" fill="white">🏥</text>
    </svg>"""
    return Response(content=favicon_svg, media_type="image/svg+xml")

@app.get("/robots.txt")
async def robots():
    robots_content = """User-agent: *
Disallow: /

# This application is for internal use only
# No crawling allowed"""
    return Response(content=robots_content, media_type="text/plain")

@app.post("/agent/api/query")
async def query_endpoint(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        user_id = data.get("user_id")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        result = await sql_agent(query, user_id)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/api/memories/{user_id}")
async def get_memories(user_id: str):
    try:
        memories = memory.get_user_memories(user_id=user_id)
        logger.info(f"Retrieved memories for user {user_id}:")
        pprint(memories)
        return JSONResponse(content={
            "success": True,
            "user_id": user_id,
            "memories": [{
                "memory": mem.memory,
                "topics": mem.topics,
                "timestamp": mem.timestamp.isoformat() if isinstance(mem.timestamp, datetime) else mem.timestamp
            } for mem in memories]
        })
    except Exception as e:
        logger.error(f"Memory retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/api/chat-history/{user_id}")
async def get_chat_history_api(
    user_id: str,
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    format: str = Query("html", description="Response format: json or html"),
    include_charts: bool = Query(True, description="Include chart data in response")
):
    """
    Enhanced REST API endpoint to retrieve chat history with proper chart rendering
    """
    try:
        # Validate user_id
        if not user_id or len(user_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="User ID is required and cannot be empty")
        
        # Get total count for pagination
        with knowledge_base.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM user_conversations WHERE user_id = %s", 
                    (user_id,)
                )
                total_count = cursor.fetchone()[0]
        
        # Calculate pagination parameters
        total_pages = (total_count + per_page - 1) // per_page
        offset = (page - 1) * per_page
        
        # Validate page number
        if page > total_pages and total_count > 0:
            raise HTTPException(
                status_code=404, 
                detail=f"Page {page} not found. Total pages: {total_pages}"
            )
        
        # Get conversations from knowledge base
        kb_conversations = knowledge_base.get_user_conversations(
            user_id, 
            limit=per_page, 
            offset=offset
        )
        
        chat_history = []
        chart_count = 0
        
        for conv in kb_conversations:
            # Add user message
            user_message = {
                "role": "user",
                "content": conv['query'],
                "timestamp": conv['timestamp'],
                "source": "knowledge_base",
                "conversation_id": conv['id']
            }
            chat_history.append(user_message)
            
            # Process response data
            response_data = conv.get('response_data', {})
            
            # Ensure proper structure
            if isinstance(response_data, str):
                try:
                    response_data = json.loads(response_data)
                except (json.JSONDecodeError, TypeError):
                    response_data = {'data': []}
            
            if not isinstance(response_data, dict) or 'data' not in response_data:
                response_data = {'data': []}
            
            # Process each data item based on format
            html_parts = []
            chart_data = []
            
            for item_index, item in enumerate(response_data.get('data', [])):
                try:
                    if item.get('type') == 'text' and 'content' in item:
                        if format == "html" and 'html' in item['content']:
                            html_parts.append(item['content']['html'])
                        elif format == "json":
                            html_parts.append(item['content'].get('text', item['content'].get('html', '')))
                    
                    elif item.get('type') == 'table' and 'content' in item:
                        table_content = item['content']
                        if format == "html" and 'headers' in table_content and 'rows' in table_content:
                            table_html = create_table_html(table_content['headers'], table_content['rows'])
                            html_parts.append(table_html)
                        elif format == "json":
                            html_parts.append(f"Table: {len(table_content.get('rows', []))} rows")
                    
                    elif item.get('type') == 'chart' and 'content' in item:
                        chart_count += 1
                        if include_charts and format == "html":
                            # Generate unique ID for API charts
                            unique_id = f"api_chart_{conv['id']}_{item_index}_{int(time.time() * 1000)}_{chart_count}"
                            chart_html = create_chart_html(item, unique_id, conv['id'])
                            html_parts.append(chart_html)
                        elif include_charts and format == "json":
                            # Return complete chart configuration for JSON format
                            chart_data.append({
                                'type': 'chart',
                                'title': item.get('title', 'Chart'),
                                'chart_type': item['content'].get('chart_type'),
                                'data': {
                                    'labels': item['content'].get('data', {}).get('labels', []),
                                    'datasets': [{
                                        'label': ds.get('label', 'Dataset'),
                                        'data': ds.get('data', [])
                                    } for ds in item['content'].get('data', {}).get('datasets', [])]
                                },
                                'options': item['content'].get('options', {})
                            })
                        else:
                            html_parts.append(f"📊 Chart: {item.get('title', 'Visualization')}")
                            
                except Exception as item_error:
                    logger.error(f"Error processing item {item_index} for conversation {conv['id']}: {item_error}")
                    if format == "html":
                        html_parts.append(f"<p style='color: #dc3545;'>Error loading content item {item_index + 1}</p>")
                    else:
                        html_parts.append(f"Error loading content item {item_index + 1}")
            
            # Create assistant response
            if format == "html":
                assistant_content = "\n".join(html_parts) if html_parts else create_fallback_content(conv)
            else:
                assistant_content = {
                    'text': "\n".join(html_parts) if html_parts else f"Query: {conv['query']}",
                    'charts': chart_data if include_charts else []
                }
            
            assistant_message = {
                "role": "assistant",
                "content": assistant_content,
                "timestamp": conv['timestamp'],
                "source": "knowledge_base",
                "conversation_id": conv['id'],
                "sql_query": conv.get('sql_query'),
                "result_count": conv.get('result_count'),
                "success": conv.get('success', False),
                "execution_time": conv.get('execution_time', 0),
                "metadata": conv.get('metadata', {}),
                "has_charts": any(item.get('type') == 'chart' for item in response_data.get('data', []))
            }
            chat_history.append(assistant_message)
        
        # Sort by timestamp (newest first)
        chat_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Prepare response
        response_data = {
            'success': True,
            'user_id': user_id,
            'chat_history': chat_history,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'total_items': total_count,
                'has_next': page < total_pages,
                'has_prev': page > 1
            },
            'metadata': {
                'format': format,
                'include_charts': include_charts,
                'chart_count': chart_count,
                'conversation_count': len(kb_conversations),
                'total_messages': len(chat_history)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API chat history error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail={
                'error': 'Internal server error',
                'message': 'Failed to retrieve chat history',
                'details': str(e)
            }
        )

@app.get("/agent/api/user-memories/{user_id}")
async def get_user_memories(user_id: str):
    try:
        user_memories = memory.get_user_memories(user_id=user_id)
        
        return JSONResponse(content={
            "success": True,
            "user_id": user_id,
            "user_memories": [{
                "memory": mem.memory,
                "topics": mem.topics,
                "timestamp": mem.timestamp.isoformat() if isinstance(mem.timestamp, datetime) else mem.timestamp
            } for mem in user_memories]
        })
    except Exception as e:
        logger.error(f"User memories retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/agent/api/clear-session/{user_id}")
async def clear_user_session(user_id: str):
    try:
        session_id = active_sessions.get(user_id)
        if session_id:
            if session_id in memory.runs:
                del memory.runs[session_id]
            del active_sessions[user_id]
            memory.clear()


            
            return JSONResponse(content={
                "success": True,
                "message": f"Session and memories cleared for user {user_id}"
            })
        else:
            return JSONResponse(content={
                "success": True,
                "message": f"No active session found for user {user_id}"
            })
    except Exception as e:
        logger.error(f"Session clearing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting server on 127.0.0.1:8000")
    uvicorn.run(
        socket_app,
        host="127.0.0.1",
        port=8000,
        log_level="debug",
        access_log=True
    )
    
#in above code need to remove  the $ sign in non finance number and all chart and coloumn underscore free but tooltip and lable in axis not work 
