import streamlit as st
import psycopg2
import pandas as pd
import plotly.express as px
from vanna.openai.openai_chat import OpenAI_Chat
from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore

class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

def get_ddl_statements(conn):
    cursor = conn.cursor()
    
    # Fetch schema data
    cursor.execute("""
        SELECT table_name, column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public'
    """)
    columns = cursor.fetchall()
    
    cursor.execute("""
        SELECT tc.table_name, kcu.column_name, 
               ccu.table_name AS foreign_table,
               ccu.column_name AS foreign_column
        FROM information_schema.table_constraints AS tc 
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
    """)
    foreign_keys = cursor.fetchall()
    
    # Generate DDL
    ddl_statements = []
    tables = set([col[0] for col in columns])
    
    for table in tables:
        cols = [col for col in columns if col[0] == table]
        fks = [fk for fk in foreign_keys if fk[0] == table]
        
        ddl = f"CREATE TABLE {table} (\n"
        for col in cols:
            ddl += f"  {col[1]} {col[2]},\n"
        for fk in fks:
            ddl += f"  FOREIGN KEY ({fk[1]}) REFERENCES {fk[2]}({fk[3]}),\n"
        ddl = ddl.rstrip(",\n") + "\n);"
        ddl_statements.append(ddl)
    
    return ddl_statements

# Streamlit UI
st.set_page_config(page_title="NL to SQL Assistant", layout="wide")
st.title("Natural Language to SQL Assistant")

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False

# Database Connection Form
with st.sidebar:
    st.header("Database Configuration")
    host = st.text_input("Host")
    database = st.text_input("Database Name")
    user = st.text_input("User")
    password = st.text_input("Password", type="password")
    openai_key = st.text_input("OpenAI API Key", type="password")
    
    if st.button("Connect & Train"):
        try:
            # Establish database connection
            conn = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password
            )
            
            # Initialize Vanna
            vn = MyVanna(config={'api_key': openai_key, 'model': 'gpt-4'})
            
            # Get and train DDL
            ddl_statements = get_ddl_statements(conn)
            for ddl in ddl_statements:
                vn.train(ddl=ddl)
            
            st.session_state.conn = conn
            st.session_state.vn = vn
            st.session_state.trained = True
            st.success("Database connected and model trained!")
            
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")

# Main Chat Interface
if st.session_state.trained:
    st.subheader("Ask your data question")
    question = st.text_input("Enter your question:", key="query_input")
    
    if question:
        vn = st.session_state.vn
        conn = st.session_state.conn
    
        try:
            # Generate SQL
            sql = vn.generate_sql(question)
            
            # Execute Query
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            
            # Display Results
            st.subheader("Generated SQL")
            st.code(sql)
            
            st.subheader("Query Results")
            
            # Interactive visualization toggle
            viz_type = st.selectbox("View as:", ["Table", "Bar Chart", "Line Chart", "Pie Chart"])
            
            if viz_type == "Table":
                st.dataframe(df)
            else:
                try:
                    if viz_type == "Bar Chart":
                        fig = px.bar(df, x=columns[0], y=columns[1])
                    elif viz_type == "Line Chart":
                        fig = px.line(df, x=columns[0], y=columns[1])
                    elif viz_type == "Pie Chart":
                        fig = px.pie(df, names=columns[0], values=columns[1])
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Couldn't create chart: {str(e)}")
                    st.dataframe(df)
            
            # Feedback mechanism
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Result looks good"):
                    vn.train(sql=sql, question=question)
                    st.success("Feedback recorded! Model improved.")
            with col2:
                if st.button("👎 Result incorrect"):
                    corrected_sql = st.text_area("Please provide correct SQL:")
                    if st.button("Submit Correction"):
                        vn.train(sql=corrected_sql, question=question)
                        st.success("Correction saved. Thank you!")
        
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")

else:
    st.info("Please configure database connection and OpenAI API key in the sidebar")