import streamlit as st
import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter


# Set environment variables
os.environ["GOOGLE_API_KEY"] = "AIzaSyDWv3Nd2zcstU0nmqjL29OcLxJ8B-QsOTs"

# Database connection details
db_user = "root"
db_password = ""
db_host = "127.0.0.1"
db_name = "retail_sales_db"

# Initialize the SQL database connection
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# Initialize LLM for query generation
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)
generate_query = create_sql_query_chain(llm, db)

# Function to clean SQL query
def clean_sql_query(query):
    return query.replace("```sql\n", "").replace("\n```", "")

# Prepare the prompt template
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
Question: {question}
SQL Query: {clean_query}
SQL Result: {result}
Answer: """
)

# Rephrase the answer using LLM
rephrase_answer = answer_prompt | llm | StrOutputParser()

# Define the Streamlit interface
def execute_query_chain(query):
    # Generate and clean the SQL query
    clean_query = clean_sql_query(generate_query.invoke({"question": query}))

    # Execute the SQL query
    execute_query = QuerySQLDataBaseTool(db=db)
    result = execute_query.invoke(clean_query)
    
    # Define the chain with clean query and rephrased answer
    chain = (
        RunnablePassthrough.assign(clean_query=lambda context: clean_sql_query(generate_query.invoke({"question": context["question"]})))
        .assign(result=itemgetter("clean_query") | execute_query)
        | rephrase_answer
    )
    
    # Execute the chain
    response = chain.invoke({"question": query})
    return response

# Streamlit layout
st.title("SQL Query Generator")
st.write("This app allows you to generate SQL queries based on user questions and display the results.")

# User input for question
user_question = st.text_input("Enter your question:")

if user_question:
    # Call the function to get the response
    response = execute_query_chain(user_question)
    st.write("Answer:", response)