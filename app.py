from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
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
os.environ["GOOGLE_API_KEY"] = "Your_API_KEY"

# Database details
db_user = "root"
db_password = ""
db_host = "127.0.0.1"
db_name = "employee_database"

# Initialize SQL database connection
try:
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")
    print("Database connection successful.")
except Exception as e:
    print(f"Database connection failed: {str(e)}")

# Initialize LLM for query generation
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)
generate_query = create_sql_query_chain(llm, db)

# Function to clean SQL query
def clean_sql_query(query):
    # Remove unnecessary prefixes and SQL formatting tags
    query = query.replace("SQLQuery:", "").replace("```sql\n", "").replace("\n```", "").strip()

    # Remove any LIMIT clause
    # if "LIMIT" in query:
    #     query = query.split("LIMIT")[0].strip()

    return query


# Prepare prompt template
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
Question: {question}
SQL Query: {clean_query}
SQL Result: {result}
Answer: """
)

# Rephrase the answer using LLM
rephrase_answer = answer_prompt | llm | StrOutputParser()

# Function to execute query chain
def execute_query_chain(query):
    try:
        # Generate SQL query
        generated_query = generate_query.invoke({"question": query})
        clean_query = clean_sql_query(generated_query)
        print(f"Generated SQL Query: {clean_query}")  # Debugging output

        # Execute the SQL query
        execute_query = QuerySQLDataBaseTool(db=db)
        result = execute_query.invoke({"query": clean_query})
        print(f"SQL Query Result: {result}")  # Debugging output

        # Define the chain
        chain = (
            RunnablePassthrough.assign(clean_query=lambda context: clean_sql_query(generate_query.invoke({"question": context["question"]})))
            .assign(result=itemgetter("clean_query") | execute_query)
            | rephrase_answer
        )
        response = chain.invoke({"question": query})
        return response

    except Exception as e:
        print(f"Error during query execution: {str(e)}")
        return f"An error occurred: {str(e)}"

# Initialize Flask app
flask_app = Flask(__name__)
CORS(flask_app)

@flask_app.route('/')
def home():
    return render_template('index.html')

@flask_app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    user_question = data.get('message')
    if not user_question:
        return jsonify({"response": "Please provide a question."}), 400

    # Process the question
    response = execute_query_chain(user_question)
    return jsonify({"response": response})

if __name__ == "__main__":
    flask_app.run(host='0.0.0.0', port=5000)
