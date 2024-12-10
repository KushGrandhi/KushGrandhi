from flask import Flask, request, jsonify, render_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import os

# Set up the environment variable for the Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAlZP2f4fR_4Tg95KcwCboq-c-wIdSl3AY"

# Initialize Flask app
app = Flask(__name__)

# Load the model
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Load the document
file_path = "bio.txt"
loader = TextLoader(file_path)
docs = loader.load()

# Define the prompt template
template = """You are a conversational chatbot on a website and you will only answer things that are given to you in the context, dont use any knowledge from outside, given context about you: {text}"""
prompt = PromptTemplate.from_template(template)

# Create the chain
llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

# Function to process user input
def mess(q, messages=[]):
    messages.append(f"User: {q}")
    response = stuff_chain.invoke({
        "input_documents": docs,  # Pass the documents here
        "text": q                 # Pass the user input as context
    })
    chatbot_reply = response.get("output_text", "No response available.")
    messages.append(f"Chatbot: {chatbot_reply}")
    return chatbot_reply, messages

# Define an API endpoint
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the user query from the request
        data = request.json
        user_query = data.get('query', '')

        if not user_query:
            return jsonify({"error": "Query not provided"}), 400

        # Generate chatbot response
        chatbot_reply, messages = mess(user_query)

        return jsonify({
            "user_query": user_query,
            "chatbot_reply": chatbot_reply,
            "conversation_history": messages
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Run the server
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
