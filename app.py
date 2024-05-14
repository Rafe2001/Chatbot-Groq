from flask import  Flask, request, render_template
import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods = ['POST'])
def chat():
    user_question = request.form['user_question']
    model_name = request.form['model_name']
    memory_length = int(request.form['memory_length'])
    
    response = get_chatbot_response(user_question, model_name, memory_length)
    
    return render_template('chat.html', user_question=user_question, chatbot_response=response)

def get_chatbot_response(question, model_name, memory_length):
    # Load environment variables
    groq_api_key = os.environ.get('GROQ_API_KEY')

    # Create a GROQ chat instance
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

    # Initialize conversation memory
    conversation_memory = ConversationBufferWindowMemory(k=memory_length)

    # Create a ConversationChain
    conversation_chain = ConversationChain(llm=groq_chat, memory=conversation_memory)

    # Get the response from the chatbot
    response = conversation_chain(question)['response']

    return response

if __name__ == '__main__':
    app.run(debug=True)
    
