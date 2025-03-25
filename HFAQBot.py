import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure all necessary environment variables are available
required_env_vars = ['AZURE_API_VERSION', 'AZURE_ENDPOINT', 'AZURE_MODEL', 'AZURE_API_KEY']
file = "Upload a PDF file and start asking questions"
for var in required_env_vars:
    if not os.getenv(var):
        st.error(f"Missing environment variable: {var}")
        st.stop()  # Stop execution if a required environment variable is missing

# Verify the values of the environment variables (for debugging purposes)
# st.write(f"Using API version: {os.getenv('AZURE_API_VERSION')}")
# st.write(f"Using endpoint: {os.getenv('AZURE_ENDPOINT')}")
# st.write(f"Using model: {os.getenv('AZURE_MODEL')}")

# Initialize the LLM for Azure OpenAI
try:
    llm = AzureChatOpenAI(
        api_version=os.getenv('AZURE_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_ENDPOINT'),
        model=os.getenv('AZURE_MODEL'),
        api_key=os.getenv('AZURE_API_KEY'),
    )
   # st.success("LLM initialized successfully!") new comment
except Exception as e:
    st.error(f"Error initializing LLM: {e}")
    st.stop()

# Upload PDF files
# st.header("My First Chatbot") new comment
st.header("FAQ HBot App")

# with st.sidebar:
#     st.title("Your Documents")
#      file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")
#     st.write(file)

# Extract the text from the PDF
if file is not None:
    # pdf_reader = PdfReader("C:/Users/1000039837/OneDrive - Hexaware Technologies/39837_VM_Lab/Programming/PYCharm/21-1-24/work1/docFiles/ManuCon.pdf")
    pdf_reader = PdfReader("ManuCon.pdf")
    # pdf_reader = PdfReader("HexavarsityRewardsandIncentive.pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Text chunking parameters
    CHUNK_SIZE = 1000  # Default chunk size (in characters)
    MAX_CHUNK_SIZE = 2048  # Maximum chunk size allowed by the library
    CHUNK_SIZE = min(CHUNK_SIZE, MAX_CHUNK_SIZE)

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_text(text)

    # Initialize the embeddings using Azure's OpenAI API
    try:
        embeddings = AzureOpenAIEmbeddings(
            api_key=os.getenv('AZURE_API_KEY'), # Pass the correct API key
            azure_endpoint=os.getenv('AZURE_ENDPOINT'), # Pass the correct endpoint
            chunk_size=CHUNK_SIZE  # Ensure chunk size is within limit



        )
       # st.success("Embeddings initialized successfully!") new comment
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        st.stop()

    # Create the vector store with FAISS
    try:
        # Ensure embeddings are passed properly (textual data, not embeddings themselves)
        vector_store = FAISS.from_texts(chunks, embeddings)
       # st.success("Vector store created successfully!") new comment
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.stop()

    # Ask a question and perform similarity search
    user_question = st.text_input("Type your question here:")

    if user_question:
        try:
            # Perform the similarity search to retrieve relevant documents
            matches = vector_store.similarity_search("When will the registrations start")

            # Load the QA chain and pass the matched documents
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=matches, question=user_question)

            if response == "I don't know.":
                st.write("Please contact Jamuna - jamunaranik@hexaware.com")
            else :
                # Display the response from the LLM
                st.write(response)

        except Exception as e:
            st.error(f"Error during similarity search or LLM execution: {e}")
