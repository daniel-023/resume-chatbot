from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv


load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="mixtral-8x7b-32768"
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db",embedding_function=embeddings)


def rag(question: str, num_docs: int = 1):

    # 1. Retrieve relevant documents
    docs = vectorstore.similarity_search(question, 3)

    # 2. Format context from documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # 3. Create RAG prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an interactive resume chatbot. 
        Your task is to answer questions about Daniel's resume.
        Answer the question based on the provided context.
        Do not repeat what you have already said in the same answer.
        If you cannot answer from the context, say "I cannot answer this from the provided information."

        Context:
        {context}"""),
        ("human", "{question}")
    ])

    # 4. Get formatted prompt
    formatted_prompt = prompt.format(
        context=context,
        question=question
    )

    # 5. Get response
    response = llm.invoke(formatted_prompt)

    print("\n=== RAG Process Details ===")
    print("\nQuery:", question)
    print("\nRetrieved Context:", context)


    print("\nFormatted Prompt:", formatted_prompt)


    print("\n=========================== RAG Final Response ===========================")
    print("\nResponse:", response.content)

    return response.content

question = ""
while question != 'end':
    question = input("Type question here: ")
    print(rag(question))

