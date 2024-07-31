import os
import PyPDF2
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set API key
os.environ['OPENAI_API_KEY'] = "API-Key"  

# Load and read PDF
pdf_reader = PyPDF2.PdfReader("ABSTRACT.pdf")
text = ''
for page in pdf_reader.pages:
    text += page.extract_text()

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1000,
chunk_overlap=100,
length_function=len,
is_separator_regex=False,
)
chunks = text_splitter.create_documents([text])


# Create embeddings and ChromaDB
db = Chroma.from_documents(chunks, OpenAIEmbeddings())

# Set up retriever
retriever = db.as_retriever()

# Initialize LLM model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Define prompt template
template = """
            <s>
            Using the information contained in the context,
            give a comprehensive answer to the question.
            Respond only to the question asked, response should be concise and relevant to the question.
            Provide the number of the source document when relevant.
            If the answer cannot be deduced from the context, do not give an answer.

            Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
            </s>
            ------
            <ctx>
            {context}
            </ctx>
            ------
            <hs>
            {history}
            </hs>
            ------
            {question}
            Answer:
            """
prompt = PromptTemplate(
                        input_variables=["history", "context", "question"],
                        template=template,
                            )

# Set up RetrievalQA
qa = RetrievalQA.from_chain_type(
                                    llm=llm,
                                    chain_type='stuff',
                                    retriever=retriever,
                                    verbose=True,
                                    chain_type_kwargs={
                                        "verbose": True,
                                        "prompt": prompt,
                                        "memory": ConversationBufferMemory(
                                            memory_key="history",
                                            input_key="question"),
                                    },
                                    )

print(qa.run("Give me the summary"))
