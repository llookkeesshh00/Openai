from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = OpenAIEmbeddings(model='text-embedding-3-small',dimensions=32)
result = model.embed_query("Delhi is the capital of India")

print(str(result))

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

result = model.embed_documents(documents)

print(str(result))