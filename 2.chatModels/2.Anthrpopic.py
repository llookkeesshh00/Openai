from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-3-0', max_tokens=1024)

result = model.invoke("What is the capital of India?")

print(result.content)