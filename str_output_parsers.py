from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen3-Coder-Next",
    task="text generation"
)

model = ChatHuggingFace(llm=llm)

# prompt 1
template1 = PromptTemplate(
    template = "Write a detailed report on {topic}",
    input_variables=["topic"]
)


template2 = PromptTemplate(
    template="Write a 5 line summary of the following text. /n {text}"
)

prompt1 = template1.invoke({"topic":"ouput parsers"})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"text":result.content})

final_result = model.invoke(prompt2)

print(result.content)
print("----------------------------------------------------------------------")
print(final_result.content)