from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",
    task="text generation"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name : str = Field(description="Name of the person")
    age : int = Field(gt=18, description="Age of the person")
    city: str = Field(description='Name of the city person lives')

parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template="Generate a {identity} fictional character and give name, age and city \n {format_instructions}",
    input_variables=["identity"],
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

# prompt = template.invoke({'identity':'Indian'})

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser

final_result = chain.invoke({'identity' : 'Canadian'})

print(final_result)