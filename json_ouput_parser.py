from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id= "Qwen/Qwen3-Coder-Next",
    task="text generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

# prompt 1
template = PromptTemplate(
    template = "Write name of the characters, movie/series title, summary, sentiment from {review}. /n {format_instructions}",
    input_variables=["review"],
    partial_variables={"format_instructions" : parser.get_format_instructions()}
)

# prompt = template.invoke({"review" : """
# I've watched up to episode four. The casting of Duncan and Egg is fantastic.
# Both actors have the skill to do the characters justice. Duncan hits the right note of not being the smartest but also not a dunce.
# Of being a moral man but not too preachy. Of showing fear and vulnerability but not cowardice. Having bravery but not an ego.
# And Egg is clearly intelligent and boisterous but still somehow innocent and endearing.

# The first episodes are slower but it's smart pacing. I liken it to opening a good bottle of wine and letting it breathe.
# It gives the audience time to get acquainted with Dunc and Egg, their personalities and how they relate to each other.
# I find some shows just want to jump into the action and shock you from the start and think you will care about the characters
# just because well, they are the main characters. A Knight of the Seven Kingdoms let's you see that Dunc and Egg are both
# empathetic characters with a moral compass, both loveable in their own ways over the first couple of episodes. By the time the
# action starts, you sincerely care about their fate. Its been a while since I've felt this invested in a character.
# I can't wait for each new episode.
# """})

# result = model.invoke(prompt)

# print(result.content)

# print("__________________________________________________")
# final_result = parser.parse(result.content)
# print(final_result)


# using chain

chain = template | model | parser

result = chain.invoke({"review" : """
I've watched up to episode four. The casting of Duncan and Egg is fantastic.
Both actors have the skill to do the characters justice. Duncan hits the right note of not being the smartest but also not a dunce.
Of being a moral man but not too preachy. Of showing fear and vulnerability but not cowardice. Having bravery but not an ego.
And Egg is clearly intelligent and boisterous but still somehow innocent and endearing.
The first episodes are slower but it's smart pacing. I liken it to opening a good bottle of wine and letting it breathe.
It gives the audience time to get acquainted with Dunc and Egg, their personalities and how they relate to each other.
I find some shows just want to jump into the action and shock you from the start and think you will care about the characters
just because well, they are the main characters. A Knight of the Seven Kingdoms let's you see that Dunc and Egg are both
empathetic characters with a moral compass, both loveable in their own ways over the first couple of episodes. By the time the
action starts, you sincerely care about their fate. Its been a while since I've felt this invested in a character.
I can't wait for each new episode.
"""})

print(result)
