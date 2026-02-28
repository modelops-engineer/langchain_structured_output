from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI()

# JSON Schema
json_schema = {
  "characters": {
    "type": "array",
    "items": {
      "type": "string"
    },
    "description": "Give the names of characters discussed in the review"
  },
  "summary": {
    "type": "string",
    "description": "A brief summary of the review"
  },
  "sentiment": {
    "type": "string",
    "enum": ["positive", "neutral", "negative"],
    "description": "Return sentiment of the review"
  },
  "movie_title": {
    "type": ["string", "null"],
    "default": None,
    "description": "Write the name of the movie/series if written in the review"
  },
  "required" : ["summary", "sentiment"]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""
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
""")


print(result.summary)
