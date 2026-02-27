from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel,EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str
    age : Optional[int] = None
    email : Optional[EmailStr] = None
    cgpa : float = Field(ge=0,le=10,description="CGPA of student between 0 and 10")


new_student = {"name" : "Rio", "cgpa":9}

student = Student(**new_student)

print(student)