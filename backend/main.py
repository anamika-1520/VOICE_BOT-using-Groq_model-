from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from persona import SYSTEM_PROMPT
import os
import traceback
import dotenv
dotenv.load_dotenv()

app = FastAPI()

# Initialize ChatGroq safely
try:
    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model="llama-3.3-70b-versatile",
        temperature=0.6
    )
except Exception as e:
    print("❌ Groq init error:", e)
    raise e


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_bot(data: Question):
    try:
        prompt = f"""
{SYSTEM_PROMPT}

Question: {data.question}
Answer:
"""

        response = llm.invoke(prompt)

        # SAFELY extract content
        answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        return {"answer": answer}

    except Exception as e:
        print("❌ ERROR in /ask")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
