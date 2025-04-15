from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from final_Response import get_gemini_response

app = FastAPI()

# Allow CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str


@app.get("/")
async def respond():
    response_text = "Hi how i can help for you ?"
    return {"response": response_text}
@app.post("/get-response")
async def respond_to_query(request: QueryRequest): 
    response_text = get_gemini_response(request.query)
    # print("queery",request.query)
    return {"response": response_text}
