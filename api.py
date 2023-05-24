from fastapi import FastAPI, Body, WebSocket
from fastapi.responses import FileResponse
# import engine
import langchain_bot

app = FastAPI()
engine_instance = None

@app.get("/")
def render_ui():
  return FileResponse("ui/assistant.html")

@app.get("/send-icon")
def send_icon():
  return FileResponse("ui/send-message.png")

@app.get("/api")
def hello():
  return "Hi! Welcome to AIPA APIs."

@app.post("/api/query")
def ask(query: str = Body(..., embed=True)):
  global langchain_bot
  return langchain_bot.answer_user_query(query).response[0]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
      while True:
        global langchain_bot
        data = await websocket.receive_text()
        response = langchain_bot.answer_user_query(data)
          #await websocket.send_text(f"{response[0]}\n"+ "\n" + "Source: " + f"{response[1]}")
        await websocket.send_json({"answer":f"{response[0]}", "source":f"{response[1]}"})
    except:
      print('exception occured')