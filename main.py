from fastapi import FastAPI
from routes import chat_routes as routes



app = FastAPI()
app.include_router(routes.router)