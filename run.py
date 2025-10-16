import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",   # path to your FastAPI app
        host="0.0.0.0",
        port=8080,
        reload=True       # optional, useful for development
    )
