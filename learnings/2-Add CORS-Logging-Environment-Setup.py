from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn 

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )


@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Hello, World!"
    }

if __name__=="__main__":
    uvicorn.run('2-Add CORS-Logging-Environment-Setup:app', host="127.0.0.1", port=8000, reload=True)