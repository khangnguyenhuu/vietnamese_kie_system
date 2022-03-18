from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from routes import check_cuda, key_info_extract
from utils.logger import logger
import time

app = FastAPI()

# Middlewares
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.middleware('http')
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f'{request.method}: {request.url}')
    logger.info(f'Execution time: {process_time:.5f}s')
    return response

# Router
app.include_router(check_cuda.router)
app.include_router(key_info_extract.router)

# Routes
@app.get('/')
def home():
    return {'message': 'Hello World!'}
