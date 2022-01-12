from fastapi import APIRouter
import torch

router = APIRouter()

@router.get('/check_cuda')
def check_cuda():
    return {
        'is_available': torch.cuda.is_available()
    }