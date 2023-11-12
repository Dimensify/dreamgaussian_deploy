#!/bin/bash
#Works on a 4 GPU machine

CUDA_VISIBLE_DEVICES=0 uvicorn main:app --host 0.0.0.0 --port 8000 &
CUDA_VISIBLE_DEVICES=1 uvicorn main:app --host 0.0.0.0 --port 8001 &
CUDA_VISIBLE_DEVICES=2 uvicorn main:app --host 0.0.0.0 --port 8002 &
CUDA_VISIBLE_DEVICES=3 uvicorn main:app --host 0.0.0.0 --port 8003 &
