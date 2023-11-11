#!/bin/bash
#Works on a 4 GPU machine

CUDA_VISIBLE_DEVICES=0 uvicorn app1_module:app --host 0.0.0.0 --port 8000 &
CUDA_VISIBLE_DEVICES=1 uvicorn app2_module:app --host 0.0.0.0 --port 8001 &
CUDA_VISIBLE_DEVICES=2 uvicorn app3_module:app --host 0.0.0.0 --port 8002 &
CUDA_VISIBLE_DEVICES=3 uvicorn app4_module:app --host 0.0.0.0 --port 8003 &
