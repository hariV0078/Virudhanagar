#!/bin/bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 30
