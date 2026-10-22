"""Run the FastAPI inference service with Uvicorn."""

import uvicorn

from transformers_from_scratch.serving import create_app

app = create_app()


if __name__ == "__main__":
    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=False)
