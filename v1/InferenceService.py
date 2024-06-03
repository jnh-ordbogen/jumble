import asyncio
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from InferenceEngine import InferenceEngine

app = Starlette()

# Define the number of available GPUs and create an InferenceEngine for each GPU
available_gpus = ["cuda:0", "cuda:1"]  # Add more GPUs as needed
inference_engines = {gpu: InferenceEngine(device=gpu) for gpu in available_gpus}

# Define a lock to ensure thread-safe access to the inference engines
engine_lock = asyncio.Lock()

@app.route("/translate", methods=["POST"])
async def translate(request):
    try:
        data = await request.json()
        text = data.get("text", "")

        # Acquire a lock to access an available engine
        async with engine_lock:
            for gpu, engine in inference_engines.items():
                if engine.is_available():
                    result = engine.translate(text)
                    return JSONResponse({"translation": result})

        # No available engines, return an error response
        return JSONResponse({"error": "No available engines"}, status_code=503)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8901)
