from llama_cpp import Llama
import time


start = time.time()

llm = Llama.from_pretrained(
    repo_id="bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF",
    filename="mistralai_Mistral-Small-3.2-24B-Instruct-2506-IQ4_XS.gguf",
    n_gpu_layers=-1,
    n_ctx=8384,
    n_batch=1024,
)

time_to_load = time.time() - start
print(f"Model loaded in {time_to_load:.2f}s")


def ask_with_monitoring(prompt: str) -> str:
    print(f"Sending prompt: '{prompt[:30]}...'")
    start_time = time.time()

    try:
        output = llm(
            prompt,
            max_tokens=1024,
            temperature=0.7,
            echo=False,
        )
        elapsed = time.time() - start_time
        print(f"Got response in {elapsed:.2f}s")
        return output["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error after {time.time() - start_time:.2f}s: {e}")
        return str(e)


# Interactive loop
while True:
    q = input("\nYou: ").strip()
    if q.lower() in {"exit", "quit", "bye"}:
        print("Bye!")
        break
    if q:
        print("AI:", ask_with_monitoring(q))
