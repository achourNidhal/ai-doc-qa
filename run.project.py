import subprocess
import sys
import socket
import argparse
import os

processes = []

def is_port_open(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = s.connect_ex(("localhost", port))
    s.close()
    return result == 0


def run(cmd, cwd=None):
    p = subprocess.Popen(cmd, cwd=cwd, shell=True)
    processes.append(p)


def start_redis(redis_port):
    if is_port_open(redis_port):
        print(f"✅ Redis already running on port {redis_port}")
    else:
        print(f"🚀 Starting Redis on port {redis_port}")
        run(f"docker run -p {redis_port}:6379 redis")


def start_ollama(model):
    try:
        subprocess.check_output("ollama list", shell=True)
        print("✅ Ollama already running")
    except:
        print("🚀 Starting Ollama server")
        run("ollama serve")

    print(f"📦 Ensuring model {model} exists")
    subprocess.run(f"ollama pull {model}", shell=True)


def start_backend():
    print("⚙️ Starting FastAPI backend")

    if os.name == "nt":
        cmd = "venv\\Scripts\\activate && uvicorn main:app --reload"
    else:
        cmd = "source venv/bin/activate && uvicorn main:app --reload"

    run(cmd, cwd="backend")


def start_frontend():
    print("🖥 Starting Streamlit frontend")
    run("streamlit run app.py", cwd="frontend")


def main():

    parser = argparse.ArgumentParser(description="Run AI Document Q&A stack")

    parser.add_argument("--model", default="llama3")

    parser.add_argument("--redis-port", default=6379, type=int)

    parser.add_argument("--skip-redis", action="store_true")

    parser.add_argument("--skip-ollama", action="store_true")

    args = parser.parse_args()

    print("\n🚀 Starting AI Document Q&A Project\n")

    try:

        if not args.skip_redis:
            start_redis(args.redis_port)

        if not args.skip_ollama:
            start_ollama(args.model)

        start_backend()

        start_frontend()

        print("\n✅ All services started")
        print("Frontend → http://localhost:8501")
        print("API → http://localhost:8000")

        for p in processes:
            p.wait()

    except KeyboardInterrupt:

        print("\n🛑 Stopping services")

        for p in processes:
            p.terminate()

        sys.exit(0)


if __name__ == "__main__":
    main()