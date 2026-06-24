#!/usr/bin/env python
"""
Launcher script for the Database Agent web interface.

Starts up the Streamlit web interface for the conversational database agent,
pre-validating environment requirements for Gemini, LangGraph, and LangChain.
"""

import os
import subprocess
import sys
import dotenv

# Load .env variables before verifying environment or starting subprocesses
dotenv.load_dotenv()



def verify_environment():
    """Verify essential ecosystem keys are set up before spinning up nodes."""
    print("[Environment] Verifying LangChain & Gemini Environment Configurations...")
    
    # 1. Gemini / Google Gen AI Verification
    # LangChain's ChatGoogleGenerativeAI typically uses GEMINI_API_KEY or GOOGLE_API_KEY
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        print("[Warning] GEMINI_API_KEY / GOOGLE_API_KEY environment variable is not set.")
        print("Make sure to declare it in your local environment or '.env' file.")
        print("   Example: export GEMINI_API_KEY='AIzaSy...'")
        print("---")

    # # 2. Optional: LangSmith / LangChain Tracing (Highly recommended for LangGraph)
    # if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    #     if not os.getenv("LANGCHAIN_API_KEY"):
    #         print("ℹ️  Note: LangChain Tracing is enabled, but LANGCHAIN_API_KEY is missing.")
    #         print("   Graph traces might not emit to LangSmith.")
    #         print("---")


def main():
    """Launch the Streamlit web interface."""
    print("[Launcher] Launching Streamlit Web Interface for Gemini-LangGraph Agent")

    # Perform runtime ecosystem sanity checks
    verify_environment()

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to your Streamlit user interface entry point
    # Updated path pattern based on typical framework layout (app/ui/streamlit_wrapper.py)
    wrapper_path = os.path.join(script_dir, "ui", "streamlit_wrapper.py")

    # Fallback to current directory structure if UI folder lives outside
    if not os.path.exists(wrapper_path):
        wrapper_path = os.path.join(script_dir, "streamlit_wrapper.py")

    # Check if the file exists
    if not os.path.exists(wrapper_path):
        print(f"[Error] Could not find Streamlit entry-point file at {wrapper_path}")
        sys.exit(1)

    try:
        # Build pristine environment dictionary
        env = os.environ.copy()

        # Inject the project root into PYTHONPATH to allow modular imports like
        # 'from app.agents.base import BaseAgent' inside your graph definitions.
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{script_dir}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = script_dir

        print(f"[PYTHONPATH] Setting PYTHONPATH to target: {script_dir}")
        print("[Agent] Running Graph network pipeline...")

        # Resolve streamlit executable path relative to sys.executable (useful in virtual environments)
        python_dir = os.path.dirname(sys.executable)
        streamlit_executable = "streamlit"
        for ext in ["", ".exe"]:
            candidate = os.path.join(python_dir, "streamlit" + ext)
            if os.path.exists(candidate):
                streamlit_executable = candidate
                break

        print(f"[Launcher] Using Streamlit executable: {streamlit_executable}")

        # Run streamleted client UI
        subprocess.run(
            [
                streamlit_executable,
                "run",
                wrapper_path,
                "--server.headless",
                "true",
            ],
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"[Error] Failed to execute LangGraph Streamlit application: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("[Error] 'streamlit' executable not found in your active environment path.")
        print("[Tip] Ensure dependencies are ready: pip install streamlit langchain-google-genai langgraph")
        sys.exit(1)


if __name__ == "__main__":
    main()