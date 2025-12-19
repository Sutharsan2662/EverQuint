A self-contained reasoning agent built with Streamlit that uses a "Plan-Execute-Verify" loop to solve word problems. Powered by Mistral 7B via Ollama, 
it handles multi-step logic and mathematical verification in one clean interface.

Key Features
->Three-Phase Reasoning: Automatically generates a plan, executes the logic with Python-style deductions, and verifies the result.
->Interactive Chat: Refine answers or provide feedback in real-time to trigger retries.
->Evaluation Suite: Built-in benchmarking tool to test accuracy against local JSON test cases.
->Local-First: Runs entirely on your machine using Ollama for privacy and speed.

Quick Start
->Install Ollama: Ensure [Ollama](https://ollama.com/) is installed and running.
->Pull the Model: ollama pull mistral:7b-instruct