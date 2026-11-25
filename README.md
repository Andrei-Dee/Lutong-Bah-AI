Lutong-Bah-AI

Project Structure

Show Code
Download diagram SVG

Prerequisites

This project uses uv for Python and dependency management.

Install uv

• macOS / Linux:bash
curl -LsSf https://astral.sh | sh
Use code with caution.
• Windows (PowerShell):powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh | iex"
Use code with caution.

Getting Started

1. Clone the repository:bash
   git clone <your-repo-url>
   cd Lutong-Bah-AI
   Use code with caution.
2. Install dependencies and set up the environment:bash
   uv sync
   Use code with caution.
   (This automatically downloads the required Python version, creates .venv, and installs pandas and scikit-learn from uv.lock.)

Usage

Run the kNN script directly from the project root:
bash
uv run src/lutong_bah_ai/kNN.py
Use code with caution.

Adding New Packages

To install additional libraries and update pyproject.toml and uv.lock:
bash
uv add <package-name>
Use code with caution.
