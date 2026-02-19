# AI Science Discovery Team 🔬

> **Automating Scientific Discovery from First Principles**

This project creates an autonomous AI team designed to tackle frontier science problems by simulating a multi-agent research process. Instead of relying on statistical analogies from training data (which limits models to "known" science), this architecture forces models to reason from first principles, enabling the discovery of novel physical pathways and engineering solutions.

---

## 💡 The Core Concept

### The Problem: Statistical Bias in LLMs
Standard Large Language Models (LLMs) are probabilistic. When asked about a frontier science problem (e.g., "How do we build a warp drive?"), they tend to revert to the "most likely" answer found in their training data: *"That is currently impossible"* or *"Here is a summary of existing sci-fi concepts."* They struggle to genuinely reason through new physics because their "reasoning" is often just retrieving analogies.

### The Solution: First-Principles Reasoning Chain
This project implements a **10-step Multi-Agent Pipeline** that breaks down complex scientific tasks into isolated, fundamental physics questions.

**Key Innovation:** We **decontextualize** the questions.
*   Instead of asking: *"How do I build a room-temperature superconductor?"* (which triggers the "that's hard/impossible" bias),
*   We ask: *"Given material structure X with electron mobility Y at temperature Z, would a Cooper pair form?"*

By stripping away the "impossible" goal, the model is forced to solve the specific physics equation/logic puzzle in front of it. We then chain these "atomic" physics validations together to build up a complex, novel solution that the model wouldn't have proposed directly.

---

## 🧠 Architecture: The 10-Step Pipeline

The system mimics a full human scientific research team:

| Step | Agent / Role | Function |
| :--- | :--- | :--- |
| **1-2** | **Orchestrator** | Takes a vague problem (e.g., "Interstellar Travel") and frames a precise, isolated physics target (e.g., "Manipulating spacetime metric tensor"). |
| **3** | **Hypothesis Generator** | A "wild idea" generator. It proposes multiple theoretical approaches, explicitly ignoring engineering practicality to maximize novelty. |
| **4** | **Step Decomposer** | Breaks a hypothesis into atomic steps. **Crucially**, it rewrites each step as a standalone physics exam question, hiding the original context. |
| **5** | **Physics Oracle** | The core reasoning engine. It answers the decomposer's questions using *only* fundamental laws (thermodynamics, QED, etc.), without knowing what the bigger picture is. |
| **6** | **Chain Assembler** | Takes the validated steps and stitches them back together into a coherent physical pathway. Checks for logical gaps or contradictions. |
| **7** | **Engineering Proposer** | Proposes how to actually build the machinery to achieve the physical conditions (e.g., "To get this magnetic field, we need a solenoid of size X..."). |
| **8** | **Requirements Challenger** | A "Devil's Advocate" loop. It relentlessly questions requirements ("Do we *really* need 100 Tesla? What if we used resonance instead?") to make the impossible practical. |
| **9** | **Overseer** | Synthesizes all approaches, validations, and engineering proposals. It spots cross-cutting insights and selects the most promising solution. |
| **10** | **Final Evaluator** | Writes a comprehensive scientific thesis, including theoretical basis, experimental design, and risk analysis. |

---

## 🛠️ Implementation vs. Original Idea

### Original Idea (The Vision)
The original concept envisioned a team of specialized, custom-trained models:
*   A **"Physics Next-Token" Model**: A model pre-trained *only* on physics equations and logic, so it literally cannot "speak" anything but physics.
*   **Massive Parallel Simulation**: Simulating 1000s of material variations simultaneously.
*   **Deep Simulation loops**: Integrating actual physics simulators (DFT, FEM) into the loop.

### Current Implementation (The Portfolio MVP)
This repository represents a **Functional MVP** of that vision, adapted to run on standard hardware/APIs:
*   **Persona-Based Agents**: Instead of training new models, we use strict system prompts to force standard LLMs (like GPT-4, Claude, or local Llama 3) into specific cognitive modes (e.g., "You are a Physics Oracle. You do not know about human history, only laws of physics").
*   **Decontextualization Logic**: Implemented via prompt engineering ensuring information leakage between the "Goal" and the "Validation" steps is minimized.
*   **Web Dashboard**: A real-time Flask interface to watch the "team" think, debug their reasoning steps, and visualize the chain of thought.

---

## 🚀 Getting Started

### Prerequisites
1.  **Python 3.8+**
2.  **LM Studio** (Recommended for local inference) or an OpenAI-compatible API endpoint.
    *   *Note: This project is optimized for local open-weights models to ensure cost-effective reasoning loops.*

### Installation
1.  Clone the repo:
    ```bash
    git clone https://github.com/yourusername/ai-science-team.git
    cd ai-science-team
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Discovery Engine
1.  Start your LLM server (e.g., LM Studio on port 1234).
2.  Run the dashboard:
    ```bash
    python app.py
    ```
3.  Open `http://localhost:5050` in your browser.
4.  Enter a frontier problem (e.g., *"Design a material that is superconducting at room temperature"*).
5.  Watch the agents work!

---

## 📂 Project Structure

*   `pipeline.py`: The main workflow engine. Manages the state and data flow between all 10 agents.
*   `agents.py`: Contains the "Prompts as Code". This is where the personality and strict constraints of each agent (Oracle, Challenger, etc.) are defined.
*   `app.py`: Flask web server for the UI.
*   `llm_client.py`: Handles communication with the AI models.
*   `results/`: Stores the generated discovery logs and final theses.

---

## 🔮 Future Roadmap
*   **Tool Use**: Give the Physics Oracle access to Python for actual calculation (WolframAlpha, calc).
*   **Vector Memory**: Allow the Overseer to remember past experiments to avoid repeating failed paths.
*   **Simulation Integration**: Connect to PySCF or similar libraries to validate quantum chemistry claims.
