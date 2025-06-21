# Gradiator - Developer Guide

## 1. Introduction

Welcome to the Gradiator developer documentation. Gradiator is a sophisticated autograding application designed for qualitative analysis of graduate-level software engineering assignments. It leverages Large Language Models (LLMs) to provide nuanced, rubric-based feedback that goes beyond simple pass/fail checks.

The system is built to be modular, configurable, and extensible, allowing it to adapt to different assignments, courses, and evaluation criteria.

### 1.1. Core Philosophy

- **Qualitative over Quantitative:** The primary goal is to provide high-quality, human-like feedback on code structure, readability, maintainability, and adherence to best practices.
- **Modularity and Extensibility:** The system is designed with clear separation of concerns, allowing new components (like LLM providers or analysis tools) to be added with minimal changes to the core logic.
- **Configuration over Code:** As much as possible, the behavior of the grader for a specific assignment is controlled through external configuration files (JSON, YAML, .txt) rather than hard-coded logic.

### 1.2. Core Features

-   **Multi-Format Structured Output**: Generates three distinct reports for every submission:
    1.  **Comprehensive Markdown**: A detailed technical report for instructors.
    2.  **Structured JSON**: Machine-readable data for further analysis or integration.
    3.  **Student-Friendly Summary**: A supportive, LLM-generated summary focusing on learning and improvement.
-   **Dynamic Project-Based Configuration**: Easily switch between different assignments or courses. Each project has its own dedicated configuration, rubrics, and prompts.
-   **Parallel LLM Processing**: Evaluates multiple rubric criteria simultaneously for significantly faster grading, with configurable parallelism.
-   **Multimodal Analysis**: Can analyze submissions that include images and PDFs (e.g., design diagrams, reports), not just code.
-   **Pluggable LLM Support**: Natively supports **Google Gemini**, **OpenAI GPT**, and local models via **Ollama**. The factory and strategy patterns make it easy to add more.
-   **Intelligent Token Management**: Automatically processes codebases with `repomix`, counts tokens, and applies compression if the context exceeds the LLM's limit.
-   **Integrated Static Analysis**: Leverages `Semgrep` with custom rule sets to find code quality issues, security vulnerabilities, and design pattern violations, and then uses an LLM to interpret the results.
-   **Advanced File Filtering**: Project configurations allow for precise control over which files are included in the analysis using `ignore_patterns`, `keep_patterns`, and `max_file_size`.
## 2. Architectural Overview

Gradiator employs several key software design patterns to ensure a robust and maintainable architecture.

### 2.1. High-Level Workflow

The application follows a pipeline architecture, where a student's submission (a ZIP file) is processed through a series of stages to produce a final evaluation.

```mermaid
graph TD
    A[Input: Student Submission ZIP] --> B{AutoGrader Core};
    B --> C[1. Process Submission];
    C -- Code Submission --> D[Repomix Processor];
    C -- Report Submission --> E[Report Processor];
    D --> F{Content for LLM};
    E --> F;
    B --> G[2. Load Prompts & Rubrics];
    G --> H[Prompt Manager];
    H --> I[Rubric Groups for Parallel Processing];
    B --> J[3. Static Analysis (Optional)];
    J --> K[Semgrep Analyzer];
    L[4. LLM Evaluation];
    subgraph Parallel LLM Calls
        direction LR
        I --> L;
        F --> L;
    end
    L --> M[Structured Rubric Feedback];
    K --> N[Semgrep Feedback];
    B --> O[5. Generate Output];
    M --> O;
    N --> O;
    O --> P[Output Manager];
    subgraph Output: 3 Formats
        P --> Q[Complete Markdown Report];
        P --> R[Structured JSON Report];
        P --> S[Friendly Student Summary];
    end
```

### 2.2. Design Patterns

-   **Facade Pattern (`AutoGrader`):** The `src/core/autograder.py` class serves as a facade, providing a simple, high-level interface (`process_assignments`) to the complex subsystem of processors, analyzers, and managers.
-   **Strategy Pattern (`LLMProvider`):** The `src/llm/base.py` defines an abstract `LLMProvider` interface. Concrete classes (`GeminiProvider`, `OpenAIProvider`, `OllamaProvider`) implement this interface. This allows the application to switch between different LLM backends by simply changing a configuration setting, without altering the core grading logic.
-   **Factory Pattern (`LLMFactory`):** The `src/llm/factory.py` class provides a centralized method (`create_provider`) for instantiating the correct LLM provider (strategy) based on the application's configuration.
-   **Singleton Pattern (`Settings`):** The `src/config/settings.py` class ensures that there is only one instance of the configuration settings throughout the application's lifecycle, providing a consistent and global point of access to all configuration parameters.

## 3. Module & File Breakdown

This section details the purpose of each key directory and file in the `src` directory.

-   **`main.py`**: The application's entry point. It handles command-line interface (CLI) presentation, initializes the `Settings` and `AutoGrader`, and kicks off the processing.

-   **`src/config/settings.py`**:
    -   **`Settings`**: A Singleton class that loads configuration from `config.env` and project-specific JSON files from `config/projects/`. It's the single source of truth for all configuration.

-   **`src/core/`**: Contains the main orchestration logic.
    -   **`autograder.py` (`AutoGrader`)**: The central coordinator. It manages the entire grading pipeline, from processing input files to orchestrating parallel LLM calls and saving the final output.
    -   **`output_manager.py` (`OutputManager`)**: Responsible for generating the final evaluation files in three distinct formats: a comprehensive Markdown report, a structured JSON file, and a student-friendly summary (which itself is generated via an LLM call).
    -   **`report_processor.py` (`ReportProcessor`)**: Handles non-code submissions (e.g., PDFs, images) when `submission_is` is set to `"report"`. It extracts and prepares these files for multimodal LLM analysis.

-   **`src/llm/`**: Manages all interactions with Large Language Models.
    -   **`base.py` (`LLMProvider`)**: The abstract base class that defines the contract for all LLM providers (e.g., `generate_response`, `count_tokens`).
    -   **`factory.py` (`LLMFactory`)**: Creates concrete LLM provider instances.
    -   **`gemini_provider.py`, `openai_provider.py`, `ollama_provider.py`**: Concrete implementations of the `LLMProvider` strategy for different services.

-   **`src/prompts/prompt_manager.py`**:
    -   **`PromptManager`**: A crucial module that dynamically builds prompts for the LLMs. It reads content from the assignment-specific `prompts/` directory, combines them, and structures the prompts for rubric evaluation, static analysis feedback, and the final friendly summary.

-   **`src/repomix/processor.py`**:
    -   **`RepomixProcessor`**: A wrapper around the `repomix` Node.js tool. It handles extracting student code, filtering files based on project configuration (`ignore_patterns`, `keep_patterns`, `max_file_size`), and consolidating the codebase into a single text block for the LLM. It also manages token limits by applying compression when needed.

-   **`src/semgrep/analyzer.py`**:
    -   **`SemgrepAnalyzer`**: A wrapper for the `semgrep` static analysis tool. It runs Semgrep with a project-specific ruleset, parses the JSON output, and formats the findings for evaluation by an LLM.

-   **`src/utils/logger.py`**:
    -   Provides a standardized logging setup for the entire application, ensuring consistent and useful log output for debugging.

## 4. Configuration Guide (The "Knobs")

The system is heavily driven by configuration. Understanding these "knobs" is key to adapting the grader.

### 4.1. Global Configuration (`config.env`)

This file controls the global behavior of the application.

| Variable             | Description                                                                                             | Example                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `LLM_PROVIDER`       | The LLM backend to use.                                                                                 | `gemini`, `openai`, `ollama`               |
| `GEMINI_API_KEY`     | API key for Google Gemini.                                                                              | `your_gemini_api_key_here`                 |
| `OPENAI_API_KEY`     | API key for OpenAI.                                                                                     | `your_openai_api_key_here`                 |
| `OLLAMA_BASE_URL`    | The base URL for a local Ollama server.                                                                 | `http://localhost:11434`                   |
| `OLLAMA_MODEL`       | The specific model to use from your local Ollama instance.                                              | `llama3.1:8b`, `deepseek-r1`               |
| `PROJECT_ASSIGNMENT` | **Crucial setting.** The name of the current assignment. This determines which subdirectories in `config/projects/` and `prompts/` are used. | `software_refractoring_assignment_1`       |
| `MAX_TOKENS`         | The token limit of the target LLM. Used by `RepomixProcessor` to decide if compression is needed.        | `128000`                                   |
| `USE_COMPRESSION`    | If `true`, `RepomixProcessor` will attempt to compress the codebase if it exceeds `MAX_TOKENS`.          | `true`                                     |
| `REMOVE_COMMENTS`    | If `true`, `RepomixProcessor` will strip comments from the code.                                        | `false`                                    |

### 4.2. Project-Specific Configuration (`config/projects/[project_name].json`)

This JSON file allows you to override global settings and define behavior for a specific assignment.

| Key                         | Description                                                                                             | Type          | Example                                 |
| --------------------------- | ------------------------------------------------------------------------------------------------------- | ------------- | --------------------------------------- |
| `max_file_size`             | Files larger than this (in bytes) will be ignored by `RepomixProcessor`.                                | `integer`     | `1000000` (1MB)                         |
| `ignore_patterns`           | A list of glob patterns for files/directories to exclude.                                               | `list[string]` | `["*.log", "node_modules/"]`             |
| `keep_patterns`             | If provided, *only* files matching these glob patterns will be included. Overrides ignore.                | `list[string]` | `["**/*.java", "README.md"]`              |
| `max_parallel_llm`          | The number of rubric criteria groups to process in parallel. More means faster but higher cost/load.   | `integer`     | `2`                                     |
| `enable_semgrep_analysis`   | Whether to run the `SemgrepAnalyzer` for this project.                                                  | `boolean`     | `true`                                  |
| `semgrep_rules_file`        | Path to the Semgrep YAML rules file to use for this project.                                            | `string`      | `config/semgrep/semgrep_test.yaml`      |
| `semgrep_timeout`           | Timeout in seconds for the Semgrep process.                                                             | `integer`     | `300`                                   |
| `prompt_has_img_pdf`        | Set to `true` if the `prompts/[project_name]/attachments` directory contains images/PDFs for the LLM. | `boolean`     | `true`                                  |
| `submission_is`             | The type of student submission. Use `"code"` for source code or `"report"` for PDF/image-based reports. | `string`      | `code`                                  |

### 4.3. Prompt Configuration (`prompts/[project_name]/`)

This directory contains the "soul" of the evaluation for a specific assignment.

-   **`assignment_details.txt`**: Contains the assignment description, objectives, and any context the LLM needs to understand the task. This is provided to the LLM in almost every call.
-   **`general_rubric.txt`**: High-level instructions on how to grade, what tone to use, and how to structure the feedback.
-   **`specific_rubric.json`**: **The core of the rubric.** A JSON array where each object defines a specific grading criterion, its max points, and the detailed prompt for evaluating it.
-   **`instruction_content.txt`**: Additional, specific instructions for the LLM about the evaluation process itself.
-   **`final_prompt.txt`**: The prompt used to generate the student-friendly summary from the detailed technical report.
-   **`static_instructions.txt`**: Instructions for the LLM on how to interpret and grade the findings from Semgrep.
-   **`attachments/`** (directory): Any images or PDF files that need to be sent to a multimodal LLM along with the prompts.

## 5. Setup & Usage

Follow these steps to get the development environment running.

1.  **Prerequisites**:
    -   Python 3.8+
    -   Node.js 18+ and npm

2.  **Clone Repository**:
    ```bash
    git clone https://github.com/cmusv-gradiatorx/qualitative_static_analysis.git
    cd qualitative_static_analysis
    ```

3.  **Install Dependencies**:
    ```bash
    # Install Python packages
    pip install -r requirements.txt

    # Install repomix globally via npx (handled on-the-fly) or install it
    npm install -g repomix
    ```

4.  **Configure Environment**:
    -   Copy `config.env.local` to `config.env`.
    -   Edit `config.env` to set your `LLM_PROVIDER` and corresponding API keys.
    -   Set the `PROJECT_ASSIGNMENT` to the assignment you want to grade.

5.  **Run the Application**:
    -   Place student submission ZIP files into the `input/` directory.
    -   Execute the main script:
        ```bash
        python main.py
        ```
    -   Check the `output/` directory for the three report formats. Logs are written to the `logs/` and `extra_logs/` directories.

## 6. How to Extend the System

### 6.1. How to Add a New Assignment

1.  **Create Project Config**: Create a new JSON file in `config/projects/`. For a new assignment `my_new_assignment`, create `config/projects/my_new_assignment.json`. Configure it as needed.
2.  **Create Prompts Directory**: Create a new directory `prompts/my_new_assignment/`.
3.  **Populate Prompts**: Inside the new directory, create all the required `.txt` and `.json` files (`assignment_details.txt`, `specific_rubric.json`, etc.).
4.  **Update Environment**: Change the `PROJECT_ASSIGNMENT` variable in `config.env` to `my_new_assignment`.
5.  **Run**: The autograder will now use your new configuration and prompts.

### 6.2. How to Add a New LLM Provider

1.  **Create Provider Class**: In the `src/llm/` directory, create a new file (e.g., `my_provider.py`). Inside, create a class `MyProvider` that inherits from `LLMProvider`.
2.  **Implement Abstract Methods**: You must implement all abstract methods from `LLMProvider`: `_validate_config`, `generate_response`, `count_tokens`, `get_max_tokens`, `supports_multimodal`, and `generate_response_with_attachments`.
3.  **Register in Factory**: Open `src/llm/factory.py`.
    -   Import your new provider: `from .my_provider import MyProvider`.
    -   Add it to the `_providers` dictionary: `'myprovider': MyProvider`.
4.  **Configure**: You can now set `LLM_PROVIDER=myprovider` in `config.env`.

## 7. Usage

### Step 1: Create a Project

A "Project" is a self-contained set of configurations and prompts for a specific assignment.

1.  **Add a Project Config**: Create your JSON file in `config/projects/`.
2.  **Add a Prompt Directory**: Create a corresponding directory in `prompts/`. For a project named `my_assignment`, create `prompts/my_assignment/`.
3.  **Populate Prompt Files**: Inside the new directory, create the necessary prompt files:
    *   `assignment_details.txt`: The full assignment description.
    *   `general_rubric.txt`: High-level grading instructions for the LLM.
    *   `specific_rubric.json`: A JSON array defining each grading criterion.
    *   `instruction_content.txt`: Extra instructions for the evaluation process.
    *   `final_prompt.txt`: The prompt for generating the friendly student summary.
    *   `static_instructions.txt`: (If using Semgrep) Instructions on how to interpret the analysis.

### Step 2: Prepare Submissions

Place the student assignment `.zip` files into the `input/` directory.

### Step 3: Run the Grader

Set the `PROJECT_ASSIGNMENT` in `config.env` to your project's name and run the application.

```bash
python main.py
```

### Step 4: Check the Output

The results will appear in the `output/` directory, organized into three subfolders:

-   `output/complete_markdown/`: Contains the full, detailed technical evaluation reports.
-   `output/json/`: Contains the structured JSON data for each evaluation.
-   `output/friendly/`: Contains the student-facing summaries.

## 8. Troubleshooting & Best Practices

### 8.1. Semgrep Specifics

-   **`.git` Directory is Required**: Semgrep's default behavior for ignoring files relies on `.gitignore`. Even if you aren't using the gitignore functionality, some underlying mechanisms work more reliably when a `.git` directory is present at the root of the scanned project. If a student submission doesn't have one, Semgrep might scan files you intend to ignore (like `node_modules`). The `RepomixProcessor`'s filtering helps mitigate this, but it's a good-to-know detail.
-   **Use the Semgrep Playground**: Before adding a new rule to a YAML file, **always test it** in the [Semgrep Playground](https://semgrep.dev/playground/). This will save you hours of debugging. You can paste a student's code and your rule to see if it triggers correctly.
-   **Semgrep Fails to Run**: Ensure `semgrep` is installed and available in your system's PATH. You can install it via `pip install semgrep`.

### 8.2. LLM and Prompt Engineering

-   **Bad JSON Output**: If the LLM frequently returns malformed JSON, try making the prompt more explicit. Use phrases like "CRITICAL: RESPOND ONLY IN VALID JSON FORMAT. Do not include any text before or after the JSON object." The current prompts do this, but may need reinforcement.
-   **Token Limits**: If even the compressed codebase exceeds token limits, you may need to:
    1.  Use a model with a larger context window.
    2.  Make your `ignore_patterns` or `keep_patterns` in the project config more aggressive to exclude more non-essential files.
    3.  Set `max_file_size` to a lower value to exclude large, potentially auto-generated files.
-   **Check `extra_logs/`**: This directory is your best friend for debugging LLM issues. It contains the exact prompt sent to the LLM and the raw response received. If the evaluation seems off, inspect these files first.

---
