"""
Code Execution Agent — hardened sandbox with resource limits.
FIX: working_dir now includes session_id for session isolation.
"""
import logging
import os
import subprocess
import sys
import tempfile
import time
import shutil
import concurrent.futures
from pathlib import Path
from typing import Optional

import httpx

from multimodal_ds.config import CODER_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT, OUTPUT_DIR
from multimodal_ds.memory.agent_memory import AgentMemory
from multimodal_ds.core.observability import agent_span, get_session_tracker

logger = logging.getLogger(__name__)

_CPU_SECONDS    = int(os.getenv("SANDBOX_CPU_SECONDS",  "60"))
_MEM_MB         = int(os.getenv("SANDBOX_MEM_MB",       "512"))
_STDOUT_CHARS   = int(os.getenv("SANDBOX_STDOUT_CHARS", "8000"))
_PROC_TIMEOUT_S = int(os.getenv("SANDBOX_TIMEOUT_S",    "300"))

SYSTEM_PROMPT = """You are a senior data scientist. You write precise, self‑contained Python.

MANDATORY RULES — follow every one, no exceptions:
1. First line of code: print(df.columns.tolist()) and print(df.shape)
2. Use ONLY column names confirmed by step 1 — never guess column names
3. Print descriptive stats: df.describe(), value_counts for every categorical column
4. ALWAYS split data: X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y)
       NEVER train and evaluate on the same data.
       For any model: print(classification_report(y_test, model.predict(X_test))),
       print(confusion_matrix(y_test, model.predict(X_test))),
       print('ROC-AUC:', roc_auc_score(y_test, model.predict_proba(X_test)[:,1])),
       print('feature importances:', dict(zip(feature_names, model.feature_importances_)))
5. ALWAYS set matplotlib backend as the ABSOLUTE FIRST THREE LINES of the entire script,
   before ANY other import including pandas, numpy, sklearn:
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   These 3 lines must appear at line 1, 2, 3 of the file. No exceptions.
   Never import matplotlib.pyplot before calling matplotlib.use('Agg').
6. Save ALL plots: plt.savefig('filename.png', dpi=100, bbox_inches='tight',
   facecolor='white', edgecolor='none')
   Then IMMEDIATELY call plt.close('all') after every savefig call.
   Never reuse a figure across multiple plots.
7. Never call plt.show() under any circumstances.
8. When creating subplot grids with plt.subplots(rows, cols):
   - Calculate rows and cols AFTER loading data, based on actual column count
   - Never hardcode a grid size before knowing the data shape
   - Always use: fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
   - Always call fig.tight_layout() before savefig
   - If a subplot axis is unused, call ax.set_visible(False) on it
9. Save trained models: joblib.dump(model, 'model.pkl')
10. End with a FINDINGS block: print('=== FINDINGS ===') then 3‑5 quantitative sentences
11. NEVER evaluate a model on training data. Always use held-out test set.
   If cross-validation: use cross_val_score with cv=5 on training data only.
12. Keep ALL string literals on a single line. Never split a string literal across lines using implicit continuation. For long titles use short versions: ax.set_title('H2: Products') not ax.set_title('H2: NumOfProducts > 2 has significantly higher churn rates than customers with 1 or 2 products')
13. NEVER import seaborn before setting matplotlib backend. Import order MUST be:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns  # only after the above

Output only valid Python code inside ```python ... ``` fences. No commentary outside the fences."""


def _sandbox_preexec() -> None:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS))
        mem_bytes = _MEM_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    except Exception:
        pass


class CodeExecutionAgent:
    AGENT_NAME = "code_execution_agent"

    def __init__(self, working_dir: Optional[str] = None, session_id: str = "default"):
        # FIX: include session_id in working_dir for session isolation
        base = Path(working_dir) if working_dir else Path(OUTPUT_DIR)
        self.working_dir = base / session_id
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.memory = AgentMemory()
        self._tracker = get_session_tracker(session_id)

    def _fix_filename_references(self, code: str, working_files: list[str]) -> str:
        """Detect and fix filename mismatches in generated code.

        The LLM often generates code with wrong filenames (e.g., 'Churn.csv' when
        the actual file is 'Churn_Modelling.csv'). This method:
        1. Finds file references in generated code (pd.read_csv, pd.read_excel, etc.)
        2. Checks if referenced files exist in working directory
        3. If not, finds the closest matching actual file and replaces the reference
        """
        if not working_files:
            return code

        actual_files = {Path(f).name.lower(): Path(f).name for f in working_files}
        import re

        # Patterns that reference data files in code
        file_ref_patterns = [
            r'pd\.read_(?:csv|excel|parquet|json)\s*\(\s*["\']([^"\']+)["\']',
            r'read_(?:csv|excel|parquet|json)\s*\(\s*["\']([^"\']+)["\']',
            r'open\s*\(\s*["\']([^"\']+\.(?:csv|json|txt))["\']',
        ]

        def find_closest_match(missing_name: str) -> str | None:
            """Find the most similar actual file using fuzzy matching."""
            import os
            missing_lower = missing_name.lower()
            # Exact match (case-insensitive)
            if missing_lower in actual_files:
                return actual_files[missing_lower]

            # Extract stem (filename without extension) for smarter matching
            missing_stem = os.path.splitext(missing_name)[0].lower()
            missing_ext = os.path.splitext(missing_name)[1].lower()

            # Try to find a match by comparing stems
            for actual_lower, actual_name in actual_files.items():
                actual_stem = os.path.splitext(actual_lower)[0]
                actual_ext = os.path.splitext(actual_lower)[1]

                # Skip if extensions don't match (could be different file types)
                if actual_ext != missing_ext:
                    continue

                # Check if stems share significant overlap:
                # 1. One stem is contained in the other (e.g., "Churn" in "Churn_Modelling")
                # 2. They share the first half of characters (e.g., "custom" vs "customer")
                if (missing_stem in actual_stem or actual_stem in missing_stem or
                    (len(missing_stem) > 3 and len(actual_stem) > 3 and
                     missing_stem[:len(missing_stem)//2] in actual_stem)):
                    return actual_name

            return None

        # Find all file references and try to fix them
        for pattern in file_ref_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for ref in matches:
                ref_lower = ref.lower()
                # Check if this exact reference exists
                if ref_lower not in actual_files:
                    # Try to find a close match
                    matched = find_closest_match(ref)
                    if matched:
                        logger.info(f"[CodeAgent] Fixing filename: '{ref}' -> '{matched}'")
                        # Replace all occurrences of this wrong filename
                        # Replace filename using a lambda to correctly insert matched filename
                        pattern = r'(["\'](?:\./)?){ref_escaped}(["\'])'.format(ref_escaped=re.escape(ref))
                        code = re.sub(
                            pattern,
                            lambda m, matched=matched: m.group(1) + matched + m.group(2),
                            code,
                            flags=re.IGNORECASE
                        )

        return code

    def execute_task(self, task: dict, data_context: str = "", file_paths: Optional[list] = None, max_retries: int = 2) -> dict:
        task_desc = task.get("description", str(task))
        task_name = task.get("name", "task")
        logger.info(f"[CodeAgent] Executing: {task_name}")

        with agent_span(self.AGENT_NAME, self.session_id, self._tracker) as span:
            span.set_metadata({"task_name": task_name})
            past_context = self._get_relevant_memory(task_desc)
            raw_code = self._generate_code(task_desc, data_context, past_context)
            code = self._extract_code(raw_code)
            if not code:
                # Retry code generation once with a simpler prompt before giving up.
                # The first attempt may fail if the model is still loading or
                # the context is too long. A simplified retry often succeeds.
                logger.warning("[CodeAgent] First code generation attempt returned empty — retrying with simplified prompt")
                simplified_desc = (
                    f"{task_desc}\n\n"
                    f"IMPORTANT: Respond with ONLY a Python code block inside ```python ... ``` fences. "
                    f"No explanation. No prose. Just the code."
                )
                raw_code = self._generate_code(simplified_desc, data_context[:500], "")
                code = self._extract_code(raw_code)
            if not code:
                logger.error(f"[CodeAgent] Code generation failed after retry for task: {task_desc[:100]}")
                return {
                    "success": False,
                    "error": "Code generation failed",
                    "code": "",
                    "output": "Code generation failed — LLM returned no parseable Python code.",
                    "files_created": [],
                }
            span.set_chars(input_chars=len(task_desc) + len(data_context), output_chars=len(code))
            result = self._execute_with_retry(code, task_desc, data_context, file_paths, max_retries)
            span.set_metadata({"task_name": task_name, "success": result["success"], "files_created": result["files_created"]})

        status_msg = "successfully" if result["success"] else "with errors"
        self.memory.store_analysis_step(
            step_name=task_name,
            result=f"Code executed {status_msg}.\nOutput: {result['output'][:500]}\nFiles: {result['files_created']}",
            session_id=self.session_id,
        )
        return result

    def execute(self, task_description: str, data_context: str = "", file_paths: Optional[list] = None, max_retries: int = 2) -> dict:
        rag_context = self._retrieve_rag_context(task_description)
        if rag_context:
            data_context = f"Relevant document context (from ChromaDB):\n{rag_context}\n\n" + data_context
        if file_paths:
            file_list = "\n".join(f"  - {Path(fp).name}" for fp in file_paths)
            data_context = f"Available data files (use exact names):\n{file_list}\n\n{data_context}"
        task = {"name": task_description[:80], "description": task_description}
        return self.execute_task(task=task, data_context=data_context, file_paths=file_paths, max_retries=max_retries)

    def _retrieve_rag_context(self, query: str, k: int = 4) -> str:
        try:
            results = self.memory.retrieve(query, n_results=k)
            if results:
                return "\n\n".join(r["content"] for r in results if r.get("content"))
        except Exception:
            pass
        return ""

    def _generate_code(self, task_desc: str, data_context: str, past_context: str) -> str:
        from multimodal_ds.core.llm_client import chat_with_fallback

        prompt = f"""Task: {task_desc}\nData Context:\n{data_context[:1500]}\nPrevious Context:\n{past_context[:500]}\nWorking directory: {self.working_dir}\nWrite Python code. Save all outputs to the current directory."""

        # Use unified LLM client - handles opencode/ and ollama/ prefixes automatically
        try:
            result = chat_with_fallback(
                primary_model=CODER_MODEL,
                fallback_model="ollama/qwen2.5:7b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=6000,
                temperature=0.1,
            )
            if result and not result.startswith("[Error:"):
                # Return raw LLM response; extraction will be performed later
                return result.strip()
            logger.warning(f"[CodeAgent] LLM returned error: {result}")
        except Exception as e:
            logger.error(f"[CodeAgent] Code generation failed: {e}")
        logger.warning(f"[CodeAgent] Returning empty code for task — LLM call failed or response unparseable")
        return ""

    def _execute_code(self, code: str, file_paths: Optional[list] = None):
        files_before = set(self.working_dir.glob("*"))
        script_path = None
        copied_files = []

        # Copy data files to working dir so code can find them locally
        # Use a dedicated temp subdir inside working_dir (same filesystem → fast rename)
        if file_paths:
            for fp in file_paths:
                src = Path(fp)
                if src.exists():
                    dst = self.working_dir / src.name
                    if not dst.exists():
                        try:
                            # Try hard-link first (instant, zero copy) — works when
                            # src and dst are on the same filesystem
                            try:
                                os.link(src, dst)
                            except (OSError, NotImplementedError):
                                # Cross-filesystem or unsupported — fall back to copy
                                shutil.copy2(src, dst)
                            copied_files.append(dst)
                        except Exception as e:
                            logger.warning(f"[CodeAgent] Failed to copy {src.name}: {e}")

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=self.working_dir, delete=False, encoding="utf-8") as f:
                f.write(code)
                script_path = Path(f.name)
            
            run_kwargs = {
                "args": [sys.executable, str(script_path)],
                "cwd": str(self.working_dir),
                "capture_output": True,
                "text": True,
                "timeout": _PROC_TIMEOUT_S
            }
            if sys.platform != "win32":
                run_kwargs["preexec_fn"] = _sandbox_preexec
            
            result = subprocess.run(**run_kwargs)
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            combined = stdout + (f"\n[stderr]:\n{stderr}" if stderr else "")
            
            if len(combined) > _STDOUT_CHARS:
                combined = combined[:_STDOUT_CHARS] + f"\n\n[OUTPUT TRUNCATED]"
            
            success = result.returncode == 0 or (result.returncode != 0 and _is_only_warnings(stderr))
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {_PROC_TIMEOUT_S}s", []
        except Exception as e:
            return False, f"Execution error: {e}", []
        finally:
            if script_path and script_path.exists():
                try: script_path.unlink()
                except Exception: pass
            # Cleanup copied data files to keep sandbox clean
            for cf in copied_files:
                try: cf.unlink()
                except Exception: pass

        files_after = set(self.working_dir.glob("*"))
        new_files = [f.name for f in (files_after - files_before) if f.is_file() and f.suffix != ".py"]
        return success, combined, new_files

    def _execute_with_retry(self, code: str, task_desc: str, data_context: str, file_paths: Optional[list], max_retries: int) -> dict:

        # Fix filename references in generated code before execution
        if file_paths:
            # Get list of actual files that will be in working dir after copy
            working_files = [str(Path(fp).name) for fp in file_paths]
            code = self._fix_filename_references(code, working_files)

        success, output, files = self._execute_code(code, file_paths)
        if success:
            return {
                "success": True,
                "code": code,
                "output": output,
                "files_created": files,
                "error": "",
                "retries_used": 0,
            }

        for attempt in range(max_retries):
            # Generate fix and execute concurrently on retry
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fix_future = ex.submit(self._generate_fix, code, output, task_desc)
                fix_code = fix_future.result(timeout=120)

            if fix_code:
                # Also fix filename references in the fixed code
                if file_paths:
                    working_files = [str(Path(fp).name) for fp in file_paths]
                    fix_code = self._fix_filename_references(fix_code, working_files)

                success, output, files = self._execute_code(fix_code, file_paths)
                if success:
                    return {
                        "success": True,
                        "code": fix_code,
                        "output": output,
                        "files_created": files,
                        "error": "",
                        "retries_used": attempt + 1,
                    }
                code = fix_code

        return {
            "success": False,
            "code": code,
            "output": output,
            "files_created": files,
            "error": output,
            "retries_used": max_retries,
        }

    def _generate_fix(self, failed_code: str, error_output: str, task_desc: str) -> str:
        from multimodal_ds.core.llm_client import chat_with_fallback

        prompt = f"""Fix this Python code that failed.\nTask: {task_desc}\nFailed code:\n```python\n{failed_code[-2000:]}\n```\nError:\n{error_output[:1000]}\nProvide ONE complete fixed Python script in ```python ... ``` fences."""

        try:
            result = chat_with_fallback(
                primary_model=CODER_MODEL,
                fallback_model="ollama/qwen2.5:7b",
                messages=[
                    {"role": "system", "content": "Fix Python code. Output only the fixed code in ```python``` fences."},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=4000,
                temperature=0.1,
            )
            if result and not result.startswith("[Error:"):
                return self._extract_code(result)
        except Exception:
            pass
        return ""

    def _extract_code(self, text: str) -> str:
        import re

        if not text or not text.strip():
            return ""

        # Strip <think>...</think> reasoning blocks (qwen3, deepseek-r1)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Try ```python ... ``` fence first (allow optional spaces, optional space after backticks, and Windows line endings)
        m = re.search(r'```\s*python\s*\r?\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            if code:
                return code

        # Try any ``` ... ``` fence (any language tag or none)
        m = re.search(r'```.*?\r?\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            if code and any(
                kw in code
                for kw in ('import ', 'from ', 'def ', 'class ', '#', 'pd.', 'df', 'print(')
            ):
                return code

        # Last resort: raw text that looks like Python code. Never return fenced text
        # by filtering out lines that start with markdown code block fences (```).
        cleaned_lines = []
        for line in text.splitlines():
            stripped_line = line.strip()
            if stripped_line.startswith("```"):
                continue
            cleaned_lines.append(line)
        cleaned_text = "\n".join(cleaned_lines).strip()

        if cleaned_text:
            first_line = cleaned_text.split('\n')[0].strip()
            if any(first_line.startswith(kw) for kw in (
                'import ', 'from ', 'def ', 'class ', '#', 'pd.', 'df', 'print('
            )):
                return cleaned_text

        logger.warning("[CodeAgent] Could not extract Python code from LLM response")
        logger.debug(f"[CodeAgent] Raw response (first 300 chars): {text[:300]!r}")
        return ""

    def _get_relevant_memory(self, query: str) -> str:
        memories = self.memory.retrieve(query, n_results=3)
        if not memories:
            return ""
        return "\n".join(m["content"][:200] for m in memories)
