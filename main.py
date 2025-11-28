# fastapi_backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import subprocess
import sys
import json
from complexity_analyzer import analyze_code_complexity


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "https://yourdomain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TraceRequest(BaseModel):
    code: str
    language: str
    stdin: str = ""
    timeout: int = 10


class TraceResponse(BaseModel):
    trace: list
    error: str | None = None
    metadata: dict = {}


class ComplexityRequest(BaseModel):
    code: str = Field(
        ..., description="Python code to analyze", min_length=1, max_length=100000
    )
    language: str = Field(
        default="python",
        description="Programming language (currently only python supported)",
    )


class ComplexityResponse(BaseModel):
    success: bool
    complexity_data: dict = {}
    visualization_data: dict = {}
    error: str | None = None
    warnings: list = []


@app.post("/visualize/trace", response_model=TraceResponse)
async def visualize_trace(req: TraceRequest):
    """Execute code with tracing and return execution trace JSON (Python Tutor format)."""
    if req.language == "python":
        return await trace_python(req.code, req.timeout)
    elif req.language == "cpp":
        return await trace_cpp(req.code, req.timeout)
    else:
        raise HTTPException(status_code=400, detail="Unsupported language")


async def trace_python(code: str, timeout: int):
    """
    Run Python code with sys.settrace() and return trace in Python Tutor format.
    Uses heap references, proper frame stacks, and return values.
    """

    tracer_code = f'''
import sys
import json
import inspect
import ast

# Use a safe literal for the user code
SOURCE = {json.dumps(code)}
_ast_tree = ast.parse(SOURCE)
_line_to_ast_info = {{}}
_source_lines = SOURCE.splitlines()

trace = []
stdout_capture = []
heap = {{}}  # Global object heap with IDs
heap_counter = [1]  # Counter for heap object IDs
obj_to_heap_id = {{}}  # Map: id(obj) -> heap_id (stable IDs)
call_stack = []  # Stack of (frame, frame_id) tuples
frame_id_counter = [0]

_INTERNAL_NAMES = {{
    "sys", "json", "inspect", "ast",
    "SOURCE", "_ast_tree", "_source_lines",
    "trace", "stdout_capture",
    "heap", "heap_counter", "obj_to_heap_id",
    "call_stack", "frame_id_counter",
    "_line_to_ast_info",
    "_build_ast_explanations",
    "_funcname_to_defline", "_classname_to_defline",
    "_already_explained_defs",
    "_attach_ast_eval",
    "_INTERNAL_NAMES", "_encode_object", "_clean_dict",
    "_build_stack_to_render", "trace_line",
    "CaptureStdout", "original_stdout", "_get_heap_id"
}}

def _build_ast_explanations():
    for _node in ast.walk(_ast_tree):
        _lineno = getattr(_node, "lineno", None)
        if _lineno is None:
            continue

        if _lineno - 1 < 0 or _lineno - 1 >= len(_source_lines):
            continue

        line_text = _source_lines[_lineno - 1].rstrip("\\n")

        # Class definition
        if isinstance(_node, ast.ClassDef):
            _line_to_ast_info[_lineno] = {{
                "actual_statement": line_text,
                "explanation": [
                    {{"type": "class_definition", "data": {{"name": _node.name}}}}
                ],
            }}

        # Function definition (top-level or method)
        elif isinstance(_node, ast.FunctionDef):
            _line_to_ast_info[_lineno] = {{
                "actual_statement": line_text,
                "explanation": [
                    {{
                        "type": "function_definition",
                        "data": {{"name": _node.name, "args": [a.arg for a in _node.args.args]}},
                    }}
                ],
            }}

        # Simple assignment: x = ...
        elif isinstance(_node, ast.Assign) and isinstance(_node.targets[0], ast.Name):
            target = _node.targets[0].id
            _line_to_ast_info[_lineno] = {{
                "actual_statement": line_text,
                "explanation": [
                    {{
                        "type": "variable_assignment",
                        "data": {{"variable": target}},
                    }}
                ],
            }}

        # Attribute assignment: self.data = ...
        elif isinstance(_node, ast.Assign) and isinstance(_node.targets[0], ast.Attribute):
            attr = _node.targets[0].attr
            obj = _node.targets[0].value.id if isinstance(_node.targets[0].value, ast.Name) else None
            _line_to_ast_info[_lineno] = {{
                "actual_statement": line_text,
                "explanation": [
                    {{
                        "type": "attribute_assignment",
                        "data": {{"object": obj, "attribute": attr}},
                    }}
                ],
            }}

        # Function call expression: foo(...), print(...)
        elif isinstance(_node, ast.Call) and hasattr(_node.func, "id"):
            fname = _node.func.id
            _line_to_ast_info.setdefault(_lineno, {{
                "actual_statement": line_text,
                "explanation": [
                    {{"type": "function_call", "data": {{"name": fname}}}}
                ],
            }})

_build_ast_explanations()

# Definition-line maps for functions and classes
_funcname_to_defline = {{}}
_classname_to_defline = {{}}
_already_explained_defs = set()

for _node in ast.walk(_ast_tree):
    _lineno = getattr(_node, "lineno", None)
    if _lineno is None:
        continue
    if isinstance(_node, ast.FunctionDef):
        _funcname_to_defline[_node.name] = _lineno
    elif isinstance(_node, ast.ClassDef):
        _classname_to_defline[_node.name] = _lineno

def _get_heap_id(obj):
    """Return a stable heap id for an object."""
    oid = id(obj)
    if oid in obj_to_heap_id:
        return obj_to_heap_id[oid]
    hid = heap_counter[0]
    heap_counter[0] += 1
    obj_to_heap_id[oid] = hid
    return hid

def _encode_object(obj, visited=None):
    """
    Encode objects in a Python Tutor-style heap:
    - primitives as-is
    - user functions: ["REF", hid] with heap[hid] = ["FUNCTION", name(sig), None]
    - classes: ["REF", hid] with heap[hid] = ["CLASS", cls_name, [], ["method", ["REF", mid]], ...]
    - instances: ["REF", hid] with heap[hid] = ["INSTANCE", cls_name, ["attr", <val>], ...]
    - lists: ["REF", hid] with heap[hid] = ["LIST", [...encoded...]]
    - dicts: ["REF", hid] with heap[hid] = ["DICT", {{k: encoded_v}}]
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)

    # Primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # Cycle detection
    if obj_id in visited:
        return None
    visited.add(obj_id)

    # Functions / callables
    if callable(obj) and hasattr(obj, "__name__") and not inspect.isclass(obj):
        hid = _get_heap_id(obj)
        key = str(hid)
        if key not in heap:
            try:
                sig = inspect.signature(obj)
                func_name = f"{{obj.__name__}}{{sig}}"
            except Exception:
                func_name = obj.__name__ + "()"
            heap[key] = ["FUNCTION", func_name, None]
        return ["REF", hid]

    # Classes
    if inspect.isclass(obj):
        hid = _get_heap_id(obj)
        key = str(hid)
        if key not in heap:
            methods = []
            for name, val in obj.__dict__.items():
                if callable(val):
                    mid = _get_heap_id(val)
                    mkey = str(mid)
                    if mkey not in heap:
                        _encode_object(val, visited)
                    methods.append([name, ["REF", mid]])
            heap[key] = ["CLASS", obj.__name__, []] + methods
        return ["REF", hid]

    # Instances (user objects with __dict__)
    if hasattr(obj, "__dict__") and not inspect.isclass(obj):
        hid = _get_heap_id(obj)
        key = str(hid)
        if key not in heap:
            attrs = []
            for name, val in obj.__dict__.items():
                attrs.append([name, _encode_object(val, visited)])
            heap[key] = ["INSTANCE", obj.__class__.__name__] + attrs
        return ["REF", hid]

    # Lists
    if isinstance(obj, list):
        hid = _get_heap_id(obj)
        key = str(hid)
        # ALWAYS refresh the snapshot so mutations show up
        encoded_list = [_encode_object(item, visited) for item in obj[:100]]
        heap[key] = ["LIST", encoded_list]
        return ["REF", hid]

    # Dicts
    if isinstance(obj, dict):
        hid = _get_heap_id(obj)
        key = str(hid)
        encoded_dict = {{}}
        for k, v in list(obj.items())[:100]:
            encoded_dict[k] = _encode_object(v, visited)
        heap[key] = ["DICT", encoded_dict]
        return ["REF", hid]

    # Fallback: repr string
    try:
        return repr(obj)[:100]
    except Exception:
        return "<unrepresentable>"

def _clean_dict(d):
    """Filter dict and encode objects, skipping internal names and modules."""
    clean = {{}}
    try:
        for k, v in d.items():
            if k in _INTERNAL_NAMES or k.startswith("_"):
                continue
            if str(type(v)).startswith("<class 'module"):
                continue
            try:
                clean[k] = _encode_object(v)
            except Exception:
                clean[k] = "<unrepresentable>"
    except Exception:
        pass
    return clean

def _build_stack_to_render(current_frame):
    """Build stack_to_render in Python Tutor format."""
    stack_frames = []
    for frame, fid in call_stack:
        if frame.f_code.co_name == "<module>":
            continue
        is_highlighted = (frame == current_frame)
        locals_dict = _clean_dict(frame.f_locals)
        stack_frames.append({{
            "func_name": frame.f_code.co_name,
            "is_parent": False,
            "frame_id": fid,
            "parent_frame_id_list": [],
            "encoded_locals": locals_dict,
            "ordered_varnames": list(locals_dict.keys()),
            "is_zombie": False,
            "is_highlighted": is_highlighted,
            "unique_hash": f"{{frame.f_code.co_name}}_f{{fid}}"
        }})
    return stack_frames

def _attach_ast_eval(step, lineno):
    info = _line_to_ast_info.get(lineno)
    step["ast_eval"] = {{
        "current_step": None,
        "actual_statement": info["actual_statement"] if info else None,
        "output": None,
        "explanation": info["explanation"] if info else [],
        "start_line": lineno,
        "end_line": None,
        "skip_line": False,
    }}
    step["dsa_steps"] = None

def trace_line(frame, event, arg):
    global heap, call_stack

    filename = frame.f_code.co_filename
    if "<string>" not in filename:
        return trace_line

    func_name = frame.f_code.co_name

    # Hide class body frames (class Task: body)
    if event in ("call", "line", "return") and func_name in _classname_to_defline:
        return trace_line
        
    if func_name in ("write", "flush"):
        return trace_line

    if event == "call":
        if func_name not in ("write", "flush", "<module>"):
            frame_id_counter[0] += 1
            call_stack.append((frame, frame_id_counter[0]))

        globals_dict = _clean_dict(frame.f_globals)
        # force-encode globals so heap gets CLASS/FUNCTION entries
        for name, value in frame.f_globals.items():
            if name in _INTERNAL_NAMES or name.startswith("_"):
                continue
            try:
                _encode_object(value)
            except Exception:
                pass

        stack_to_render = _build_stack_to_render(frame)

        step = {{
            "line": frame.f_lineno,
            "event": "call",
            "func_name": func_name,
            "globals": globals_dict,
            "ordered_globals": list(globals_dict.keys()),
            "stack_to_render": stack_to_render,
            "heap": dict(heap),
            "stdout": "".join(stdout_capture)
        }}
        _attach_ast_eval(step, frame.f_lineno)
        trace.append(step)

    elif event == "return":
        globals_dict = _clean_dict(frame.f_globals)
        for name, value in frame.f_globals.items():
            if name in _INTERNAL_NAMES or name.startswith("_"):
                continue
            try:
                _encode_object(value)
            except Exception:
                pass

        stack_to_render = _build_stack_to_render(frame)

        for sf in stack_to_render:
            if sf["func_name"] == frame.f_code.co_name and sf["is_highlighted"]:
                sf["encoded_locals"]["__return__"] = _encode_object(arg)
                sf["ordered_varnames"].append("__return__")

        step = {{
            "line": frame.f_lineno,
            "event": "return",
            "func_name": frame.f_code.co_name,
            "globals": globals_dict,
            "ordered_globals": list(globals_dict.keys()),
            "stack_to_render": stack_to_render,
            "heap": dict(heap),
            "stdout": "".join(stdout_capture)
        }}
        _attach_ast_eval(step, frame.f_lineno)
        trace.append(step)

        if call_stack and call_stack[-1][0] == frame:
            call_stack.pop()

    elif event == "line":
        if func_name == "write":
            return trace_line

        globals_dict = _clean_dict(frame.f_globals)
        for name, value in frame.f_globals.items():
            if name in _INTERNAL_NAMES or name.startswith("_"):
                continue
            try:
                _encode_object(value)
            except Exception:
                pass

        def_line_to_use = None
        if frame.f_code.co_name == "<module>":
            for name in globals_dict.keys():
                if name in _funcname_to_defline and name not in _already_explained_defs:
                    def_line_to_use = _funcname_to_defline[name]
                    _already_explained_defs.add(name)
                    break
                if name in _classname_to_defline and name not in _already_explained_defs:
                    def_line_to_use = _classname_to_defline[name]
                    _already_explained_defs.add(name)
                    break

        stack_to_render = _build_stack_to_render(frame)

        step = {{
            "line": frame.f_lineno,
            "event": "step_line",
            "func_name": func_name,
            "globals": globals_dict,
            "ordered_globals": list(globals_dict.keys()),
            "stack_to_render": stack_to_render,
            "heap": dict(heap),
            "stdout": "".join(stdout_capture)
        }}
        if def_line_to_use is not None:
            _attach_ast_eval(step, def_line_to_use)
        else:
            _attach_ast_eval(step, frame.f_lineno)
        trace.append(step)

    return trace_line

class CaptureStdout:
    def write(self, text):
        stdout_capture.append(text)
        return len(text)

    def flush(self):
        pass

original_stdout = sys.stdout

try:
    sys.stdout = CaptureStdout()
    sys.settrace(trace_line)

    frame_id_counter[0] = 0

    # Execute the same SOURCE we parsed
    exec(SOURCE, globals(), globals())

except Exception as e:
    trace.append({{
        "line": 0,
        "event": "error",
        "error": type(e).__name__,
        "message": str(e),
        "stdout": "".join(stdout_capture)
    }})

finally:
    sys.settrace(None)
    sys.stdout = original_stdout

result = {{"trace": trace, "metadata": {{"totalSteps": len(trace)}}}}
print(json.dumps(result))
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", tracer_code],
            capture_output=True,
            text=True,
            timeout=min(timeout, 30),
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if stderr and ("Traceback" in stderr or "Error" in stderr):
            return TraceResponse(
                trace=[],
                error=f"Runtime Error: {stderr[:300]}",
            )

        if not stdout:
            return TraceResponse(
                trace=[],
                error="No output from tracer",
            )

        try:
            output = json.loads(stdout)
            return TraceResponse(
                trace=output.get("trace", []),
                metadata=output.get("metadata", {}),
            )
        except json.JSONDecodeError as e:
            return TraceResponse(
                trace=[],
                error=f"JSON Parse Error: {str(e)[:100]}",
            )

    except subprocess.TimeoutExpired:
        return TraceResponse(
            trace=[],
            error="Execution timeout (infinite loop?)",
        )
    except Exception as e:
        return TraceResponse(
            trace=[],
            error=f"Backend Error: {str(e)[:200]}",
        )


async def trace_cpp(code: str, timeout: int):
    """Placeholder for C++ support."""
    return TraceResponse(
        trace=[],
        error="C++ support coming soon",
    )


@app.get("/visualize/languages")
async def get_languages():
    return {
        "languages": [
            {"value": "python", "label": "Python 3.x"},
            {"value": "cpp", "label": "C++"},
        ]
    }


@app.post("/analyze-complexity", response_model=ComplexityResponse)
async def analyze_complexity(req: ComplexityRequest):
    """
    Analyze code complexity and return detailed metrics with visualization data.

    Handles edge cases:
    - Empty or whitespace-only code
    - Syntax errors
    - Very large code files
    - Unsupported languages
    - Invalid characters
    """
    warnings = []

    try:
        # Edge case: Validate language
        if req.language.lower() != "python":
            return ComplexityResponse(
                success=False,
                error=f"Unsupported language: {req.language}. Currently only Python is supported.",
                warnings=warnings,
            )

        # Edge case: Empty or whitespace-only code
        if not req.code or not req.code.strip():
            return ComplexityResponse(
                success=False,
                error="Code cannot be empty or contain only whitespace",
                warnings=warnings,
            )

        # Edge case: Code too large (performance protection)
        if len(req.code) > 100000:
            warnings.append("Code is very large, analysis may take longer")

        # Edge case: Check for common issues
        if req.code.count("\n") < 1:
            warnings.append("Code appears to be a single line, results may be limited")

        # Perform complexity analysis
        analysis_result = analyze_code_complexity(req.code)

        # Edge case: Check if analysis returned an error
        if "error" in analysis_result and analysis_result["error"]:
            return ComplexityResponse(
                success=False,
                complexity_data=analysis_result,
                error=analysis_result["error"],
                warnings=warnings,
            )

        # Generate visualization data based on detected complexity
        visualization_data = _generate_visualization_data(analysis_result)

        # Add helpful warnings based on analysis
        if analysis_result.get("timeComplexity", {}).get("notation") == "O(2^(2^n))":
            warnings.append(
                "⚠️ CRITICAL: Double exponential complexity detected - code may be extremely slow"
            )
        elif "2^n" in analysis_result.get("timeComplexity", {}).get("notation", ""):
            warnings.append(
                "⚠️ WARNING: Exponential complexity detected - consider optimization"
            )
        elif "n³" in analysis_result.get("timeComplexity", {}).get("notation", ""):
            warnings.append(
                "⚠️ WARNING: Cubic complexity detected - may be slow for large inputs"
            )

        # Check for low confidence
        if analysis_result.get("timeComplexity", {}).get("confidence") == "low":
            warnings.append(
                "Complexity analysis has low confidence - manual review recommended"
            )

        return ComplexityResponse(
            success=True,
            complexity_data=analysis_result,
            visualization_data=visualization_data,
            warnings=warnings,
        )

    except RecursionError:
        return ComplexityResponse(
            success=False,
            error="Code analysis exceeded recursion limit - code may be too complex or deeply nested",
            warnings=warnings,
        )
    except MemoryError:
        return ComplexityResponse(
            success=False,
            error="Code analysis exhausted available memory - code file may be too large",
            warnings=warnings,
        )
    except Exception as e:
        return ComplexityResponse(
            success=False,
            error=f"Unexpected error during analysis: {type(e).__name__}: {str(e)[:200]}",
            warnings=warnings,
        )


def _generate_visualization_data(analysis_result: dict) -> dict:
    """
    Generate data points for complexity visualization charts.
    Creates sample data points showing growth rate for detected complexity.
    """
    time_complexity = analysis_result.get("timeComplexity", {}).get("notation", "O(1)")

    # Generate sample input sizes
    n_values = [1, 10, 20, 30, 40, 50]

    # Map complexity notations to growth functions
    complexity_map = {
        "O(1)": lambda n: 1,
        "O(log n)": lambda n: max(1, int(n**0.5)),  # approximation
        "O(n)": lambda n: n,
        "O(n log n)": lambda n: int(n * (n**0.5)),  # approximation
        "O(n²)": lambda n: n**2,
        "O(n³)": lambda n: n**3,
        "O(2^n)": lambda n: min(2 ** min(n, 20), 1000000),  # cap exponential
        "O(n!)": lambda n: min(_factorial_approx(n), 1000000),
        "O(2^(2^n))": lambda n: min(2 ** min(2 ** min(n, 4), 20), 1000000),
    }

    # Find the best matching complexity
    detected_func = complexity_map.get("O(n)")  # default
    for key in complexity_map:
        if key in time_complexity:
            detected_func = complexity_map[key]
            break

    # Generate data points
    data_points = [detected_func(n) for n in n_values]

    # Also include common complexities for comparison
    comparison_data = {
        "O(1)": [complexity_map["O(1)"](n) for n in n_values],
        "O(log n)": [complexity_map["O(log n)"](n) for n in n_values],
        "O(n)": [complexity_map["O(n)"](n) for n in n_values],
        "O(n log n)": [complexity_map["O(n log n)"](n) for n in n_values],
        "O(n²)": [complexity_map["O(n²)"](n) for n in n_values],
    }

    return {
        "detected_complexity": time_complexity,
        "data_points": data_points,
        "input_sizes": n_values,
        "comparison_data": comparison_data,
        "space_complexity": analysis_result.get("spaceComplexity", {}).get(
            "notation", "O(1)"
        ),
    }


def _factorial_approx(n: int) -> int:
    """Approximate factorial for visualization (capped to prevent overflow)"""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, min(n + 1, 21)):
        result *= i
        if result > 1000000:
            return 1000000
    return result


@app.get("/get-complexity-data")
async def get_complexity_data():
    """Legacy endpoint - returns sample complexity data for visualization."""
    sample_data = {
        "O(1)": [1, 1, 1, 1, 1],
        "O(log n)": [1, 2, 3, 4, 5],
        "O(n)": [1, 2, 3, 4, 5],
        "O(n log n)": [1, 3, 5, 7, 9],
        "O(n^2)": [1, 4, 9, 16, 25],
    }
    return sample_data


@app.post("/visualize/validate")
async def validate_code(req: TraceRequest):
    if req.language == "python":
        try:
            compile(req.code, "<string>", "exec")
            return {"valid": True}
        except SyntaxError as e:
            return {"valid": False, "errors": [f"Line {e.lineno}: {e.msg}"]}
    return {"valid": True}


@app.get("/health")
async def health():
    return {"status": "ok"}
