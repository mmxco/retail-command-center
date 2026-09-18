"""
tests/test_architecture_governance.py
Automated architectural governance and code quality verification test suite (Milestone 5 / R1, R2, R5, AGENTS.md).
Verifies:
1. Pure domain isolation of core/ (zero Streamlit imports in core/rim_engine.py, core/store_clustering.py, core/value_engine.py)
2. Backward-compatible root module re-exports
3. Strict AST cyclomatic complexity (< 10) for all top-level tab renderers & retail KPIs
4. Strict presentation function line length (<= 80 LOC) across presentation files
5. app.py coordinator line length strictly < 100 LOC
6. Deprecated Streamlit parameter scan: exactly 0 occurrences of use_container_width across all Python files
"""

import ast
import unittest
from pathlib import Path


class TestArchitectureGovernance(unittest.TestCase):
    """Test suite enforcing clean architecture, complexity limits, and API standards."""

    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.core_dir = cls.root_dir / "core"
        cls.ui_dir = cls.root_dir / "ui"
        cls.app_py = cls.root_dir / "app.py"

    # --------------------------------------------------------------------------
    # 1. PURE DOMAIN ISOLATION OF core/ (ZERO STREAMLIT IMPORTS)
    # --------------------------------------------------------------------------
    def test_core_pure_domain_isolation(self):
        """Verify zero occurrences of Streamlit imports across all modules in core/."""
        self.assertTrue(self.core_dir.exists(), "core/ directory must exist")
        core_files = list(self.core_dir.glob("*.py"))
        self.assertGreaterEqual(len(core_files), 3, "core/ must contain rim_engine, store_clustering, value_engine")

        violations = []
        for file_path in core_files:
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "streamlit" or alias.name.startswith("streamlit."):
                            violations.append((file_path.name, node.lineno, f"import {alias.name}"))
                elif isinstance(node, ast.ImportFrom):
                    if node.module and ("streamlit" in node.module):
                        violations.append((file_path.name, node.lineno, f"from {node.module} import ..."))

        self.assertEqual(
            violations,
            [],
            f"Pure domain violation! Streamlit imports found in core/: {violations}"
        )

    # --------------------------------------------------------------------------
    # 2. ROOT MODULE BACKWARD-COMPATIBLE RE-EXPORTS
    # --------------------------------------------------------------------------
    def test_root_module_backward_compatible_reexports(self):
        """Verify all legacy root modules exist and cleanly re-export their public API symbols."""
        expected_exports = {
            "rim_engine": ["RIMEngine", "compute_period_rim"],
            "store_clustering": [
                "ARCHETYPE_LABELS", "FEATURE_COLS", "StoreBenchmarkKNN",
                "load_and_aggregate_store_data", "load_pipeline", "save_pipeline",
                "train_clustering_pipeline"
            ],
            "value_calculator": [
                "ValueCalculatorInputs", "ValueRealizationOutput", "calculate_value_realization",
                "create_value_waterfall_chart", "create_3year_horizon_chart",
                "generate_heuristic_business_case", "generate_executive_business_case",
                "render_value_calculator_tab"
            ],
            "cfo_advisor": [
                "RIMDashboardState", "build_cfo_reasoning_prompt",
                "generate_heuristic_cfo_brief", "generate_cfo_risk_brief",
                "normalize_cfo_markdown"
            ],
            "tab_discovery": [
                "PRESET_PROFILES", "PRESET_DOSSIERS", "build_generic_prospect_profile",
                "render_tab_discovery"
            ],
            "tab_autogen": [
                "SCENARIOS", "CONSENSUS_TAG", "generate_capped_resolution",
                "render_tab_autogen"
            ],
            "tab_rim_knn": [
                "load_or_train_clustering_pipeline", "load_apparel_rim_dataset",
                "project_forward_12_periods", "generate_ai_cfo_brief_text",
                "render_tab_rim_knn"
            ],
            "app": ["render_tab_header", "render_retail_kpis", "main"],
        }

        missing_exports = []
        for mod_name, symbols in expected_exports.items():
            try:
                mod = __import__(mod_name)
            except Exception as e:
                missing_exports.append(f"Failed to import root module {mod_name}: {e}")
                continue

            for sym in symbols:
                if not hasattr(mod, sym):
                    missing_exports.append(f"Module {mod_name} missing re-export of '{sym}'")

        self.assertEqual(
            missing_exports,
            [],
            f"Backward-compatible re-export failures: {missing_exports}"
        )

    # --------------------------------------------------------------------------
    # 3. AST CYCLOMATIC COMPLEXITY STRICTLY < 10
    # --------------------------------------------------------------------------
    @staticmethod
    def _compute_ast_cyclomatic_complexity(fn_node: ast.FunctionDef) -> int:
        """Computes McCabe cyclomatic complexity: 1 + decision points."""
        complexity = 1
        for node in ast.walk(fn_node):
            if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.Assert, ast.Match)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

    def test_ast_cyclomatic_complexity_strict_limits(self):
        """Verify cyclomatic complexity is strictly < 10 for all top-level tab renderers & retail KPIs."""
        target_functions = [
            ("ui/tab_autogen.py", "render_tab_autogen"),
            ("ui/tab_discovery.py", "render_tab_discovery"),
            ("ui/tab_rim_knn.py", "render_tab_rim_knn"),
            ("ui/value_calculator.py", "render_value_calculator_tab"),
            ("app.py", "render_retail_kpis"),
        ]

        violations = []
        for rel_path, fn_name in target_functions:
            file_path = self.root_dir / rel_path
            # Fallback to root if UI module not yet moved
            if not file_path.exists():
                file_path = self.root_dir / Path(rel_path).name

            self.assertTrue(file_path.exists(), f"Target file not found: {file_path}")
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

            found = False
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == fn_name:
                    found = True
                    cc = self._compute_ast_cyclomatic_complexity(node)
                    if cc >= 10:
                        violations.append(f"{rel_path}::{fn_name} CC={cc} (must be strictly < 10)")
                    break

            self.assertTrue(found, f"Function '{fn_name}' not found in {file_path}")

        self.assertEqual(
            violations,
            [],
            f"Cyclomatic complexity violations (Requirement R2): {violations}"
        )

    # --------------------------------------------------------------------------
    # 4. PRESENTATION FUNCTIONS LENGTH STRICTLY <= 80 LOC
    # --------------------------------------------------------------------------
    def test_presentation_functions_length_limits(self):
        """Verify no single presentation function in UI modules or app.py exceeds 80 lines of code."""
        presentation_files = [
            self.app_py,
            self.ui_dir / "tab_discovery.py",
            self.ui_dir / "tab_autogen.py",
            self.ui_dir / "tab_rim_knn.py",
            self.ui_dir / "value_calculator.py",
        ]

        violations = []
        for file_path in presentation_files:
            if not file_path.exists():
                fallback = self.root_dir / file_path.name
                if fallback.exists():
                    file_path = fallback
                else:
                    continue

            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    loc = node.end_lineno - node.lineno + 1
                    if loc > 80:
                        violations.append(
                            f"{file_path.name}::{node.name} (lines {node.lineno}–{node.end_lineno}) is {loc} LOC (limit <= 80)"
                        )

        self.assertEqual(
            violations,
            [],
            f"Presentation function line length violations (Requirement R2): {violations}"
        )

    # --------------------------------------------------------------------------
    # 5. COORDINATOR app.py LOC STRICTLY < 100 LOC
    # --------------------------------------------------------------------------
    def test_app_coordinator_loc_limit(self):
        """Verify coordinator app.py has strictly < 100 lines of code."""
        self.assertTrue(self.app_py.exists(), "app.py must exist")
        lines = self.app_py.read_text(encoding="utf-8").splitlines()
        loc = len(lines)
        self.assertLess(
            loc,
            100,
            f"Coordinator app.py has {loc} lines of code, exceeding the limit of < 100 LOC (Requirement R1)"
        )

    # --------------------------------------------------------------------------
    # 6. DEPRECATED PARAMETER SCAN (ZERO OCCURRENCES OF use_container_width)
    def test_deprecated_parameter_scan_zero_use_container_width(self):
        """Verify zero occurrences of deprecated use_container_width across all workspace Python files."""
        violations = []
        this_file = Path(__file__).resolve()
        for py_file in self.root_dir.rglob("*.py"):
            # Exclude this test file itself, virtual environment, agents, git, and cache directories
            if py_file.resolve() == this_file:
                continue
            if any(part in py_file.parts for part in [".venv", ".git", ".agents", "__pycache__"]):
                continue

            with open(py_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    if "use_container_width" in line:
                        violations.append((str(py_file.relative_to(self.root_dir)), idx, line.strip()))

        self.assertEqual(
            violations,
            [],
            f"Forbidden Streamlit parameter 'use_container_width' found in Python files: {violations}"
        )


if __name__ == "__main__":
    unittest.main()
