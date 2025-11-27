import ast
import re
from typing import Dict, Any, List, Tuple

class ComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.patterns = {
            'nestedLoops': 0,
            'recursiveCalls': 0,
            'dynamicProgramming': False,
            'divideAndConquer': False,
            'backtracking': False,
            'memoization': False,
        }
        self.loops = {
            'for': 0,
            'while': 0,
            'listComp': 0,
            'map': 0,
            'filter': 0
        }
        self.dataStructures = {
            'lists': 0,
            'dicts': 0,
            'sets': 0,
            'tuples': 0,
            'deque': 0,
            'heapq': 0
        }
        self.operations = {
            'sorting': 0,
            'searching': 0,
            'binarySearch': 0,
            'exponential': 0,
            'factorial': 0
        }
        self.recursion = {
            'count': 0,
            'functions': [],
            'multipleRecursive': False
        }
        self.advancedPatterns = {
            'itertools': {
                'product': 0,
                'combinations': 0,
                'permutations': 0,
                'chain': 0
            },
            'nestedProduct': False,
            'doubleExponential': False,
            'cartesianProduct': 0,
            'combinatorialExplosion': False
        }
        
        self.current_function = None
        self.function_calls = {}
        self.loop_depth = 0
        self.max_loop_depth = 0
        self.source_code = ""
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """Main analysis entry point"""
        self.source_code = code
        
        try:
            tree = ast.parse(code)
            self.visit(tree)
            self._detect_patterns_from_source()
            self._analyze_recursion()
            
            time_complexity = self._calculate_time_complexity()
            space_complexity = self._calculate_space_complexity()
            
            return {
                'patterns': self.patterns,
                'loops': self.loops,
                'dataStructures': self.dataStructures,
                'operations': self.operations,
                'recursion': self.recursion,
                'advancedPatterns': self.advancedPatterns,
                'timeComplexity': time_complexity,
                'spaceComplexity': space_complexity,
                'totalLines': len([l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')])
            }
        except SyntaxError as e:
            return {
                'error': f'Syntax error: {str(e)}',
                'patterns': {},
                'loops': {},
                'dataStructures': {},
                'operations': {},
                'recursion': {},
                'advancedPatterns': {},
                'timeComplexity': {'notation': 'N/A', 'explanation': 'Syntax error in code', 'confidence': 'none'},
                'spaceComplexity': {'notation': 'N/A', 'explanation': 'Syntax error in code', 'confidence': 'none'},
                'totalLines': 0
            }
    
    def visit_For(self, node):
        """Detect for loops"""
        self.loops['for'] += 1
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1
    
    def visit_While(self, node):
        """Detect while loops"""
        self.loops['while'] += 1
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1
    
    def visit_ListComp(self, node):
        """Detect list comprehensions"""
        self.loops['listComp'] += 1
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Track function definitions and calls"""
        prev_function = self.current_function
        self.current_function = node.name
        
        if node.name not in self.function_calls:
            self.function_calls[node.name] = []
        
        self.generic_visit(node)
        self.current_function = prev_function
    
    def visit_Call(self, node):
        """Detect function calls and special operations"""
        func_name = self._get_call_name(node)
        
        # Track calls for recursion detection
        if self.current_function and func_name:
            self.function_calls.setdefault(self.current_function, []).append(func_name)
        
        # Detect itertools
        if func_name in ['product', 'combinations', 'permutations', 'chain']:
            self.advancedPatterns['itertools'][func_name] += 1
        
        # Detect special operations
        if func_name in ['sorted', 'sort']:
            self.operations['sorting'] += 1
        elif func_name == 'map':
            self.loops['map'] += 1
        elif func_name == 'filter':
            self.loops['filter'] += 1
        elif func_name in ['pow', 'factorial']:
            if func_name == 'factorial':
                self.operations['factorial'] += 1
            else:
                self.operations['exponential'] += 1
        elif func_name in ['heappush', 'heappop', 'heapq']:
            self.dataStructures['heapq'] += 1
        elif func_name == 'deque':
            self.dataStructures['deque'] += 1
        
        self.generic_visit(node)
    
    def visit_List(self, node):
        """Detect list usage"""
        self.dataStructures['lists'] += 1
        self.generic_visit(node)
    
    def visit_Dict(self, node):
        """Detect dictionary usage"""
        self.dataStructures['dicts'] += 1
        self.generic_visit(node)
    
    def visit_Set(self, node):
        """Detect set usage"""
        self.dataStructures['sets'] += 1
        self.generic_visit(node)
    
    def visit_Tuple(self, node):
        """Detect tuple usage"""
        self.dataStructures['tuples'] += 1
        self.generic_visit(node)
    
    def visit_BinOp(self, node):
        """Detect exponential operations"""
        if isinstance(node.op, ast.Pow):
            self.operations['exponential'] += 1
        self.generic_visit(node)
    
    def visit_Compare(self, node):
        """Detect 'in' operator (searching)"""
        for op in node.ops:
            if isinstance(op, ast.In):
                self.operations['searching'] += 1
        self.generic_visit(node)
    
    def _get_call_name(self, node) -> str:
        """Extract function name from Call node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""
    
    def _detect_patterns_from_source(self):
        """Detect patterns using regex on source code"""
        code = self.source_code.lower()
        
        # Nested loops
        self.patterns['nestedLoops'] = self.max_loop_depth
        
        # Dynamic programming patterns
        if re.search(r'\bmemo\b|\bcache\b|\bdp\[|\btable\[', code):
            self.patterns['dynamicProgramming'] = True
            self.patterns['memoization'] = True
        
        # Divide and conquer
        if re.search(r'\bmid\b|left.*right|partition', code):
            self.patterns['divideAndConquer'] = True
        
        # Backtracking
        if re.search(r'backtrack|undo|restore', code):
            self.patterns['backtracking'] = True
        
        # Binary search
        if re.search(r'binary.*search|left.*right.*mid', code):
            self.operations['binarySearch'] += 1
        
        # Double exponential patterns
        if re.search(r'product\([^)]*len\([^)]*product', self.source_code):
            self.advancedPatterns['nestedProduct'] = True
            self.advancedPatterns['doubleExponential'] = True
            self.advancedPatterns['combinatorialExplosion'] = True
        
        # Multiple products in loops
        product_in_loop = re.findall(r'for\s+\w+\s+in\s+product', self.source_code)
        if len(product_in_loop) >= 2:
            self.advancedPatterns['doubleExponential'] = True
        
        # Nested product calls
        if re.search(r'product\([^)]*product\(', self.source_code):
            self.advancedPatterns['cartesianProduct'] += 1
            self.advancedPatterns['doubleExponential'] = True
    
    def _analyze_recursion(self):
        """Analyze recursive function calls"""
        for func_name, calls in self.function_calls.items():
            if func_name in calls:
                self.recursion['count'] += 1
                self.recursion['functions'].append(func_name)
                
                # Check for multiple recursive calls
                if calls.count(func_name) >= 2:
                    self.recursion['multipleRecursive'] = True
    
    def _calculate_time_complexity(self) -> Dict[str, str]:
        """Calculate time complexity based on detected patterns"""
        notation = 'O(1)'
        explanation = 'Constant time - no loops or recursion detected'
        confidence = 'high'
        
        # Double exponential (highest priority)
        if self.advancedPatterns['doubleExponential'] or self.advancedPatterns['nestedProduct']:
            notation = 'O(2^(2^n))'
            explanation = '⚠️ DOUBLE EXPONENTIAL - nested exponential growth (nested product/combinations)'
            confidence = 'high'
        
        # Combinatorial explosion
        elif self.advancedPatterns['itertools']['product'] >= 2 or self.advancedPatterns['itertools']['permutations'] > 0:
            notation = 'O(2^n) to O(n!)'
            explanation = 'Combinatorial explosion from multiple itertools operations'
            confidence = 'high'
        
        # Single product
        elif self.advancedPatterns['itertools']['product'] == 1:
            notation = 'O(k^n)'
            explanation = 'Exponential from itertools.product - generates k^n combinations'
            confidence = 'high'
        
        # Exponential/Factorial
        elif self.operations['exponential'] > 0 or self.operations['factorial'] > 0:
            notation = 'O(2^n) or O(n!)'
            explanation = 'Exponential/factorial operations detected'
            confidence = 'high'
        
        # Multiple recursive calls without memoization
        elif self.recursion['multipleRecursive'] and not self.patterns['memoization']:
            notation = 'O(2^n)'
            explanation = 'Multiple recursive calls without memoization (exponential growth)'
            confidence = 'high'
        
        # Triple nested loops
        elif self.patterns['nestedLoops'] >= 3:
            notation = 'O(n³)'
            explanation = f"{self.patterns['nestedLoops']} levels of nested loops detected"
            confidence = 'high'
        
        # Double nested loops
        elif self.patterns['nestedLoops'] == 2:
            notation = 'O(n²)'
            explanation = 'Two levels of nested loops detected'
            confidence = 'high'
        
        # Sorting
        elif self.operations['sorting'] > 0:
            notation = 'O(n log n)'
            explanation = 'Sorting operation detected (typically O(n log n))'
            confidence = 'medium'
        
        # Binary search or divide and conquer
        elif self.operations['binarySearch'] > 0 or self.patterns['divideAndConquer']:
            notation = 'O(log n) or O(n log n)'
            explanation = 'Divide and conquer or binary search pattern'
            confidence = 'medium'
        
        # Recursion with memoization
        elif self.recursion['count'] > 0:
            if self.patterns['memoization'] or self.patterns['dynamicProgramming']:
                notation = 'O(n) or O(n²)'
                explanation = 'Recursion with memoization/DP optimization'
            else:
                notation = 'O(n) to O(2^n)'
                explanation = 'Recursion detected - complexity depends on recursive structure'
            confidence = 'medium'
        
        # Single loop
        elif sum(self.loops.values()) > 0:
            notation = 'O(n)'
            explanation = 'Linear iteration through data'
            confidence = 'high'
        
        # Search operations
        elif self.operations['searching'] > 0:
            notation = 'O(n) or O(1)'
            explanation = 'Search operation - O(1) for hash-based, O(n) for linear'
            confidence = 'medium'
        
        return {
            'notation': notation,
            'explanation': explanation,
            'confidence': confidence
        }
    
    def _calculate_space_complexity(self) -> Dict[str, str]:
        """Calculate space complexity based on detected patterns"""
        notation = 'O(1)'
        explanation = 'Constant space - no significant data structures'
        confidence = 'high'
        
        total_ds = sum(self.dataStructures.values())
        
        # Double exponential space
        if self.advancedPatterns['doubleExponential']:
            notation = 'O(2^(2^n))'
            explanation = '⚠️ DOUBLE EXPONENTIAL space - storing all combinations of combinations'
            confidence = 'high'
        
        # Exponential space from itertools
        elif self.advancedPatterns['itertools']['product'] > 0 and total_ds > 0:
            notation = 'O(2^n) to O(k^n)'
            explanation = 'Exponential space - storing all product combinations'
            confidence = 'high'
        
        # Recursion uses stack space
        elif self.recursion['count'] > 0:
            notation = 'O(n)'
            explanation = f"Recursion uses call stack space - {self.recursion['count']} recursive function(s)"
            confidence = 'high'
        
        # Multiple data structures
        elif total_ds >= 3:
            notation = 'O(n)'
            explanation = 'Multiple data structures allocated proportional to input'
            confidence = 'medium'
        
        # Single data structure
        elif total_ds > 0:
            notation = 'O(n)'
            explanation = 'Data structures proportional to input size'
            confidence = 'medium'
        
        # Dynamic programming
        if self.patterns['dynamicProgramming'] and notation != 'O(2^(2^n))':
            notation = 'O(n) or O(n²)'
            explanation = 'Dynamic programming table/memoization storage'
            confidence = 'medium'
        
        return {
            'notation': notation,
            'explanation': explanation,
            'confidence': confidence
        }


def analyze_code_complexity(code: str) -> Dict[str, Any]:
    """Public API for complexity analysis"""
    analyzer = ComplexityAnalyzer()
    return analyzer.analyze(code)
