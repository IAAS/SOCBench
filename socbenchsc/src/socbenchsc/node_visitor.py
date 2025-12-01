import ast
import logging
from socbenchsc.value_handler import ValueHandler
from socbenchsc.models import Function, Variable, RequestImport
from socbenchsc.generator import generate, add_value


class AnalysisVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.scope = Function()
        self.scope.variables["requests"] = Variable(values=frozenset({RequestImport()}))
        self.value_handler = ValueHandler()
        self.request_methods = set()

    def get_variables(self):
        return self.scope.variables

    def get_function_calls(self):
        return self.value_handler.function_calls

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "requests":
                name = alias.asname if alias.asname else alias.name
                self.scope.variables[name] = Variable(values=frozenset({RequestImport()}))

    def visit_ImportFrom(self, node):
        if node.module == "requests":
            for alias in node.names:
                if alias.name == "Session":
                    raise NotImplementedError("Import Session not supported")
                self.request_methods.add(alias.name)

    def visit_Assign(self, node):
        generated = generate(node, self.scope, self.value_handler)
        self.scope.variables = generated

    def visit_AnnAssign(self, node):
        if node.simple == 0:
            raise NotImplementedError("Complex annotations are not supported yet")
        assignment = ast.Assign(targets=[node.target], value=node.value)
        self.visit_Assign(assignment)

    def visit_Call(self, node):
        self.value_handler.handle_call(node, self.scope)

    def visit_FunctionDef(self, node):
        function = Function(parent=self.scope)

        def transform(*args, **kwargs):
            if len(args) + len(kwargs) != len(node.args.args):
                logging.warning("Input arguments do not match the function signature")
            previous_scope = self.scope
            self.scope = function
            self.scope.variables = {}
            for arg, value in zip(node.args.args, args):
                self.scope.variables[arg.arg] = Variable(values=frozenset({value}))
            for arg, value in kwargs.items():
                self.scope.variables[arg] = Variable(values=frozenset({value}))
            ast.NodeVisitor.generic_visit(self, node)
            self.scope = previous_scope
            returns = function.returns
            function.returns = None
            return returns.values if returns else set()
        function.transform = transform
        self.scope.variables[node.name] = Variable(values=frozenset({function}))

    def visit_Return(self, node):
        if node.value is None:
            pass
        value = self.value_handler.extract_value(node.value, self.scope)
        if self.scope.returns is None:
            self.scope.returns = value
        else:
            self.scope.returns = Variable(values=self.scope.returns.values | value.values)

    def visit_If(self, node):
        previous = self.scope.variables
        if node.orelse:
            for statement in node.body:
                self.visit(statement)
            if_results = self.scope.variables
            self.scope.variables = previous
            for statement in node.orelse:
                self.visit(statement)
            for key, value in if_results.items():
                self.scope.variables[key] = add_value(value, self.scope.variables, key)
        else:
            for statement in node.body:
                self.visit(statement)
            for key, value in previous.items():
                self.scope.variables[key] = add_value(value, self.scope.variables, key)

    def visit_While(self, node):
        self.handle_loop(node)

    def visit_For(self, node):
        self.handle_loop(node)

    def handle_loop(self, node):
        assert isinstance(node, ast.For) or isinstance(node, ast.While)
        previous = self.scope.variables
        count = 0
        while count < 5:
            count += 1
            for statement in node.body:
                self.visit(statement)
            for key, value in previous.items():
                self.scope.variables[key] = add_value(value, self.scope.variables, key)
            if previous == self.scope.variables:
                break
            previous = self.scope.variables
            if count == 5:
                logging.warning("While loop terminated after 5 iterations")

    def visit_Try(self, node):
        previous = self.scope.variables
        for statement in node.body:
            self.visit(statement)
        try_results = self.scope.variables
        for exception_handler in node.handlers:
            self.scope.variables = previous
            for statement in exception_handler.body:
                self.visit(statement)
            for key, value in self.scope.variables.items():
                try_results[key] = add_value(value, try_results, key)
        self.scope.variables = try_results

    def visit_TryStar(self, node):
        raise NotImplementedError("TryStar statements are not supported yet")

    def visit_With(self, node):
        logging.warning("With statements are not supported yet")
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Match(self, node):
        raise NotImplementedError("Match statements are not supported yet")

    def visit_Yield(self, node):
        logging.warning("Yield statements are not supported yet")
        ast.NodeVisitor.generic_visit(self, node)

    def visit_YieldFrom(self, node):
        logging.warning("YieldFrom statements are not supported yet")
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Delete(self, node):
        logging.warning("Delete statements are not supported yet")
        ast.NodeVisitor.generic_visit(self, node)
