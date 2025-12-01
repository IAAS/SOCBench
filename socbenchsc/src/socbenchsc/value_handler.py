import itertools
import ast
import logging
from socbenchsc.models import Constant, Function, FunctionCall, Unknown, Variable, List, Dict, RequestImport, UNKNOWN


class ValueHandler:
    def __init__(self):
        self.function_calls = []
        self.call_stack = set()

    def extract_value(self, statement: ast.AST, scope: Function) -> Variable:
        if isinstance(statement, ast.Constant):
            return Variable(values=frozenset({Constant(value=str(statement.value))}))
        elif isinstance(statement, ast.Name):
            return scope.lookup_value(statement.id)
        if isinstance(statement, ast.UnaryOp):
            result = self.extract_value(statement.operand, scope)
            if (isinstance(statement.op, ast.USub)
                    and len(result.values) == 1
                    and isinstance(list(result.values)[0], Constant)):
                result = {Constant(value="-" + list(result.values)[0].value)}
            else:
                logging.warning("Unsupported unary operator " + str(type(statement.op).__name__))
                return UNKNOWN
            return Variable(values=frozenset(result))
        elif isinstance(statement, ast.Call):
            return self.handle_call(statement, scope)
        elif isinstance(statement, ast.BinOp):
            return self.handle_binop(statement, scope)
        elif isinstance(statement, ast.FormattedValue):
            if statement.format_spec:
                logging.warning("Format specifications are not supported yet")
            return self.extract_value(statement.value, scope)
        elif isinstance(statement, ast.JoinedStr):
            return self.handle_format_string(statement, scope)
        elif isinstance(statement, ast.Tuple) or isinstance(statement, ast.List):
            return Variable(values=frozenset({
                List(values=tuple([self.extract_value(value, scope) for value in statement.elts]))
            }))
        elif isinstance(statement, ast.Attribute):
            self.extract_value(statement.value, scope)
            logging.warning("Extraction from attributes is not supported yet")
            return UNKNOWN
        elif isinstance(statement, ast.Dict):
            return self.handle_dict(statement, scope)
        elif isinstance(statement, ast.Subscript):
            dictionaries = self.extract_value(statement.value, scope)
            key = self.extract_value(statement.slice, scope)
            result = set()
            for dictionary in dictionaries.values:
                if not isinstance(dictionary, Dict):
                    logging.warning("Unsupported dictionary type " + str(type(dictionary).__name__))
                    result.add(Unknown())
                    continue
                result.update(dictionary.lookup(key).values)
            return Variable(values=frozenset(result))
        else:
            logging.warning("Unsupported argument type " + str(type(statement).__name__))
            return UNKNOWN

    def get_arguments(self, node, scope: Function) -> tuple[list[Variable], dict[str, Variable]]:
        arguments = [self.extract_value(argument, scope) for argument in node.args]
        kwargs = {}
        for keyword in node.keywords:
            kwargs.update(self.extract_kwargs(keyword, scope))
        return arguments, kwargs

    def extract_kwargs(self, node: ast.keyword, scope: Function):
        if node.arg:
            return {node.arg: self.extract_value(node.value, scope)}
        value = self.extract_value(node.value, scope)
        if len(value.values) == 1 and isinstance(list(value.values)[0], Dict):
            value = list(value.values)[0]
            result = {}
            for key, item in value.value:
                if len(key.values) != 1:
                    raise NotImplementedError("Multiple possible values for key in kwargs are not supported yet")
                key_value = list(key.values)[0]
                if not isinstance(key_value, Constant):
                    raise NotImplementedError("Unsupported key type " + str(type(key.values[0]).__name__))
                result[key_value.value] = item
            return result
        elif len(value.values) == 1 and list(value.values)[0] == Unknown():
            logging.warning("Unknown kwarg")
            return {}
        else:
            logging.warning("Unsupported keyword argument type " + str(type(node.value).__name__))
            return {}

    def handle_call(self, node, scope: Function) -> Variable:
        assert isinstance(node, ast.Call)
        if isinstance(node.func, ast.Attribute):
            function = node.func.attr
            if isinstance(node.func.value, ast.Name) and function != "format":
                self.add_call(node.func.value.id, function, node, scope)
                values = scope.lookup_value(node.func.value.id)
                for value in values.values:
                    if isinstance(value, RequestImport) and function == "Session":
                        return Variable(values=frozenset({RequestImport()}))
            else:
                module = self.extract_value(node.func.value, scope)
                result = set()
                for module_value in module.values:
                    if function == "format":
                        if not isinstance(module_value, Constant):
                            logging.warning("Unsupported format string type " + str(type(module_value).__name__))
                            continue
                        format_string = f'"{module_value.value}"'
                        self.add_call(Constant(value=format_string), function, node, scope)
                        result.update(self.handle_string_format(module_value, node, scope))
                    else:
                        self.add_call(module_value, function, node, scope)
                if function == "format":
                    return Variable(values=frozenset(result))
            return UNKNOWN
        if isinstance(node.func, ast.Name):
            function_name = node.func.id
            results = set()
            if function_name in self.call_stack:
                logging.warning(f"Call stack contains recursive function name \"{function_name}\" -> Skip")
                results.add(Unknown())
            else:
                (arguments, kwargs) = self.get_arguments(node, scope)
                self.function_calls.append(FunctionCall(name=function_name, arguments=arguments, kwargs=kwargs))
                functions = scope.lookup_value(function_name)
                if functions == UNKNOWN:
                    return UNKNOWN
                for function in functions.values:
                    if isinstance(function, Function):
                        self.call_stack.add(function_name)
                        result = invoke_function(function.transform, arguments, kwargs)
                        results.update(result)
                        self.call_stack.remove(function_name)
            return Variable(values=frozenset(results))
        logging.warning("Unsupported function call type " + str(type(node.func).__name__))
        return UNKNOWN

    def add_call(self, module, function, node, scope):
        function_signature = get_signature(module, function, scope)
        (arguments, kwargs) = self.get_arguments(node, scope)
        self.function_calls.append(FunctionCall(name=function_signature, arguments=arguments, kwargs=kwargs))

    def handle_binop(self, node: ast.BinOp, scope: Function) -> Variable:
        assert isinstance(node, ast.BinOp)
        if isinstance(node.op, ast.Add):
            left = self.extract_value(node.left, scope)
            if left != Unknown():
                left = [value for value in left.values if value != Unknown()]
            right = self.extract_value(node.right, scope)
            if right != Unknown():
                right = [value for value in right.values if value != Unknown()]
            if not left:
                left = [Unknown()]
            if not right:
                right = [Unknown()]
            results = itertools.product(left, right)
            result_set = {add_variables(left, right) for left, right in results}
            return Variable(values=frozenset(result_set))
        else:
            logging.warning("Unsupported binary operator " + str(type(node.op).__name__))
            return UNKNOWN

    def handle_format_string(self, node: ast.JoinedStr, scope: Function) -> Variable:
        assert isinstance(node, ast.JoinedStr)
        values = [self.extract_value(value, scope).values for value in node.values]
        combinations = itertools.product(*values)
        result = {Constant(value="".join(map(str, value))) for value in combinations}
        return Variable(values=frozenset(result))

    def handle_string_format(self, format_string, node, scope: Function):
        if not isinstance(format_string, Constant):
            logging.warning("Unsupported format string type " + str(type(format_string).__name__))
            return {Unknown()}
        (arguments, kwargs) = self.get_arguments(node, scope)

        def transform(*args, **kwargs):
            try:
                return {Constant(value=format_string.value.format(*args, **kwargs))}
            except KeyError:
                logging.warning("KeyError in string.format")
                return {format_string}
        return invoke_function(transform, arguments, kwargs)

    def handle_dict(self, node: ast.Dict, scope: Function):
        result = Dict()
        for key, value in zip(node.keys, node.values):
            value_value = self.extract_value(value, scope)
            if key is None:
                for current_dict in value_value.values:
                    if not isinstance(current_dict, Dict):
                        logging.warning("Unsupported dictionary type " + str(type(current_dict).__name__))
                        continue
                    for key, item in current_dict.value:
                        result = result.add(key, item)
            else:
                key_value = self.extract_value(key, scope)
                result = result.add(key_value, value_value)
        return Variable(values=frozenset({result}))


def invoke_function(transform, arguments, kwargs):
    result = set()
    if not arguments and not kwargs:
        result = transform()
        assert isinstance(result, set) or isinstance(result, frozenset)
        return result
    keys = list(kwargs.keys())
    values = [kwargs[key] for key in keys]

    def get_values(input_arguments):
        return [value.values for value in input_arguments]
    parameters = itertools.product(*get_values(arguments), *get_values(values))
    for parameter_set in parameters:
        parameter_set = list(parameter_set)
        arguments = parameter_set[:len(arguments)]
        kwargs = {}
        for i in range(len(keys)):
            index = len(parameter_set) - len(keys) + i
            kwargs[keys[i]] = parameter_set[index]
        current_result = transform(*arguments, **kwargs)
        assert isinstance(current_result, set) or isinstance(current_result, frozenset)
        result.update(current_result)
    return result


def add_variables(left, right):
    if isinstance(left, Constant) and isinstance(right, Constant):
        return Constant(value=left.value + right.value)
    if isinstance(left, List) and isinstance(right, List):
        return List(values=left.values + right.values)
    if left != Unknown() and right == Unknown():
        return left
    if left == Unknown() and right != Unknown():
        return right
    if isinstance(left, List) or isinstance(right, List):
        logging.warning("Adding list to constant or Unknown results in Unknown")
    return Unknown()


def get_signature(module, function, scope: Function) -> str:
    values = scope.lookup_value(module)
    for value in values.values:
        if isinstance(value, RequestImport):
            return f"requests.{function}"
    if ((not isinstance(module, str) and function != "format")
            or (isinstance(module, str) and scope.lookup_value(module) != UNKNOWN)):
        logging.warning("Attribute calls are not supported yet")
    return f"{module}.{function}" if len(str(module)) > 0 else function
