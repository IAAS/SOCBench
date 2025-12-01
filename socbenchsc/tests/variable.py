from socbenchsc.models import Variable, Constant, Unknown, Function, List, Dict, RequestImport


def variable(*args) -> Variable:
    variables = set()
    for value in args:
        if isinstance(value, str):
            variables.add(Constant(value=value))
        elif isinstance(value, Constant):
            variables.add(value)
        elif value == Unknown():
            variables.add(Unknown())
        elif isinstance(value, Function):
            variables.add(value)
        elif isinstance(value, Variable):
            variables.update(value.values)
        elif isinstance(value, dict):
            result = []
            for key, item in value.items():
                result.append((key, item))
            variables.add(Dict(value=frozenset(result)))
        elif isinstance(value, RequestImport):
            variables.add(value)
    return Variable(values=frozenset(variables))


def create_list(*args) -> List:
    variables = []
    for value in args:
        if isinstance(value, str):
            variables.append(Variable(values=frozenset({Constant(value=value)})))
        elif isinstance(value, Constant):
            variables.append(Variable(values=frozenset({value})))
        elif value == Unknown():
            variables.append(Variable(values=frozenset({Unknown()})))
        elif isinstance(value, Function):
            variables.append(Variable(values=frozenset({value})))
        elif isinstance(value, Variable):
            variables.append(value)
    return List(values=tuple(variables))


def variable_list(*args) -> Variable:
    result = create_list(*args)
    return Variable(values=frozenset({result}))
