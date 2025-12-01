import ast
import logging
from socbenchsc.value_handler import ValueHandler
from socbenchsc.models import Function, Variable, List, Dict, Constant, UNKNOWN


def kill_targets(target) -> list[str] | str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Tuple) or isinstance(target, ast.List):
        return [kill_targets(element) for element in target.elts]
    if isinstance(target, ast.Subscript):
        if isinstance(target.value, ast.Name) and isinstance(target.slice, ast.Constant):
            return f"{target.value.id}['{target.slice.value}']"
        elif isinstance(target.value, ast.Name) and isinstance(target.slice, ast.Name):
            return f"{target.value.id}[{target.slice.id}]"
        else:
            logging.warning("Unsupported subscript assignment")
        return []
    logging.warning("Unknown target type " + str(type(target).__name__))
    return []


def flatten(target_list) -> list[str]:
    result = []
    for target in target_list:
        if isinstance(target, list):
            result.extend(flatten(target))
        else:
            result.append(target)
    return result


def kill(statement: ast.Assign) -> list[str]:
    assert isinstance(statement, ast.Assign)
    targets = [kill_targets(target) for target in statement.targets]
    return flatten(targets)


def add_value(value: Variable, variables: dict[str, Variable], variable_name: str) -> Variable:
    if variable_name in variables:
        return Variable(values=value.values | variables[variable_name].values)
    return value


def merge_simple_assignment(variables: dict[str, Variable], variable_name: str, values: Variable, scope: Function):
    result = variables.copy()
    if "[" in variable_name:
        variable, slice = split_key_name(variable_name, scope)
        if slice == UNKNOWN:
            return result
        if variable not in result:
            result[variable] = Variable(values=frozenset({Dict()}))
        current_result = []
        for value in result[variable].values:
            if not isinstance(value, Dict):
                logging.warning("Subscript assignment on non-dict variable. Skip")
                current_result.append(value)
                continue
            current_result.append(value.add(slice, values))
        result[variable] = Variable(values=frozenset(current_result))
    else:
        result[variable_name] = values
    return result


def merge_destructuring_assignment(targets: list, variable: Variable, variables: dict[str, Variable], scope: Function):
    for i, target in enumerate(targets):
        sub_values = []
        for list_val in variable.values:
            if not isinstance(list_val, List):
                logging.warning("Unsupported destructuring assignment. Non-list to list. Skip")
                continue
            elems = list(list_val.values)
            if len(elems) <= i:
                logging.warning("List does not contain enough elements for destructuring assignment. Skip")
                continue
            sub_values.append(elems[i])
        if not sub_values:
            continue
        if isinstance(target, str):
            for sub_value in sub_values:
                variables = merge_simple_assignment(variables, target, sub_value, scope)
        else:
            sub_maps = [merge_assignments(target, sub_value, variables, scope) for sub_value in sub_values]
            for sub_map in sub_maps:
                for key, value in sub_map.items():
                    variables[key] = add_value(value, variables, key)
    return variables


def merge_assignments(targets: list[str], value: Variable, variables: dict[str, Variable], scope: Function):
    assert isinstance(value, Variable)
    if isinstance(targets, str):
        targets = [targets]
    if len(targets) == 1:
        return merge_simple_assignment(variables, targets[0], value, scope)
    return merge_destructuring_assignment(targets, value, variables, scope)


def generate(statement: ast.Assign | ast.AnnAssign, scope: Function, value_handler: ValueHandler)\
        -> dict[str, Variable]:
    assert isinstance(statement, ast.Assign) or isinstance(statement, ast.AnnAssign)
    assert len(statement.targets) > 0
    variables = scope.variables.copy()
    for target in reversed(statement.targets):
        targets = kill_targets(target)
        values = value_handler.extract_value(statement.value, scope)
        remove_killed_variables(targets, variables, scope)
        variables = merge_assignments(targets, values, variables, scope)
    return variables


def remove_killed_variables(targets, variables: dict[str, Variable], scope: Function):
    if isinstance(targets, str):
        if "[" in targets:
            variable_name, slice = split_key_name(targets, scope)
            if variable_name not in variables:
                return
            variable = variables[variable_name]
            result = set()
            for value in variable.values:
                if not isinstance(value, Dict):
                    logging.warning("Subscript deletion on non-dict variable. Skip")
                    result.add(value)
                    continue
                result.add(value.remove(slice))
            variables[variable_name] = Variable(values=frozenset(result))
        elif targets in variables:
            del variables[targets]
    else:
        for target in targets:
            remove_killed_variables(target, variables, scope)


def split_key_name(complete: str, scope: Function) -> tuple[str, Variable]:
    variable_name = complete.split("[")[0]
    slice = complete.split("[")[1][:-1]
    if "'" in slice:
        return variable_name, Variable(values=frozenset({Constant(value=slice[1:-1])}))
    else:
        return variable_name, scope.lookup_value(slice)
