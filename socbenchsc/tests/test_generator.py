import ast
from socbenchsc.generator import kill, generate
from socbenchsc.value_handler import ValueHandler
from socbenchsc.models import Constant, Function, Unknown
from tests.variable import create_list, variable, variable_list


def test_kill():
    code = "a = 1"
    result = kill(ast.parse(code).body[0])
    assert result == ["a"]
    code = "b = 5"
    result = kill(ast.parse(code).body[0])
    assert result == ["b"]
    code = "a = b = 2"
    result = kill(ast.parse(code).body[0])
    assert result == ["a", "b"]
    code = "x, y = 5, 6"
    result = kill(ast.parse(code).body[0])
    assert result == ["x", "y"]
    code = "[x, y] = [5, 6]"
    result = kill(ast.parse(code).body[0])
    assert result == ["x", "y"]
    code = "[[[x]], y] = [[[9]], -1]"
    result = kill(ast.parse(code).body[0])
    assert result == ["x", "y"]
    code = "test['5'] = 1"
    result = kill(ast.parse(code).body[0])
    assert result == ["test['5']"]
    code = "x[y] = 6"
    result = kill(ast.parse(code).body[0])
    assert result == ["x[y]"]
    code = "a = b = c"
    result = kill(ast.parse(code).body[0])
    assert result == ["a", "b"]


def test_generate_empty_scope():
    code = "a = 1"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"a": variable("1")}
    code = "b = 5"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"b": variable("5")}
    code = "z = x"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"z": variable(Unknown())}
    code = "x, y = 5, 6"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "x": variable("5"),
        "y": variable("6")
    }
    code = "[x1, x2] = [8, 5]"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "x1": variable("8"),
        "x2": variable("5")
    }
    code = "x = [5], 6"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "x": variable_list(variable_list(variable("5")), variable("6"))
    }
    code = "a, [b, c] = 10, [20, 30]"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "a": variable("10"),
        "b": variable("20"),
        "c": variable("30"),
    }
    code = "[a, [b, c]], d = [[-5, [5, 8]], 110]"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "a": variable("-5"),
        "b": variable("5"),
        "c": variable("8"),
        "d": variable("110"),
    }


def test_generate_assign_variables():
    code = "a = x"
    result = generate(ast.parse(code).body[0], Function(variables={
        "x": variable("9")
    }), ValueHandler())
    assert result == {
        "a": variable("9"),
        "x": variable("9")
    }
    code = "b = a"
    result = generate(ast.parse(code).body[0], Function(variables={
        "a": variable("-5", "8")
    }), ValueHandler())
    assert result == {
        "a": variable("-5", "8"),
        "b": variable("-5", "8")
    }
    code = "b = a"
    result = generate(ast.parse(code).body[0], Function(variables={
        "a": variable("6", "-8"),
        "b": variable("4")
    }), ValueHandler())
    assert result == {
        "a": variable("6", "-8"),
        "b": variable("6", "-8")
    }
    code = "t1 = t2"
    result = generate(ast.parse(code).body[0], Function(variables={
        "x": variable("50")
    }), ValueHandler())
    assert result == {
        "t1": variable(Unknown()),
        "x": variable("50")
    }
    code = "y = [x]"
    result = generate(ast.parse(code).body[0], Function(variables={
        "x": variable("-1")
    }), ValueHandler())
    assert result == {
        "y": variable_list(variable("-1")),
        "x": variable("-1")
    }
    code = "test2 = [[test]]"
    result = generate(ast.parse(code).body[0], Function(variables={
        "test": variable("-1", "p", Unknown())
    }), ValueHandler())
    assert result == {
        "test2": variable_list(variable_list(variable("-1", "p", Unknown()))),
        "test": variable("-1", "p", Unknown())
    }
    code = "[x, y] = x"
    result = generate(ast.parse(code).body[0], Function(variables={
        "x": variable_list(variable("-43", "25"), variable("result"))
    }), ValueHandler())
    assert result == {
        "x": variable("-43", "25"),
        "y": variable("result")
    }


def test_string_add():
    code = "r = 'a' + 'b'"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"r": variable("ab")}
    code = "x = 'acb' + 'qw' + 'erty'"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"x": variable("acbqwerty")}
    code = "result = 'a' + b"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"result": variable("a")}
    code = "y = x + 'z'"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"y": variable("z")}
    code = "n = '1' + var + '2'"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"n": variable("12")}
    code = "res = '2' + v"
    result = generate(ast.parse(code).body[0], Function(variables={"v": variable("test")}), ValueHandler())
    assert result == {
        "v": variable("test"),
        "res": variable("2test")
    }
    code = "t2 = x + 'testtest'"
    result = generate(ast.parse(code).body[0], Function(variables={"x": variable("input")}), ValueHandler())
    assert result == {
        "x": variable("input"),
        "t2": variable("inputtesttest")
    }
    code = "tt = in_1 + in_2"
    result = generate(ast.parse(code).body[0], Function(variables={
        "in_1": variable("value1"),
        "in_2": variable("value2")
    }), ValueHandler())
    assert result == {
        "in_1": variable("value1"),
        "in_2": variable("value2"),
        "tt": variable("value1value2")
    }
    code = "r = a + 'b' + 'c'"
    result = generate(ast.parse(code).body[0], Function(variables={"a": variable("b")}), ValueHandler())
    assert result == {
        "a": variable("b"),
        "r": variable("bbc")
    }
    code = "y = 'in' + x + z"
    result = generate(ast.parse(code).body[0], Function(variables={
        "x": variable("2"),
        "z": variable("8")
    }), ValueHandler())
    assert result == {
        "x": variable("2"),
        "z": variable("8"),
        "y": variable("in28")
    }
    code = "result = r1 + r2 + r3"
    result = generate(ast.parse(code).body[0], Function(variables={
        "r1": variable("t3"),
        "r2": variable("t1"),
        "r3": variable("t2")
    }), ValueHandler())
    assert result == {
        "r1": variable("t3"),
        "r2": variable("t1"),
        "r3": variable("t2"),
        "result": variable("t3t1t2")
    }
    code = "result = 'a' + b"
    result = generate(ast.parse(code).body[0], Function(variables={
        "b": variable("t2", "t1")
    }), ValueHandler())
    assert result == {
        "b": variable("t2", "t1"),
        "result": variable("at2", "at1")
    }
    code = "out = input + '2'"
    result = generate(ast.parse(code).body[0], Function(variables={
        "input": variable("v1", "v2", "v3")
    }), ValueHandler())
    assert result == {
        "input": variable("v1", "v2", "v3"),
        "out": variable("v12", "v22", "v32")
    }
    code = "r = 't1' + b + 't2'"
    result = generate(ast.parse(code).body[0], Function(variables={
        "b": variable("a", "b")
    }), ValueHandler())
    assert result == {
        "b": variable("a", "b"),
        "r": variable("t1at2", "t1bt2")
    }
    code = "a = a1 + a2"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "a": variable(Unknown()),
    }
    code = "z = x + y"
    result = generate(ast.parse(code).body[0], Function(variables={
        "x": variable("x1", "x2"),
        "y": variable("y1", "y2")
    }), ValueHandler())
    assert result == {
        "x": variable("x1", "x2"),
        "y": variable("y1", "y2"),
        "z": variable("x1y1", "x1y2", "x2y1", "x2y2")
    }
    code = "r = v1 + v2 + v3"
    result = generate(ast.parse(code).body[0], Function(variables={
        "v1": variable("a1", "a2", "a3"),
        "v2": variable("b1", "b2", "b3"),
        "v3": variable("c1", "c2", "c3")
    }), ValueHandler())
    assert result == {
        "v1": variable("a1", "a2", "a3"),
        "v2": variable("b1", "b2", "b3"),
        "v3": variable("c1", "c2", "c3"),
        "r": variable(
            "a1b1c1", "a1b1c2", "a1b1c3",
            "a1b2c1", "a1b2c2", "a1b2c3",
            "a1b3c1", "a1b3c2", "a1b3c3",
            "a2b1c1", "a2b1c2", "a2b1c3",
            "a2b2c1", "a2b2c2", "a2b2c3",
            "a2b3c1", "a2b3c2", "a2b3c3",
            "a3b1c1", "a3b1c2", "a3b1c3",
            "a3b2c1", "a3b2c2", "a3b2c3",
            "a3b3c1", "a3b3c2", "a3b3c3")
    }
    code = "r = 'test' + v2"
    result = generate(ast.parse(code).body[0], Function(variables={
        "v2": variable("v2", Unknown())
    }), ValueHandler())
    assert result == {
        "v2": variable("v2", Unknown()),
        "r": variable("testv2")
    }
    code = "c = ['a'] + ['b']"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"c": variable_list("a", "b")}
    code = "result = r1 + r2"
    result = generate(ast.parse(code).body[0], Function(variables={
        "r1": variable_list(variable("t1")),
        "r2": variable_list(variable("t2"))
    }), ValueHandler())
    assert result == {
        "r1": variable_list(variable("t1")),
        "r2": variable_list(variable("t2")),
        "result": variable_list(variable("t1"), variable("t2"))
    }
    code = "t = a + b"
    result = generate(ast.parse(code).body[0], Function(variables={
        "a": variable_list(variable("a"))
    }), ValueHandler())
    assert result == {
        "a": variable_list(variable("a")),
        "t": variable_list(variable("a"))
    }
    code = "res = a1 + a2"
    result = generate(ast.parse(code).body[0], Function(variables={
        "a1": variable(variable_list("a"), Unknown()),
        "a2": variable(Unknown())
    }), ValueHandler())
    assert result == {
        "a1": variable(variable_list("a"), Unknown()),
        "a2": variable(Unknown()),
        "res": variable(variable_list("a"))
    }
    code = "y = f(x1) + x2 + 'c'"
    f = Function(transform=lambda x: {x})
    result = generate(ast.parse(code).body[0], Function(variables={
        "x1": variable("qwer"),
        "x2": variable("ty"),
        "f": variable(f)
    }), ValueHandler())
    assert result == {
        "x1": variable("qwer"),
        "x2": variable("ty"),
        "f": variable(f),
        "y": variable("qwertyc")
    }


def test_format_string():
    code = "left = f'test'"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"left": variable("test")}
    code = "y = f'{v}'"
    result = generate(ast.parse(code).body[0], Function(variables={"v": variable("result")}), ValueHandler())
    assert result == {
        "v": variable("result"),
        "y": variable("result")
    }
    code = "result = f'{x1} {x2}'"
    result = generate(ast.parse(code).body[0], Function(variables={
        "x1": variable("test1"),
        "x2": variable("test2")
    }), ValueHandler())
    assert result == {
        "x1": variable("test1"),
        "x2": variable("test2"),
        "result": variable("test1 test2")
    }
    code = "result = f'Value: {in_1}{in_2}'"
    result = generate(ast.parse(code).body[0], Function(variables={
        "in_1": variable("in1", "in2"),
        "in_2": variable("in3", "in4")
    }), ValueHandler())
    assert result == {
        "in_1": variable("in1", "in2"),
        "in_2": variable("in3", "in4"),
        "result": variable("Value: in1in3", "Value: in1in4", "Value: in2in3", "Value: in2in4")
    }
    code = "t = f'T123 {y1} T234 {y}'"
    result = generate(ast.parse(code).body[0], Function(variables={
        "y1": variable("test1"),
    }), ValueHandler())
    assert result == {
        "y1": variable("test1"),
        "t": variable("T123 test1 T234 ")
    }
    code = "res = f'a{x1}b{x2}c{x3}d'"
    result = generate(ast.parse(code).body[0], Function(variables={}), ValueHandler())
    assert result == {
        "res": variable("abcd")
    }
    code = "y = f'{x + z}'"
    result = generate(ast.parse(code).body[0], Function(variables={
        "x": variable("t1"),
        "z": variable("t2")
    }), ValueHandler())
    assert result == {
        "x": variable("t1"),
        "z": variable("t2"),
        "y": variable("t1t2")
    }
    code = "result = f'List {l}'"
    result = generate(ast.parse(code).body[0], Function(variables={
        "l": variable(variable_list("a", "b"), variable_list("c", "d"))
    }), ValueHandler())
    assert result == {
        "l": variable(variable_list("a", "b"), variable_list("c", "d")),
        "result": variable("List ['a', 'b']", "List ['c', 'd']")
    }
    code = "fun = f'{f(x1)}'"
    f = Function(transform=lambda x: {x})
    result = generate(ast.parse(code).body[0], Function(variables={
        "f": variable(f),
        "x1": variable("x_value")
    }), ValueHandler())
    assert result == {
        "f": variable(f),
        "x1": variable("x_value"),
        "fun": variable("x_value")
    }


def test_string_format():
    code = "left = 'test'.format()"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"left": variable("test")}
    code = "y = 'x{z}'.format(z='1')"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "y": variable("x1")
    }
    code = "result = '{x1}{x2}'.format(x1='y1', x2='y2')"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "result": variable("y1y2")
    }
    code = "r = 'a{x}b{y}c'.format(x=in_1, y=in_2)"
    result = generate(ast.parse(code).body[0], Function(variables={
        "in_1": variable("in1", "in2"),
        "in_2": variable("in3", "in4")
    }), ValueHandler())
    assert result == {
        "in_1": variable("in1", "in2"),
        "in_2": variable("in3", "in4"),
        "r": variable("ain1bin3c", "ain1bin4c", "ain2bin3c", "ain2bin4c")
    }
    code = "res = 'test{}'.format('x')"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"res": variable("testx")}
    code = "y = 'a{b}c{}'.format(d, b=x)"
    result = generate(ast.parse(code).body[0], Function(variables={
        "d": variable("d1", "d2"),
        "x": variable("x1", "x2")
    }), ValueHandler())
    assert result == {
        "d": variable("d1", "d2"),
        "x": variable("x1", "x2"),
        "y": variable("ax1cd1", "ax1cd2", "ax2cd1", "ax2cd2")
    }
    code = """res = x.format("test")"""
    result = generate(ast.parse(code).body[0], Function(variables={
        "x": variable("x1{}y1", "x2{}y2")
    }), ValueHandler())
    assert result == {
        "x": variable("x1{}y1", "x2{}y2"),
        "res": variable("x1testy1", "x2testy2")
    }


def test_generate_function_call():
    code = "a = x()"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"a": variable(Unknown())}
    code = "result = my_function()"
    result = generate(ast.parse(code).body[0], Function(variables={
        "y": variable(Unknown(), "1")
    }), ValueHandler())
    assert result == {"result": variable(Unknown()), "y": variable(Unknown(), "1")}
    code = "r = f()"
    f = Function(transform=lambda: {Constant(value="5")})
    result = generate(ast.parse(code).body[0], Function(variables={"f": variable(f)}), ValueHandler())
    assert result == {"r": variable("5"), "f": variable(f)}
    del f
    code = "r1, r2 = f1(), f2()"
    f1 = Function(transform=lambda: {Constant(value="result")})
    f2 = Function(transform=lambda: {Constant(value="r2")})
    result = generate(ast.parse(code).body[0], Function(variables={
        "f1": variable(f1),
        "f2": variable(f2)
    }), ValueHandler())
    assert result == {
        "r1": variable("result"),
        "r2": variable("r2"),
        "f1": variable(f1),
        "f2": variable(f2)
    }
    del f1, f2
    code = "res = fun(x1)"
    fun = Function(transform=lambda in_1: {in_1})
    result = generate(ast.parse(code).body[0], Function(variables={
        "fun": variable(fun),
        "x1": variable("54")
    }), ValueHandler())
    assert result == {
        "res": variable("54"),
        "fun": variable(fun),
        "x1": variable("54")
    }
    del fun
    code = "p = fun2(p1)"
    f = Function(transform=lambda x: {x})
    result = generate(ast.parse(code).body[0], Function(variables={
        "fun2": variable(f),
        "p1": variable("p43", "p44")
    }), ValueHandler())
    assert result == {
        "fun2": variable(f),
        "p1": variable("p43", "p44"),
        "p": variable("p43", "p44"),
    }
    del f
    code = "y = x(o)"
    f1 = Function(transform=lambda x: {x})
    f2 = Function(transform=lambda x: {Constant(value="5")})
    result = generate(ast.parse(code).body[0], Function(variables={
        "x": variable(f1, f2),
        "o": variable("t", "7")
    }), ValueHandler())
    assert result == {
        "x": variable(f1, f2),
        "o": variable("t", "7"),
        "y": variable("t", "7", "5")
    }
    del f1, f2
    code = "result = (f(), f(), f(), f())"
    f = Function(transform=lambda: {Constant(value="t1t2")})
    result = generate(ast.parse(code).body[0], Function(variables={"f": variable(f)}), ValueHandler())
    assert result == {
        "result": variable_list(variable("t1t2"), variable("t1t2"), variable("t1t2"), variable("t1t2")),
        "f": variable(f)
    }
    del f
    code = "p899 = (a(), i2, a(), i1)"
    a = Function(transform=lambda: {Constant(value="899")})
    result = generate(ast.parse(code).body[0], Function(variables={
        "a": variable(a),
        "i1": variable("24"),
        "i2": variable(Unknown())
    }), ValueHandler())
    assert result == {
        "p899": variable_list(variable("899"), Unknown(), variable("899"), variable("24")),
        "a": variable(a),
        "i1": variable("24"),
        "i2": variable(Unknown())
    }
    del a


def test_generate_parent_scope():
    parent = Function(variables={"p1": variable("test", "p2")})
    code = "p2 = p1"
    result = generate(ast.parse(code).body[0], Function(parent=parent), ValueHandler())
    assert result == {"p2": variable("test", "p2")}
    parent = Function(variables={"x": variable("1"), "y": variable("2")})
    code = "y = x"
    result = generate(ast.parse(code).body[0], Function(parent=parent), ValueHandler())
    assert result == {"y": variable("1")}
    parent2 = Function(variables={"x": variable("15")}, parent=parent)
    result = generate(ast.parse(code).body[0], Function(parent=parent2), ValueHandler())
    assert result == {"y": variable("15")}
    code = "result = y"
    result = generate(ast.parse(code).body[0], Function(parent=parent2), ValueHandler())
    assert result == {"result": variable("2")}
    f = Function(transform=lambda in_1: {in_1})
    parent = Function(variables={"x": variable(f)})
    code = """t24 = x("25")"""
    result = generate(ast.parse(code).body[0], Function(parent=parent), ValueHandler())
    assert result == {"t24": variable("25")}
    y = Function(transform=lambda: {Constant(value="test")})
    parent2 = Function(variables={"x": variable(y)}, parent=parent)
    code = "res = x()"
    result = generate(ast.parse(code).body[0], Function(parent=parent2), ValueHandler())
    assert result == {"res": variable("test")}


def test_kwargs():
    code = "result = f(x='res1')"

    def f_impl(x):
        return {x}
    f = Function(transform=f_impl)
    result = generate(ast.parse(code).body[0], Function(variables={"f": variable(f)}), ValueHandler())
    assert result == {"result": variable("res1"), "f": variable(f)}
    del f
    del f_impl
    code = "r = fun(x1=535, x2=23)"

    def fun_impl(x1, x2):
        return {create_list(x1, x2)}
    fun = Function(transform=fun_impl)
    result = generate(ast.parse(code).body[0], Function(variables={"fun": variable(fun)}), ValueHandler())
    assert result == {"r": variable_list("535", "23"), "fun": variable(fun)}
    del fun
    del fun_impl
    code = "result = my_function(35, second=75)"

    def my_function_impl(x, second):
        return {create_list(x, second)}
    my_function = Function(transform=my_function_impl)
    result = generate(ast.parse(code).body[0], Function(variables={
        "my_function": variable(my_function)
    }), ValueHandler())
    assert result == {"result": variable_list("35", "75"), "my_function": variable(my_function)}
    del my_function
    del my_function_impl
    code = "c = [15, y(c=c, a=b, b=a)]"

    def y_impl(c, a, b):
        return {create_list(b, a)}
    y = Function(transform=y_impl)
    result = generate(ast.parse(code).body[0], Function(variables={
        "y": variable(y),
        "a": variable("4523"),
        "b": variable("7"),
        "c": variable("28")
    }), ValueHandler())
    assert result == {
        "y": variable(y),
        "a": variable("4523"),
        "b": variable("7"),
        "c": variable_list(variable("15"), variable_list("4523", "7")),
    }
    del y
    del y_impl
    code = "r1 = r2(x1, x2=x3)"

    def r2_impl(x1, x2):
        return {x2}
    r2 = Function(transform=r2_impl)
    result = generate(ast.parse(code).body[0], Function(variables={
        "x1": variable("test"),
        "r2": variable(r2)
    }), ValueHandler())
    assert result == {
        "x1": variable("test"),
        "r2": variable(r2),
        "r1": variable(Unknown())
    }
    del r2
    del r2_impl
    code = "r = f(**test)"
    f = Function(transform=lambda y: {y})
    result = generate(ast.parse(code).body[0], Function(variables={
        "f": variable(f),
        "test": variable({
            variable("y"): variable("test2")
        })
    }), ValueHandler())
    assert result == {
        "f": variable(f),
        "test": variable({
            variable("y"): variable("test2")
        }),
        "r": variable("test2")
    }
    del f


def test_dict():
    code = "r1 = {}"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {"r1": variable({})}
    code = "y = {'a': 1, 'b': 2}"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "y": variable({
            variable("a"): variable("1"),
            variable("b"): variable("2")
        })
    }
    code = "result = {x1: y1, x2: y2, x3: y3}"
    result = generate(ast.parse(code).body[0], Function(variables={
        "x1": variable("x11", "x12"),
        "x2": variable("x21", "x22"),
        "x3": variable("x31", "x32"),
        "y1": variable("y11", "y12"),
        "y2": variable("y21", "y22"),
        "y3": variable("y31", "y32")
    }), ValueHandler())
    assert result == {
        "x1": variable("x11", "x12"),
        "x2": variable("x21", "x22"),
        "x3": variable("x31", "x32"),
        "y1": variable("y11", "y12"),
        "y2": variable("y21", "y22"),
        "y3": variable("y31", "y32"),
        "result": variable({
            variable("x11"): variable("y11", "y12"),
            variable("x12"): variable("y11", "y12"),
            variable("x21"): variable("y21", "y22"),
            variable("x22"): variable("y21", "y22"),
            variable("x31"): variable("y31", "y32"),
            variable("x32"): variable("y31", "y32"),
        })
    }
    code = "res = x['key']"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "res": variable(Unknown())
    }
    result = generate(ast.parse(code).body[0], Function(variables={
        "x": variable({
            variable("test"): variable("test2")
        })
    }), ValueHandler())
    assert result == {
        "x": variable({
            variable("test"): variable("test2")
        }),
        "res": variable(Unknown())
    }
    result = generate(ast.parse(code).body[0], Function(variables={
        "x": variable({
            variable("key"): variable("result"),
            variable("key2"): variable("result2")
        })
    }), ValueHandler())
    assert result == {
        "x": variable({
            variable("key"): variable("result"),
            variable("key2"): variable("result2")
        }),
        "res": variable("result")
    }
    code = "test = in_1[in_2]"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "test": variable(Unknown())
    }
    result = generate(ast.parse(code).body[0], Function(variables={
        "in_1": variable(Unknown),
        "in_2": variable("key")
    }), ValueHandler())
    assert result == {
        "in_1": variable(Unknown),
        "in_2": variable("key"),
        "test": variable()
    }
    result = generate(ast.parse(code).body[0], Function(variables={
        "in_1": variable({
            variable("key"): variable("result")
        }),
        "in_2": variable("key2")
    }), ValueHandler())
    assert result == {
        "in_1": variable({
            variable("key"): variable("result")
        }),
        "in_2": variable("key2"),
        "test": variable(Unknown())
    }
    result = generate(ast.parse(code).body[0], Function(variables={
        "in_1": variable({
            variable("x"): variable("r1"),
            variable("y"): variable("r2")
        }),
        "in_2": variable("x")
    }), ValueHandler())
    assert result == {
        "in_1": variable({
            variable("x"): variable("r1"),
            variable("y"): variable("r2")
        }),
        "in_2": variable("x"),
        "test": variable("r1")
    }
    result = generate(ast.parse(code).body[0], Function(variables={
        "in_1": variable({
            variable("a"): variable("b1", "b2"),
            variable("c"): variable("d")
        }),
        "in_2": variable("a", "c")
    }), ValueHandler())
    assert result == {
        "in_1": variable({
            variable("a"): variable("b1", "b2"),
            variable("c"): variable("d")
        }),
        "in_2": variable("a", "c"),
        "test": variable("b1", "b2", "d")
    }
    code = "y['test'] = '5'"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "y": variable({
            variable("test"): variable("5")
        })
    }
    result = generate(ast.parse(code).body[0], Function(variables={
        "y": variable({
            variable("test"): variable("10", "20")
        })
    }), ValueHandler())
    assert result == {
        "y": variable({
            variable("test"): variable("5")
        })
    }
    code = "result[in_1] = in_2[x1]"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {}
    result = generate(ast.parse(code).body[0], Function(variables={
        "result": variable({
            variable("in1"): variable("out1"),
            variable("in2"): variable("out2"),
            variable("in3"): variable("out3"),
        }),
        "in_1": variable("in1", "in3"),
        "in_2": variable({
            variable("test"): variable("a", "b", "c")
        }),
        "x1": variable("test")
    }), ValueHandler())
    assert result == {
        "result": variable({
            variable("in1"): variable("a", "b", "c"),
            variable("in2"): variable("out2"),
            variable("in3"): variable("a", "b", "c"),
        }),
        "in_1": variable("in1", "in3"),
        "in_2": variable({
            variable("test"): variable("a", "b", "c")
        }),
        "x1": variable("test")
    }
    code = "t = {**a, **b}"
    result = generate(ast.parse(code).body[0], Function(variables={
        "a": variable({
            variable("x"): variable("1")
        }),
        "b": variable({
            variable("x"): variable("2")
        })
    }), ValueHandler())
    assert result == {
        "a": variable({
            variable("x"): variable("1")
        }),
        "b": variable({
            variable("x"): variable("2")
        }),
        "t": variable({
            variable("x"): variable("1", "2")
        })
    }


def test_multi_assignments():
    code = "x = y = z = 1"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "x": variable("1"),
        "y": variable("1"),
        "z": variable("1")
    }
    code = "x1 = x2 = x3 = 1, 2, 3"
    result = generate(ast.parse(code).body[0], Function(), ValueHandler())
    assert result == {
        "x1": variable_list("1", "2", "3"),
        "x2": variable_list("1", "2", "3"),
        "x3": variable_list("1", "2", "3")
    }
    code = "a = b = c = d = y"
    result = generate(ast.parse(code).body[0], Function(variables={
        "y": variable("test")
    }), ValueHandler())
    assert result == {
        "a": variable("test"),
        "b": variable("test"),
        "c": variable("test"),
        "d": variable("test"),
        "y": variable("test")
    }
