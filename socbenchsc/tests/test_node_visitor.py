import ast
import logging
from socbenchsc.models import FunctionCall, Unknown, RequestImport
from socbenchsc.node_visitor import AnalysisVisitor
from tests.variable import variable
import pytest


def perform_analysis(code):
    visitor = AnalysisVisitor()
    tree = ast.parse(code)
    visitor.visit(tree)
    return visitor


def evaluate_scope(code):
    visitor = perform_analysis(code)
    return visitor.scope, visitor.value_handler.function_calls


def test_import_parsing_none(caplog):
    code = ""
    visitor = perform_analysis(code)
    assert visitor.request_methods == set()
    assert caplog.record_tuples == []


def test_import_parsing_regular(caplog):
    code = "import requests"
    visitor = perform_analysis(code)
    assert visitor.request_methods == set()
    assert caplog.record_tuples == []


def test_import_parsing_from(caplog):
    code = ""
    visitor = perform_analysis(code)
    assert visitor.request_methods == set()
    code = "from requests import get"
    visitor = perform_analysis(code)
    assert visitor.request_methods == {"get"}
    code = "from requests import post, put"
    visitor = perform_analysis(code)
    assert visitor.request_methods == {"post", "put"}
    code = """from requests import delete, options
import requests as p
"""
    visitor = perform_analysis(code)
    assert visitor.request_methods == {"delete", "options"}
    assert caplog.record_tuples == []


def test_no_arguments(caplog):
    code = ""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    code = """requests.get()"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [FunctionCall(name="requests.get")]
    code = """requests.post()"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [FunctionCall(name="requests.post")]
    code = """requests.post()
requests.get()"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="requests.post"),
        FunctionCall(name="requests.get")]
    code = """post()
get()"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="post"),
        FunctionCall(name="get")]
    assert caplog.record_tuples == []


def test_simple_function_call(caplog):
    code = """requests.get("https://example.com/api/users")"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [FunctionCall(name="requests.get", arguments=[variable("https://example.com/api/users")])]
    code = """requests.post("http://blogsite.org/posts/latest")"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="requests.post", arguments=[variable("http://blogsite.org/posts/latest")])
    ]
    code = """requests.post("http://blogsite.org/posts/latest")
requests.get("https://example.com/api/users")"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="requests.post", arguments=[variable("http://blogsite.org/posts/latest")]),
        FunctionCall(name="requests.get", arguments=[variable("https://example.com/api/users")])
    ]
    code = """post("http://blogsite.org/posts/latest")
get("https://example.com/api/users")"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="post", arguments=[variable("http://blogsite.org/posts/latest")]),
        FunctionCall(name="get", arguments=[variable("https://example.com/api/users")])
    ]
    assert caplog.record_tuples == []
    code = """post("http://blogsite.org/posts/latest").json()
get("https://example.com/api/users").result
result = put("https://example.com/api/users").result"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="post", arguments=[variable("http://blogsite.org/posts/latest")]),
        FunctionCall(name="json"),
        FunctionCall(name="get", arguments=[variable("https://example.com/api/users")]),
        FunctionCall(name="put", arguments=[variable("https://example.com/api/users")])
    ]
    assert caplog.record_tuples == [
        ("root", logging.WARNING, "Attribute calls are not supported yet"),
        ("root", logging.WARNING, "Extraction from attributes is not supported yet")
    ]


def test_import_function_call(caplog):
    code = """from requests import post, put
post("http://blogsite.org/posts/latest")
put("https://example.com/api/users")"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="post", arguments=[variable("http://blogsite.org/posts/latest")]),
        FunctionCall(name="put", arguments=[variable("https://example.com/api/users")]),
    ]
    code = """import requests as r
r.get("https://example.com/api/users")"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="requests.get", arguments=[variable("https://example.com/api/users")]),
    ]
    assert caplog.record_tuples == []


def test_string_format(caplog):
    code = """url = "https://example.com/api/users/{}".format("123456")
url2 = "https://shopnow.io/products/{x}".format(x="view")
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name='"https://example.com/api/users/{}".format', arguments=[variable("123456")]),
        FunctionCall(name='"https://shopnow.io/products/{x}".format', kwargs={
            "x": variable("view")
        })
    ]
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "url": variable("https://example.com/api/users/123456"),
        "url2": variable("https://shopnow.io/products/view")
    }
    assert caplog.record_tuples == []
    code = """stock_level_url = "https://api.pharma-inventory.com/stock-levels/{item-id}"
requests.get(stock_level_url.format(item_id=item_id))"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name='"https://api.pharma-inventory.com/stock-levels/{item-id}".format', kwargs={
            "item_id": variable(Unknown())
        }),
        FunctionCall(name="requests.get", arguments=[
            variable("https://api.pharma-inventory.com/stock-levels/{item-id}")
        ])
    ]
    assert caplog.record_tuples == [
        ("root", logging.WARNING, "KeyError in string.format")
    ]
    caplog.clear()


def test_assignment(caplog):
    code = """x = "https://shopnow.io/products/view"
requests.get(x)
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [FunctionCall(name="requests.get", arguments=[
        variable("https://shopnow.io/products/view")])
    ]
    code = """url = "https://weatherapi.net/data/today"
requests.post(url)
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [FunctionCall(name="requests.post", arguments=[
        variable("https://weatherapi.net/data/today")])
    ]
    code = """url = "https://weatherapi.net/data/today"
requests.post(url)
x = "https://shopnow.io/products/view"
requests.get(x)
url = "http://newsportal.com/api/articles"
requests.get(url)
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="requests.post", arguments=[variable("https://weatherapi.net/data/today")]),
        FunctionCall(name="requests.get", arguments=[variable("https://shopnow.io/products/view")]),
        FunctionCall(name="requests.get", arguments=[variable("http://newsportal.com/api/articles")])
    ]
    code = """url = "https://weatherapi.net/data/today"
post(url)
x = "https://shopnow.io/products/view"
get(x)
url = "http://newsportal.com/api/articles"
get(url)
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="post", arguments=[variable("https://weatherapi.net/data/today")]),
        FunctionCall(name="get", arguments=[variable("https://shopnow.io/products/view")]),
        FunctionCall(name="get", arguments=[variable("http://newsportal.com/api/articles")])
    ]
    code = """result = requests.get("https://mybank.co/accounts/details")"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="requests.get", arguments=[variable("https://mybank.co/accounts/details")])]
    code = """x="https://photoshare.app/images/upload"
res = requests.post("http://travelhub.org/flights/search")
r2 = requests.get(x)
x = post(x)
x = post(x)"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="requests.post", arguments=[variable("http://travelhub.org/flights/search")]),
        FunctionCall(name="requests.get", arguments=[variable("https://photoshare.app/images/upload")]),
        FunctionCall(name="post", arguments=[variable("https://photoshare.app/images/upload")]),
        FunctionCall(name="post", arguments=[variable(Unknown())])
    ]
    code = """x = "http://travelhub.org/flights/search"
y: dict = get(x)
"""
    (scope, function_calls) = evaluate_scope(code)
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "x": variable("http://travelhub.org/flights/search"),
        "y": variable(Unknown()),
    }
    assert function_calls == [
        FunctionCall(name="get", arguments=[variable("http://travelhub.org/flights/search")])
    ]
    assert caplog.record_tuples == []


def test_function_definition(caplog):
    code = """def f():
    return 5
"""
    (scope, function_calls) = evaluate_scope(code)
    assert len(scope.variables) == 2
    assert "requests" in scope.variables
    assert "f" in scope.variables
    result = list(scope.variables["f"].values)
    assert len(result) == 1
    assert result[0].parent == scope
    code = """x = 2
def g2():
    return fun()
def fun():
    pass
"""
    (scope, function_calls) = evaluate_scope(code)
    assert len(scope.variables) == 4
    assert "requests" in scope.variables
    assert "x" in scope.variables and scope.variables["x"] == variable("2")
    assert "g2" in scope.variables
    result = list(scope.variables["g2"].values)
    assert len(result) == 1
    assert result[0].parent == scope
    assert "fun" in scope.variables
    result = list(scope.variables["fun"].values)
    assert len(result) == 1
    assert result[0].parent == scope
    code = """def x():
    return 3.14 + 8
x = -6
"""
    (scope, function_calls) = evaluate_scope(code)
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "x": variable("-6")
    }
    assert caplog.record_tuples == []


def test_function_scope(caplog):
    code = """def f1():
    return requests.get("https://moviesdb.com/api/v1/films")
def f2(x):
    return requests.get(x)
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    code = """def g1():
    return requests.get("https://moviesdb.com/api/v1/films")
def g2(x):
    return requests.get(x)
def g3(x1, x2):
    get(x1)
    return get(x2)
g1()
g3("https://shopnow.io/products/view", "https://newsportal.com/api/articles")
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="g1"),
        FunctionCall(name="requests.get", arguments=[variable("https://moviesdb.com/api/v1/films")]),
        FunctionCall(name="g3", arguments=[
            variable("https://shopnow.io/products/view"),
            variable("https://newsportal.com/api/articles")
        ]),
        FunctionCall(name="get", arguments=[variable("https://shopnow.io/products/view")]),
        FunctionCall(name="get", arguments=[variable("https://newsportal.com/api/articles")]),
    ]
    code = """url = "https://example.com/api/users"
def func():
    return requests.get(url)
func()
url = "https://shopnow.io/products/view"
func()
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="func"),
        FunctionCall(name="requests.get", arguments=[variable("https://example.com/api/users")]),
        FunctionCall(name="func"),
        FunctionCall(name="requests.get", arguments=[variable("https://shopnow.io/products/view")]),
    ]
    code = """resource = "https://weatherapi.net/data/today"
def f():
    resource = "https://learninghub.edu/courses/enroll"
    requests.post(resource)
f()
post(resource)
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="f"),
        FunctionCall(name="requests.post", arguments=[variable("https://learninghub.edu/courses/enroll")]),
        FunctionCall(name="post", arguments=[variable("https://weatherapi.net/data/today")]),
    ]
    code = """def a1(y):
    def a2(y):
        def a3(z):
            return z
        post(a3(y))
    a2("https://moviesdb.com/api/v1/films")
    a2(y)
a3("http://blogsite.org/posts/latest")
a1("https://shopnow.io/products/view")
a2("https://mybank.co/accounts/details")
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="a3", arguments=[variable("http://blogsite.org/posts/latest")]),
        FunctionCall(name="a1", arguments=[variable("https://shopnow.io/products/view")]),
        FunctionCall(name="a2", arguments=[variable("https://moviesdb.com/api/v1/films")]),
        FunctionCall(name="a3", arguments=[variable("https://moviesdb.com/api/v1/films")]),
        FunctionCall(name="post", arguments=[variable("https://moviesdb.com/api/v1/films")]),
        FunctionCall(name="a2", arguments=[variable("https://shopnow.io/products/view")]),
        FunctionCall(name="a3", arguments=[variable("https://shopnow.io/products/view")]),
        FunctionCall(name="post", arguments=[variable("https://shopnow.io/products/view")]),
        FunctionCall(name="a2", arguments=[variable("https://mybank.co/accounts/details")]),
    ]
    assert caplog.record_tuples == []


def test_function_scope_assignment(caplog):
    code = """def f():
    return 5
x = f()
"""
    (scope, function_calls) = evaluate_scope(code)
    assert "x" in scope.variables and scope.variables["x"] == variable("5")
    code = """def my_fun():
    return "test"
y = my_fun()
t1 = my_fun()
"""
    (scope, function_calls) = evaluate_scope(code)
    assert "y" in scope.variables and scope.variables["y"] == variable("test")
    assert "t1" in scope.variables and scope.variables["t1"] == variable("test")
    code = """def t1(x):
    return x
res = t1(541564)
"""
    (scope, function_calls) = evaluate_scope(code)
    assert "res" in scope.variables and scope.variables["res"] == variable("541564")
    code = """def t2(x1, x2):
    res = [x1, x2]
    return res
x2, x1 = t2(234, 6)
"""
    (scope, function_calls) = evaluate_scope(code)
    assert "x1" in scope.variables and scope.variables["x1"] == variable("6")
    assert "x2" in scope.variables and scope.variables["x2"] == variable("234")
    assert caplog.record_tuples == []


def test_function_scope_nested(caplog):
    code = """def f1():
    return "https://moviesdb.com/api/v1/films"
def f2(x):
    return x
res = f2(f1())
"""
    (scope, function_calls) = evaluate_scope(code)
    assert "res" in scope.variables and scope.variables["res"] == variable("https://moviesdb.com/api/v1/films")
    code = """def fun():
    return 12343214
def fff(x, y):
    var = (fun(), x, y)
    return fun()
def y(in_1):
    return in_1
r1 = fff(0, 2)
result = y(fff(0, 2))
result2 = fff(y(51), y("test"))
"""
    (scope, function_calls) = evaluate_scope(code)
    assert "r1" in scope.variables and scope.variables["r1"] == variable("12343214")
    assert "result" in scope.variables and scope.variables["result"] == variable("12343214")
    assert "result2" in scope.variables and scope.variables["result2"] == variable("12343214")
    assert caplog.record_tuples == []


def test_string_addition(caplog):
    code = """x = "https://shopnow.io/"
y = "products"
z = "users"
post(x + y)
url = x + z
get(url)
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="post", arguments=[variable("https://shopnow.io/products")]),
        FunctionCall(name="get", arguments=[variable("https://shopnow.io/users")]),
    ]
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "x": variable("https://shopnow.io/"),
        "y": variable("products"),
        "z": variable("users"),
        "url": variable("https://shopnow.io/users"),
    }
    code = """base = "https://shop.com"
i = "items"
a = "accounts"
requests.get(f'{base}/{i}')
url = f'{base}/{a}'
requests.post(url)
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="requests.get", arguments=[variable("https://shop.com/items")]),
        FunctionCall(name="requests.post", arguments=[variable("https://shop.com/accounts")]),
    ]
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "base": variable("https://shop.com"),
        "i": variable("items"),
        "a": variable("accounts"),
        "url": variable("https://shop.com/accounts"),
    }
    assert caplog.record_tuples == []


def test_kwargs(caplog):
    code = """def call(url, data):
    pass
url = "https://weatherapi.net/data/today"
call(url="https://shopnow.io/products/view", data='test')
call(data='qwerty', url=url)
call(url="https://mybank.co/accounts/details")
call("https://moviesdb.com/api/v1/films", data="url2")
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="call", kwargs={
            "url": variable("https://shopnow.io/products/view"),
            "data": variable("test")
        }),
        FunctionCall(name="call", kwargs={
            "url": variable("https://weatherapi.net/data/today"),
            "data": variable("qwerty")
        }),
        FunctionCall(name="call", kwargs={"url": variable("https://mybank.co/accounts/details")}),
        FunctionCall(name="call", arguments=[variable("https://moviesdb.com/api/v1/films")], kwargs={
            "data": variable("url2")
        }),
    ]
    assert len(scope.variables) == 3
    assert "requests" in scope.variables
    assert "call" in scope.variables and len(scope.variables["call"].values) == 1
    assert "url" in scope.variables and scope.variables["url"] == variable("https://weatherapi.net/data/today")
    assert caplog.record_tuples == [("root", logging.WARNING, "Input arguments do not match the function signature")]
    caplog.clear()
    code = """def invoke_service(a, b):
    def invoke():
        requests.post(a, data=b)
    invoke()
post(url="https://weatherapi.net/data/today", data='payload')
invoke_service(a="https://shopnow.io/products/view", b='a')
invoke_service('url')
invoke_service(b='testb', a='https://moviesdb.com/api/v1/films')
invoke_service('https://mybank.co/accounts/details', 'parameter')
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="post", kwargs={
            "url": variable("https://weatherapi.net/data/today"),
            "data": variable("payload")
        }),
        FunctionCall(name="invoke_service", kwargs={
            "a": variable("https://shopnow.io/products/view"),
            "b": variable("a")
        }),
        FunctionCall(name="invoke", kwargs={}),
        FunctionCall(name="requests.post", arguments=[variable("https://shopnow.io/products/view")], kwargs={
            "data": variable("a")
        }),
        FunctionCall(name="invoke_service", arguments=[variable("url")], kwargs={}),
        FunctionCall(name="invoke", kwargs={}),
        FunctionCall(name="requests.post", arguments=[variable("url")], kwargs={"data": variable(Unknown())}),
        FunctionCall(name="invoke_service", kwargs={
            "a": variable("https://moviesdb.com/api/v1/films"),
            "b": variable("testb")
        }),
        FunctionCall(name="invoke", kwargs={}),
        FunctionCall(name="requests.post", arguments=[variable("https://moviesdb.com/api/v1/films")], kwargs={
            "data": variable("testb")
        }),
        FunctionCall(name="invoke_service", arguments=[
            variable("https://mybank.co/accounts/details"), variable("parameter")], kwargs={}
        ),
        FunctionCall(name="invoke", kwargs={}),
        FunctionCall(name="requests.post", arguments=[variable("https://mybank.co/accounts/details")],
                     kwargs={"data": variable("parameter")})
    ]
    assert caplog.record_tuples == [("root", logging.WARNING, "Input arguments do not match the function signature")]
    caplog.clear()


def test_if(caplog):
    code = """if True:
    url = "https://shopnow.io/products/view"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "url": variable("https://shopnow.io/products/view")
    }
    code = """if False:
    data = "https://moviesdb.com/api/v1/films"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "data": variable("https://moviesdb.com/api/v1/films")
    }
    code = """if a == b:
    a = "b"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "a": variable("b")
    }
    code = """url = "test"
if True:
    url = "https://mybank.co/accounts/details"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "url": variable("test", "https://mybank.co/accounts/details")
    }
    code = """d = "data"
if b:
    test = d
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "d": variable("data"),
        "test": variable("data")
    }
    code = """a = "initial"
if a == []:
    a = url
    a = "abc"
    a = "https://shopnow.io/"
    a = a
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "a": variable("initial", "https://shopnow.io/")
    }
    code = """if result == False:
    result = in_1
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "result": variable(Unknown())
    }
    code = """if True:
    url = "https://weatherapi.net/data/today"
url = "abc"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "url": variable("abc")
    }
    assert caplog.record_tuples == []


def test_if_else(caplog):
    code = """if True:
    url = "https://shopnow.io/products/view"
else:
    url = "https://weatherapi.net/data/today"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "url": variable("https://shopnow.io/products/view", "https://weatherapi.net/data/today")
    }
    code = """if False:
    data = "https://moviesdb.com/api/v1/films"
    a = "b"
else:
    a = "a"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "data": variable("https://moviesdb.com/api/v1/films"),
        "a": variable("a", "b")
    }
    code = """in_1 = 5
if a == b:
    in_1 = "test"
else:
    in_1 = 123
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "in_1": variable("test", "123")
    }
    code = """result = 7
if result == False:
    result = 8
    result = "test2"
else:
    t = 0
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "result": variable("test2", "7"),
        "t": variable("0")
    }
    code = """url = "https://shopnow.io/"
if test:
    url = "in_1"
else:
    url = "123"
url = "https://mybank.co/accounts/details"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "url": variable("https://mybank.co/accounts/details")
    }
    assert caplog.record_tuples == []


def test_elif(caplog):
    code = """url = "123"
if True:
    url = "https://mybank.co/accounts/details"
elif a == b:
    url = "https://shopnow.io/products/view"
else:
    url = "https://weatherapi.net/data/today"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "url": variable(
            "https://mybank.co/accounts/details",
            "https://shopnow.io/products/view",
            "https://weatherapi.net/data/today"
        )
    }
    code = """data = 6
if test:
    data = "https://moviesdb.com/api/v1/films"
elif b:
    data = "abc"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "data": variable("6", "https://moviesdb.com/api/v1/films", "abc")
    }
    assert caplog.record_tuples == []


@pytest.mark.parametrize("loop_head", [
    "while True:",
    "while a == b:",
    "while result == False:",
    "while abc:",
    "for i in range(5):",
    "for x in y:",
    "for (a, b) in zip(d, e):"
])
def test_loop(loop_head: str, caplog):
    caplog.set_level(logging.INFO)
    code = f"""{loop_head}
    url = "https://shopnow.io/products/view"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "url": variable("https://shopnow.io/products/view")
    }
    assert caplog.get_records(when="call") == []
    code = f"""test = "input"
{loop_head}
    test = "https://moviesdb.com/api/v1/films"
    data = 123
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "test": variable("input", "https://moviesdb.com/api/v1/films"),
        "data": variable("123")
    }
    assert caplog.record_tuples == []
    code = f"""{loop_head}
    result = in_1
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "result": variable(Unknown())
    }
    assert caplog.record_tuples == []
    code = f"""a = "b"
{loop_head}
    a = a
    a = "c"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "a": variable("b", "c")
    }
    assert caplog.record_tuples == []
    code = f"""{loop_head}
    a = 1
a = 2
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "a": variable("2")
    }
    assert caplog.record_tuples == []
    code = f"""in_1 = "a"
{loop_head}
    in_1 = in_1 + "b"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "in_1": variable("a", "ab", "abb", "abbb", "abbbb", "abbbbb")
    }
    assert caplog.record_tuples == [("root", logging.WARNING, "While loop terminated after 5 iterations")]


def test_try_except(caplog):
    code = """try:
    url = "https://example.com/api/users"
    res = requests.get(url)
except Exception as e:
    res = "test"
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name="requests.get", arguments=[variable("https://example.com/api/users")], kwargs={}),
    ]
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "url": variable("https://example.com/api/users"),
        "res": variable(Unknown(), "test")
    }
    code = """try:
    a = 5
except NotImplementedError as e:
    a = 7
except Exception as e:
    a = 6
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "a": variable("5", "6", "7")
    }
    assert caplog.record_tuples == []


def test_unsupported_binop(caplog):
    code = """a = 5
b = 6
c = a - b"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "a": variable("5"),
        "b": variable("6"),
        "c": variable(Unknown())
    }
    assert caplog.record_tuples == [("root", logging.WARNING, "Unsupported binary operator Sub")]
    caplog.clear()
    code = """x = 10
y = 1
y1 = y * x"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == []
    assert scope.variables == {
        "requests": variable(RequestImport()),
        "x": variable("10"),
        "y": variable("1"),
        "y1": variable(Unknown())
    }
    assert caplog.record_tuples == [("root", logging.WARNING, "Unsupported binary operator Mult")]
    caplog.clear()


def test_recursion(caplog):
    code = """def a():
    return a()
r = a()
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [FunctionCall(name='a')]
    assert "a" in scope.variables
    assert "r" in scope.variables and scope.variables["r"] == variable(Unknown())
    assert caplog.record_tuples == [
        ("root", logging.WARNING, "Call stack contains recursive function name \"a\" -> Skip")
    ]
    caplog.clear()
    code = """def y1():
    return y2()
def y2():
    return y1()
r1 = y1()
r2 = y2()
"""
    (scope, function_calls) = evaluate_scope(code)
    assert function_calls == [
        FunctionCall(name='y1'),
        FunctionCall(name='y2'),
        FunctionCall(name='y2'),
        FunctionCall(name='y1'),
    ]
    assert "y1" in scope.variables
    assert "y2" in scope.variables
    assert "r1" in scope.variables and scope.variables["r1"] == variable(Unknown())
    assert "r2" in scope.variables and scope.variables["r2"] == variable(Unknown())
    assert caplog.record_tuples == [
        ("root", logging.WARNING, "Call stack contains recursive function name \"y1\" -> Skip"),
        ("root", logging.WARNING, "Call stack contains recursive function name \"y2\" -> Skip")
    ]
    caplog.clear()
