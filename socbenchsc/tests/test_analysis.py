from socbenchsc.analysis import Analysis
import pytest


def test_none():
    code = ""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == set()
    code = """requests.get()"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == set()
    code = """requests.post()
requests.head()
requests.options()"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == set()


@pytest.mark.parametrize("method,url,expected", [
    ("get",     "http://example.com/1234",            {"GET /1234"}),
    ("post",    "http://www.abc.eu/test",             {"POST /test"}),
    ("put",     "https://api.test.org/resource",      {"PUT /resource"}),
    ("delete",  "http://example.com/item/42",         {"DELETE /item/42"}),
    ("patch",   "https://api.service.com/v1/update",  {"PATCH /v1/update"}),
    ("options", "http://example.com/query",           {"OPTIONS /query"}),
    ("head",    "https://host.com/check",             {"HEAD /check"}),
])
def test_all_methods_single_call(method, url, expected):
    code = f'requests.{method}("{url}")'
    analysis = Analysis(code)
    assert analysis.perform_analysis() == expected
    code = f'requests.{method}(url="{url}")'
    analysis = Analysis(code)
    assert analysis.perform_analysis() == expected
    code = f'requests.request("{method.upper()}", "{url}")'
    analysis = Analysis(code)
    assert analysis.perform_analysis() == expected
    code = f'requests.request(method="{method.upper()}", url="{url}")'
    analysis = Analysis(code)
    assert analysis.perform_analysis() == expected


def test_multiple_calls():
    code = """
requests.get("http://a/b1")
requests.post(url="http://a/b2")
requests.delete("http://a/b3/c3/", "test")
requests.patch("http://a/b4/")
requests.options(url="http://a.bc/test/t2")
"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == {
        "GET /b1", "POST /b2", "DELETE /b3/c3/", "PATCH /b4/", "OPTIONS /test/t2"
    }


def test_assignment():
    code = """x = "https://test.org/abc"
requests.get(x)
"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == {"GET /abc"}
    code = """url = "http://example.com/endpoint"
requests.post(url)
"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == {"POST /endpoint"}


def test_rename_requests():
    code = """import requests as r
r.get("https://test.org/abc")
"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == {"GET /abc"}
    code = """import requests as a
import requests as b
import requests as c
b.post("http://example.com/endpoint")
a.put("https://test.org/def")
c.delete("http://example.com/item/42")
"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == {"POST /endpoint", "PUT /def", "DELETE /item/42"}
    code = """import requests as re
re.request("GET", "http://example.com/query")
"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == {"GET /query"}


def test_import_from():
    code = """from requests import get, post
get("https://api.service.com/v1/update")
post("http://www.abc.eu/test")
get("http://example.com/1234")
"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == {"GET /v1/update", "POST /test", "GET /1234"}
    code = """from requests import get, post, put
put("http://example.com/query")
"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == {"PUT /query"}


def test_unknown():
    code = """from requests import put
requests.request("GET", url)
requests.get(url2)
request.post(url=in_2)
put(url=in_3)
put(y)
"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == set()


def test_analysis_attribute():
    code = """from requests import get, post, put
result = requests.get("https://mybank.co/accounts/details").json()
d = get("http://test.com/abc").c
"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == {"GET /accounts/details", "GET /abc"}


def test_invalid_endpoints():
    code = """requests.get("/test")
requests.put(url="abc")
"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == set()


def test_unknown_parameter():
    code = """requests.get(url="http://example.com/1234", unknown="abc")"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == {"GET /1234"}


def test_session():
    code = """session = requests.Session()
session.post(url="https://example.com/endpoint")"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == {"POST /endpoint"}
    code = """import requests as r
s = r.Session()
s.get("http://test/test")
s.put(url="https://request.com/put")"""
    analysis = Analysis(code)
    assert analysis.perform_analysis() == {"GET /test", "PUT /put"}
