import pytest
import benchmark.socbenchd.create_benchmark as create_benchmark

def test_check_openapi_empty():
    create_benchmark._check_openapi({"paths": {}}, [])

def test_check_openapi_valid_single():
    openapi = {
        "paths": {
            "/": {
                "get": {
                    "summary": "description",
                }
            }
        }
    }
    endpoints = [create_benchmark.Endpoint(endpoint="GET /", description="description")]
    create_benchmark._check_openapi(openapi, endpoints)

def test_check_openapi_valid_double():
    openapi = {
        "paths": {
            "/": {
                "get": {
                    "summary": "main",
                },
                "post": {
                    "summary": "post",
                }
            }
        }
    }
    endpoints = [
        create_benchmark.Endpoint(endpoint="GET /", description="main"),
        create_benchmark.Endpoint(endpoint="POST /", description="post"),
    ]
    create_benchmark._check_openapi(openapi, endpoints)

def test_check_openapi_endpoint_missing():
    with pytest.raises(ValueError) as e:
        create_benchmark._check_openapi({"paths": {}}, [create_benchmark.Endpoint(endpoint="GET /", description="")])
    assert 'The endpoint(s) "GET /" is/are missing in the OpenAPI. It should contain the endpoints "GET /"' in str(e.value)
    with pytest.raises(ValueError) as e:
        create_benchmark._check_openapi({"paths": {"/": {}}}, [create_benchmark.Endpoint(endpoint="GET /", description="")])
    assert 'The endpoint(s) "GET /" is/are missing in the OpenAPI. It should contain the endpoints "GET /"' in str(e.value)
    with pytest.raises(ValueError) as e:
        create_benchmark._check_openapi({"paths": {"/": {"post": {"summary": ""}}}}, [create_benchmark.Endpoint(endpoint="GET /", description="")])
    assert 'The endpoint(s) "GET /" is/are missing in the OpenAPI. It should contain the endpoints "GET /"' in str(e.value)
    with pytest.raises(ValueError) as e:
        create_benchmark._check_openapi({"paths": {"/": {}}}, [create_benchmark.Endpoint(endpoint="GET /", description=""), create_benchmark.Endpoint(endpoint="POST /data", description="data")])
    assert 'The endpoint(s) "GET /", "POST /data" is/are missing in the OpenAPI. It should contain the endpoints "GET /", "POST /data"' in str(e.value)

def test_check_openapi_endpoint_count_missmatch():
    openapi = {
        "paths": {
            "/test": {
                "post": {
                    "summary": "update data",
                }
            }
        }
    }
    with pytest.raises(ValueError) as e:
        create_benchmark._check_openapi(openapi, [])
    assert "Number of endpoints in OpenAPI (1) does not match the number of expected endpoints (0)" in str(e.value)
