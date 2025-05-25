import openai
import json
from pathlib import Path
from openapi_spec_validator.validation.exceptions import OpenAPIValidationError
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import openapi_spec_validator
from scipy.stats import truncnorm
import random
from ..benchmark import Benchmark, Query, Queryset
import logging

MODEL = "gpt-4o-2024-11-20"
NUMBER_OF_SERVICES = 5
NUMBER_OF_ENDPOINTS = 10
NUMBER_OF_QUERIES = 10
NUMBER_OF_RETRIES = 3
THRESHOLD = 0.8
ENDPOINTS_MY = 5
ENDPOINTS_SIGMA = 2


class Service(BaseModel):
    name: str
    description: str


class Services(BaseModel):
    services: list[Service]


class Endpoint(BaseModel):
    endpoint: str
    description: str


class Endpoints(BaseModel):
    endpoints: list[Endpoint]


class AdditionalEndpoints(BaseModel):
    additional_endpoints: list[str]


class LlmQuery(BaseModel):
    query: str
    endpoints: list[str]


class LlmQueryEmbedding(BaseModel):
    query: str
    query_embeddings: list[float]
    endpoints: list[str]


class LlmQueries(BaseModel):
    queries: list[LlmQueryEmbedding]


def load_file(path):
    with open(path, "r") as file:
        return file.read()


def save_file(path, content):
    with open(path, "w") as file:
        file.write(content)


client = openai.Client()


def query_openai_model_text(messages):
    chat_completion = client.chat.completions.create(
        messages=messages, model=MODEL, max_completion_tokens=16384, timeout=20
    )
    return chat_completion.choices[0].message.content


def query_openai_model_json_no_model(messages):
    chat_completion = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=messages,
        max_completion_tokens=16384,
    )
    return chat_completion.choices[0].message.content


def query_openai_model_json(input_text, model):
    chat_completion = client.beta.chat.completions.parse(
        model=MODEL,
        response_format=model,
        messages=[{"role": "user", "content": input_text}],
        max_completion_tokens=16384,
    )
    return chat_completion.choices[0].message.parsed


def _get_embeddings(query: str) -> list[float]:
    response = client.embeddings.create(input=query, model="text-embedding-3-large")
    return response.data[0].embedding


def create_services(
    domain_short, domain_long, parent_directory, template_directory
) -> list:
    path = f"{parent_directory}/services.json"
    if Path(path).exists():
        services = Services.model_validate_json(load_file(path))
    else:
        services_template = load_file(f"{template_directory}/services-template.txt")
        services = Services(services=[])
        while len(services.services) < NUMBER_OF_SERVICES:
            service_template_instance = services_template.format(
                domain_short=domain_short,
                domain_long=domain_long,
                number_of_services=5,
                previous_services="\n".join(
                    [
                        f"{service.name}: {service.description}"
                        for service in services.services
                    ]
                ),
            )
            save_file(
                f"{parent_directory}/services-query-{len(services.services):02d}.txt",
                service_template_instance,
            )
            new_services = query_openai_model_json(service_template_instance, Services)
            services.services.extend(new_services.services)
        services.services = services.services[:NUMBER_OF_SERVICES]
        save_file(path, services.model_dump_json(indent=4))
    return services.services


def create_endpoints(service_description, domain, parent_directory, template_directory):
    path = f"{parent_directory}/endpoints.json"
    if Path(path).exists():
        endpoints = Endpoints.model_validate_json(load_file(path))
    else:
        endpoints_template = load_file(f"{template_directory}/endpoints-template.txt")
        endpoints = Endpoints(endpoints=[])
        while len(endpoints.endpoints) < NUMBER_OF_ENDPOINTS:
            endpoint_template_instance = endpoints_template.format(
                service=service_description,
                domain=domain,
                number_of_endpoints=10,
                previous_endpoints="\n".join(
                    [
                        f"{endpoint.endpoint}: {endpoint.description}"
                        for endpoint in endpoints.endpoints
                    ]
                ),
            )
            save_file(
                f"{parent_directory}/endpoints-query-{len(endpoints.endpoints):02d}.txt",
                endpoint_template_instance,
            )
            new_endpoints = query_openai_model_json(
                endpoint_template_instance, Endpoints
            )
            endpoints.endpoints.extend(new_endpoints.endpoints)
        endpoints.endpoints = endpoints.endpoints[:NUMBER_OF_ENDPOINTS]
        save_file(path, endpoints.model_dump_json(indent=4))
    return endpoints.endpoints


def _format_endpoints(endpoints: list[Endpoint]) -> str:
    return "\n".join(
        [f"{endpoint.endpoint}: {endpoint.description}" for endpoint in endpoints]
    )


def _check_openapi(openapi, endpoints):
    missing_endpoints = []
    for endpoint in endpoints:
        [verb, path] = endpoint.endpoint.split(" ")
        if not openapi["paths"].get(path) or not openapi["paths"][path].get(
            verb.lower()
        ):
            missing_endpoints.append(endpoint.endpoint)
    if len(missing_endpoints) > 0:
        raise ValueError(
            f"The endpoint(s) \"{'", "'.join(missing_endpoints)}\" is/are missing in the OpenAPI. It should contain the endpoints \"{'", "'.join([endpoint.endpoint for endpoint in endpoints])}\""
        )
    count_endpoints = sum(
        [len(endpoint.values()) for endpoint in openapi["paths"].values()]
    )
    if count_endpoints != len(endpoints):
        raise ValueError(
            f"Number of endpoints in OpenAPI ({count_endpoints}) does not match the number of expected endpoints ({len(endpoints)})"
        )


def _judge_openapi(
    template_directory: str, parent_directory: str, openapi: str, domain: str, i: int
) -> None:
    openapi_check_template = load_file(
        f"{template_directory}/open-api-check-template.txt"
    ).format(openapi=openapi, domain=domain)
    save_file(
        f"{parent_directory}/openapi-check-{i:02d}-query.txt", openapi_check_template
    )
    response = query_openai_model_text(
        [{"role": "user", "content": openapi_check_template}]
    )
    path = f"{parent_directory}/openapi-check-{i:02d}.txt"
    save_file(path, response)
    if not (response.startswith("Valid") or response.startswith("**Valid**")):
        raise ValueError(f"Evaluating the OpenAPI leads to: {response}")


def _validate_openapi(
    openapi, endpoints, template_directory, parent_directory, domain, i
):
    try:
        openapi_spec_validator.validate(openapi)
    except OpenAPIValidationError as e:
        raise ValueError(e.message)
    except Exception as e:
        raise ValueError(str(e))
    _check_openapi(openapi, endpoints)
    _judge_openapi(
        template_directory, parent_directory, json.dumps(openapi, indent=4), domain, i
    )


def create_openapi(
    service_description, domain, endpoints, parent_directory, template_directory: str
):
    path = f"{parent_directory}/openapi.json"
    if Path(path).exists():
        return load_file(path)
    openapi_template = load_file(f"{template_directory}/open-api-template.txt").format(
        service=service_description,
        domain=domain,
        endpoints=_format_endpoints(endpoints),
    )
    save_file(f"{parent_directory}/openapi-query.txt", openapi_template)
    messages = [{"role": "user", "content": openapi_template}]
    openapi_response = query_openai_model_json_no_model(messages)
    save_file(f"{parent_directory}/openapi-00.json", openapi_response)
    openapi = json.loads(openapi_response)
    for i in range(NUMBER_OF_RETRIES):
        try:
            _validate_openapi(
                openapi, endpoints, template_directory, parent_directory, domain, i
            )
            break
        except ValueError as e:
            save_file(
                f"{parent_directory}/openapi-{i:02d}.json",
                json.dumps(openapi, indent=4),
            )
            logging.warning(f"Validation error: {str(e)}. Retry.")
            messages.append({"role": "assistant", "content": openapi_response})
            messages.append(
                {
                    "role": "user",
                    "content": f"Fix the following validation error: {str(e)}",
                }
            )
            openapi_response = query_openai_model_json_no_model(messages)
            save_file(f"{parent_directory}/openapi-{i+1:02d}.json", openapi_response)
            openapi = json.loads(openapi_response)
            if i == NUMBER_OF_RETRIES - 1:
                _validate_openapi(
                    openapi,
                    endpoints,
                    template_directory,
                    parent_directory,
                    domain,
                    i + 1,
                )
    result_openapi = json.dumps(openapi, indent=4)
    save_file(path, json.dumps(openapi, indent=4))
    return result_openapi


def _reformat_query(query: LlmQueryEmbedding) -> Query:
    return Query(query=query.query, solution=query.endpoints)


def _check_query_similarity(
    queries: list[LlmQueryEmbedding], new_query: LlmQueryEmbedding
) -> bool:
    if len(queries) == 0:
        return True
    similarity = max(
        [
            cosine_similarity([query.query_embeddings], [new_query.query_embeddings])[
                0
            ][0]
            for query in queries
        ]
    )
    return similarity < THRESHOLD


def _format_services(services: list[str]) -> str:
    return "\n\n".join([service for service in services])


def _determine_expected_endpoint(all_endpoints: list[Endpoint]) -> list[str]:
    assert len(all_endpoints) > 0
    a_trunc = 1
    b_trunc = len(all_endpoints)
    loc = ENDPOINTS_MY
    scale = ENDPOINTS_SIGMA
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
    count = truncnorm.rvs(a, b, loc=loc, scale=scale)
    return random.sample(all_endpoints, int(count))


def _create_query(
    template_directory: str,
    query_template: str,
    joined_services: str,
    all_endpoints: list[Endpoint],
    previous_queries: list[LlmQueryEmbedding],
) -> LlmQueryEmbedding:
    expected_endpoints = _determine_expected_endpoint(all_endpoints)
    instantiated_query_template = query_template.format(
        services=joined_services,
        expected_endpoints=expected_endpoints,
        previous_queries="\n".join([query.query for query in previous_queries]),
    )
    messages = [
        {
            "role": "system",
            "content": "Only return the query. Do not return anything else.",
        },
        {"role": "user", "content": instantiated_query_template},
    ]
    new_query = query_openai_model_text(messages)
    count = 0
    while count < NUMBER_OF_RETRIES:
        count += 1
        necessary_endpoints = _get_necessary_endpoints(
            template_directory, joined_services, new_query, expected_endpoints
        )
        additional_endpoints = list(set(necessary_endpoints) - set(expected_endpoints))
        missing_endpoints = list(set(expected_endpoints) - set(necessary_endpoints))
        if len(additional_endpoints) == 0 and len(missing_endpoints) == 0:
            count = 0
            break
        messages.append({"role": "assistant", "content": new_query})
        messages.append(
            {
                "role": "user",
                "content": f"The query misses the expected endpoints [{", ".join(missing_endpoints)}] and needs the additional endpoints [{", ".join(additional_endpoints)}]. The list of expected endpoints is [{", ".join(expected_endpoints)}]. Fix this.",
            }
        )
        logging.warning(messages[-1]["content"])
        new_query = query_openai_model_text(messages)
    if count == NUMBER_OF_RETRIES:
        raise RuntimeError(
            f"Could not create a query that fulfills the endpoints [{",".join(expected_endpoints)}]"
        )
    embeddings = _get_embeddings(new_query)
    return LlmQueryEmbedding(
        query=new_query, query_embeddings=embeddings, endpoints=expected_endpoints
    )


def _get_necessary_endpoints(
    template_directory: str, joined_services: str, query: str, endpoints: list[str]
) -> list[str]:
    additional_endpoints_template = load_file(
        f"{template_directory}/query-check-further-endpoints-template.txt"
    )
    necessary_endpoints_template = load_file(
        f"{template_directory}/query-check-necessary-template.txt"
    )
    additional_endpoints_template_instance = additional_endpoints_template.format(
        services=joined_services, query=query, endpoints="\n".join(endpoints)
    )
    additional_endpoints = query_openai_model_json(
        additional_endpoints_template_instance, AdditionalEndpoints
    )
    endpoints = list(set(endpoints) | set(additional_endpoints.additional_endpoints))
    necessary_endpoints_template_instance = necessary_endpoints_template.format(
        services=joined_services, query=query, endpoints="\n".join(endpoints)
    )
    necessary_endpoints = []
    messages = [
        {"role": "user", "content": necessary_endpoints_template_instance},
        {"role": "assistant", "content": "Ok, please provide me the first endpoint."},
    ]
    for endpoint in endpoints:
        messages.append({"role": "user", "content": endpoint})
        response = query_openai_model_text(messages)
        messages.append({"role": "assistant", "content": response})
        if response.startswith("Yes") or response.startswith("**Yes**"):
            necessary_endpoints.append(endpoint)

    return necessary_endpoints


def create_queries(
    services: list[str],
    all_endpoints: list[Endpoint],
    parent_directory: str,
    template_directory: str
) -> list[Query]:
    assert NUMBER_OF_QUERIES % 10 == 0
    path = f"{parent_directory}/queries.json"
    if Path(path).exists():
        queries = LlmQueries.model_validate_json(load_file(path)).queries
    else:
        joined_services = _format_services(services)
        queries_template = load_file(f"{template_directory}/query-template.txt")
        queries = []
        while len(queries) < NUMBER_OF_QUERIES:
            try:
                new_query = _create_query(
                    template_directory,
                    queries_template,
                    joined_services,
                    all_endpoints,
                    queries,
                )
                if _check_query_similarity(queries, new_query):
                    queries.append(new_query)
                else:
                    logging.warning(
                        f'Discarding query "{new_query.query}" due to similarity.'
                    )
            except RuntimeError as e:
                logging.warning(str(e))
        save_file(path, LlmQueries(queries=queries).model_dump_json(indent=4))
    return [_reformat_query(query) for query in queries]


def create_benchmark(root_directory: str, benchmark_name: str) -> Benchmark:
    domains = json.loads(load_file(f"{root_directory}/domains.json"))
    querysets = []
    for i, domain in enumerate(domains):
        parent_directory = (
            f'{root_directory}/{benchmark_name}/{i+1:02d}-{domain["name-short"]}'
        )
        domain_short = domain["name-short"]
        domain_long = f'{domain["name"]}: {domain["description"]}'
        logging.info(f'Processing {domain["name"]} [{i + 1}/{len(domains)}]')
        Path(parent_directory).mkdir(parents=True, exist_ok=True)
        services = create_services(
            domain_short, domain_long, parent_directory, root_directory
        )
        openapis = []
        all_endpoints = []
        for j, service in enumerate(services):
            logging.info(f"Processing {service.name} [{j + 1}/{len(services)}]")
            service_directory = f"{parent_directory}/{j+1:02d}-{service.name}"
            Path(service_directory).mkdir(parents=True, exist_ok=True)
            endpoints = create_endpoints(
                service.description, domain_long, service_directory, root_directory
            )
            all_endpoints.extend([endpoint.endpoint for endpoint in endpoints])
            openapi = create_openapi(
                service.description,
                domain_long,
                endpoints,
                service_directory,
                root_directory,
            )
            openapis.append(openapi)
        queries = create_queries(openapis, all_endpoints, parent_directory, root_directory)
        queryset = Queryset(name=domain_short, queries=queries, openapis=openapis)
        querysets.append(queryset)
    return Benchmark(name=benchmark_name, queries=querysets)
