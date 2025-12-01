import ast
from socbenchsc.models import Unknown
from socbenchsc.node_visitor import AnalysisVisitor
from urllib.parse import urlparse


def parse_argument(argument: str) -> str | None:
    if not argument:
        return None
    try:
        url = urlparse(argument)
    except ValueError:
        return None
    if not url.scheme:
        return None
    if not url.path:
        return None
    path = url.path[1:]
    return f"/{path}"


VERBS = {"get", "options", "head", "post", "put", "patch", "delete"}
FUNCTION_CALLS = {f"requests.{verb}" for verb in VERBS}
FUNCTION_MAPPING = {f"requests.{verb}": verb.upper() for verb in VERBS}


class Analysis:
    def __init__(self, code):
        self.tree = ast.parse(code)
        self.visitor = AnalysisVisitor()
        self.request_names = FUNCTION_CALLS.copy()
        self.request_mapping = FUNCTION_MAPPING.copy()
        self.request = {"requests.request"}

    def perform_analysis(self):
        self.visitor.visit(self.tree)
        self.compute_request_names()
        return self.compute_sinks()

    def compute_request_names(self):
        # for request_import in self.visitor.requests:
        #     self.request_names = self.request_names | {f"{request_import}.{verb}" for verb in VERBS}
        #     self.request_mapping.update({f"{request_import}.{verb}": verb.upper() for verb in VERBS})
        #     self.request.add(f"{request_import}.request")
        for request_name in self.visitor.request_methods:
            self.request_names = self.request_names | {request_name for verb in VERBS}
            self.request_mapping.update({request_name: request_name.upper()})

    def compute_sinks(self):
        sinks = set()
        for function_call in self.visitor.get_function_calls():
            if function_call.name in self.request:
                urls, methods = extract_request(function_call)
            elif function_call.name in self.request_names:
                urls, methods = self.extract_attribute(function_call)
            else:
                continue
            if len(urls) == 0:
                continue
            for url in urls:
                for method in methods:
                    sinks.add(f"{method} {url}")
        return sinks

    def extract_attribute(self, function_call):
        urls = extract_urls(function_call, 0)
        method = [self.request_mapping[function_call.name]]
        return urls, method


def extract_request(function_call):
    urls = extract_urls(function_call, 1)
    if "method" in function_call.kwargs:
        method = [value.value.upper() for value in function_call.kwargs["method"].values if value != Unknown()]
    else:
        method = [value.value.upper() for value in function_call.arguments[0].values if value != Unknown()]
    return urls, method


def extract_urls(function_call, index):
    if "url" in function_call.kwargs:
        return extract_kwargs_url(function_call)
    elif len(function_call.arguments) >= 1:
        return extract_url_from(function_call.arguments, index)
    return []


def extract_kwargs_url(function_call):
    arguments = [parse_argument(value.value) for value in function_call.kwargs["url"].values if value != Unknown()]
    return filter_none(arguments)


def extract_url_from(arguments, index):
    if len(arguments) < index + 1:
        return []
    result_arguments = [parse_argument(value.value) for value in arguments[index].values if value != Unknown()]
    return filter_none(result_arguments)


def filter_none(arguments):
    return [argument for argument in arguments if argument]
