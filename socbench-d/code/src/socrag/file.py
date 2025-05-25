import os.path
from os.path import join
import json
import pathlib


def read_file(path):
    with open(path, "r") as file:
        return file.read()


def extract_endpoints(file: str) -> list[str]:
    specification = json.loads(file)
    endpoints = []
    for path, path_content in specification["paths"].items():
        for verb, endpoint_content in path_content.items():
            if str(verb).lower().startswith("x") or str(verb).lower() == "parameters" or not "description" in endpoint_content:
                continue
            endpoints.append(f"{verb.upper()} {path}")
    return endpoints


def read_all_endpoints() -> list[str]:
    return extract_endpoints(
        read_file("./benchmark/specs/spotify_oas.json")
    ) + extract_endpoints(read_file("./benchmark/specs/tmdb_oas.json"))


def write_file(filename, data):
    parent_directory = join("data", os.path.split(filename)[0])
    pathlib.Path(parent_directory).mkdir(parents=True, exist_ok=True)
    with open(join("data", filename), "w") as file:
        file.write(json.dumps(data, indent=4))
        file.close()
