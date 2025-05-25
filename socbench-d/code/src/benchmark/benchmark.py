from pydantic import BaseModel


class Query(BaseModel):
    query: str
    solution: list[str]


class Queryset(BaseModel):
    name: str
    queries: list[Query]
    openapis: list[str]


class Benchmark(BaseModel):
    name: str
    queries: list[Queryset]
