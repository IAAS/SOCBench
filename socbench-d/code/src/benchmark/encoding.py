import tiktoken

def get_encoding():
    return tiktoken.encoding_for_model("gpt-4o")


def count_token(text: str) -> int:
    encoding = get_encoding()
    return len(encoding.encode(text))
