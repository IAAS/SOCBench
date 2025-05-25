import json

if __name__ == "__main__":
    with open("src/benchmark/restbench/data/datasets/spotify.json") as file:
        spotify = json.loads(file.read())

    spotify_size = [len(element["solution"]) for element in spotify]
    print(min(spotify_size))
    print(max(spotify_size))

    with open("src/benchmark/restbench/data/datasets/tmdb.json") as file:
        tmdb = json.loads(file.read())

    tmdb_size = [len(element["solution"]) for element in tmdb]
    print(min(tmdb_size))
    print(max(tmdb_size))
