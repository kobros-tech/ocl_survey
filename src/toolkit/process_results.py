import os

import jsonlines
import pandas as pd
import seaborn as sns
import collections

NAME_MAPS = {
    "logs": "training",
    "logs_continual": "continual",
    "linear_probing/logs": "probing",
}


def map_name(name):
    biggest_candidate = None
    for candidate in NAME_MAPS:
        if candidate in str(name):
            if biggest_candidate is None:
                biggest_candidate = candidate
                continue
            if len(candidate) > len(biggest_candidate):
                biggest_candidate = candidate
    return NAME_MAPS.get(biggest_candidate)


def gather_json_lines(filename):
    json_lines = []
    with open(filename, "r") as fs:
        reader = jsonlines.Reader(fs)
        for line in reader:
            json_lines.append(line)
    return json_lines


def annotate_lines(json_lines, **kwargs):
    for line in json_lines:
        line.update(**kwargs)


def get_seed_from_path(path):
    """Find the member of path that is only composed of int"""
    for pm in path.split("/"):
        if pm.isdigit():
            return int(pm)


def extract_results(directory, verbose=True):
    """
    Extracts recognized JSON results from a directory.

    Files that do not match one of NAME_MAPS are ignored. This allows runs to
    contain auxiliary metadata (for example Skill Memory's audit JSON) without
    accidentally treating it as a benchmark result or crashing extraction.
    """

    all_lines = collections.defaultdict(list)
    for root, dirs, files in os.walk(directory):
        for f in files:
            if "json" not in f:
                continue

            path = os.path.join(root, f)
            name = map_name(path)
            if name is None:
                if verbose:
                    print(f"Skipping unrecognized JSON result: {path}")
                continue

            if verbose:
                print(path)
                print(name)

            seed = get_seed_from_path(root)
            new_lines = gather_json_lines(path)
            annotate_lines(new_lines, seed=seed, name=name)
            all_lines[name] += new_lines

    dataframes = {}
    for name in all_lines:
        dataframes[name] = pd.DataFrame(all_lines[name])

    return dataframes
