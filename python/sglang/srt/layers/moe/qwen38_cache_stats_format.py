# SPDX-License-Identifier: Apache-2.0
"""Lossless column encoding of repeated dictionaries in cache-stat logs."""


def _same_json_value(left, right):
    # Python equality conflates bool/int/float values and signed floating zero.
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_json_value(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_json_value(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float):
        return left.hex() == right.hex()
    return left == right


def compact_stats(value):
    if isinstance(value, dict):
        return {key: compact_stats(item) for key, item in value.items()}
    if isinstance(value, list):
        if value and all(
            isinstance(row, dict) and row.keys() == value[0].keys() for row in value
        ):
            common, columns = {}, {}
            for key in value[0]:
                items = [row[key] for row in value]
                if all(_same_json_value(item, items[0]) for item in items):
                    common[key] = compact_stats(items[0])
                else:
                    columns[key] = [compact_stats(item) for item in items]
            return {
                "__qwen38_columns_v1__": len(value),
                "common": common,
                "columns": columns,
            }
        return [compact_stats(item) for item in value]
    return value


def expand_stats(value):
    if isinstance(value, dict):
        if "__qwen38_columns_v1__" in value:
            count = value["__qwen38_columns_v1__"]
            common = expand_stats(value["common"])
            columns = {
                key: [expand_stats(item) for item in items]
                for key, items in value["columns"].items()
            }
            return [
                {**common, **{key: items[row] for key, items in columns.items()}}
                for row in range(count)
            ]
        return {key: expand_stats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_stats(item) for item in value]
    return value
