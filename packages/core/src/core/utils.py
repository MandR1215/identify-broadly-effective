import datetime
import json


def current_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def format_json_compact_lists(value: object, indent: int = 0) -> str:
    """Format JSON with scalar lists on one line."""
    if isinstance(value, dict):
        if len(value) == 0:
            return "{}"

        lines = ["{"]
        items = list(value.items())

        for index, (key, item) in enumerate(items):
            comma = "," if index < len(items) - 1 else ""
            item_text = format_json_compact_lists(item, indent=indent + 2)
            lines.append(
                f"{' ' * (indent + 2)}{json.dumps(key)}: {item_text}{comma}"
            )

        lines.append(f"{' ' * indent}}}")
        return "\n".join(lines)

    if isinstance(value, list):
        if all(not isinstance(item, dict | list) for item in value):
            return json.dumps(value)

        if len(value) == 0:
            return "[]"

        lines = ["["]

        for index, item in enumerate(value):
            comma = "," if index < len(value) - 1 else ""
            item_text = format_json_compact_lists(item, indent=indent + 2)
            lines.append(f"{' ' * (indent + 2)}{item_text}{comma}")

        lines.append(f"{' ' * indent}]")
        return "\n".join(lines)

    return json.dumps(value)