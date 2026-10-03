"""Generator data catalog, extracted without the legacy remote executor."""

def build_rich_data_catalog(binding_descriptors):
    """Render binding descriptors into a rich Python-comment-style catalog.

    Args:
        binding_descriptors: dict of serialized BindingDescriptor dicts,
            as produced by PersistentStateMemory.get_binding_descriptors_payload()
            (sourced from ToolContextModule's binding descriptors).

    Returns:
        A multi-line string suitable for inclusion in a coding LLM prompt.
    """
    if not binding_descriptors:
        return ""

    lines = ["# Available variables (already in scope, use directly):"]

    for _key, desc in binding_descriptors.items():
        key = desc.get("key", _key)
        source_tool = desc.get("source_tool", "")
        source_args = desc.get("source_args", {})
        fields = desc.get("fields", {})

        # Header line with provenance
        source_parts = []
        if source_tool:
            arg_str = ", ".join(
                f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                for k, v in source_args.items()
            )
            source_parts.append(
                f"{source_tool}({arg_str})" if arg_str else source_tool
            )
        source_suffix = f" -- from {source_parts[0]}" if source_parts else ""
        lines.append(f"")
        lines.append(f"# {key}{source_suffix}")

        # Field lines
        for fname, finfo in fields.items():
            if isinstance(finfo, dict):
                dtype = finfo.get("dtype", "")
                description = finfo.get("description", "")
                preview = finfo.get("preview", "")
            else:
                dtype = description = preview = ""

            parts = [f'#   ["{fname}"]: {dtype}']
            if description:
                parts[0] += f" -- {description}"
            lines.append(parts[0])
            if preview:
                lines.append(f"#     preview: {preview}")

    return "\n".join(lines)
