def remove_prompt_from_response(
    data: dict,
    prompt_fieldname: str = "prompt",
    response_fieldname: str = "watermarked_text",
):
    if prompt_fieldname in data:
        data[response_fieldname] = (
            data[response_fieldname].replace(data[prompt_fieldname], "").strip()
        )
    return data


def remove_role_tags(
    data: dict,
    role_tags: list[str] = ["### Instruction:", "## Instruction:"],
    fieldname: str = "prompt",
):
    for role_tag in role_tags:
        if role_tag in data[fieldname]:
            data[fieldname] = data[fieldname].replace(role_tag, "").strip()
    return data


def prune_multiple_turns(
    data: dict,
    fieldname: str = "watermarked_text",
    role_tag: str = "### Instruction:",
):
    if role_tag in data[fieldname]:
        data[fieldname] = data[fieldname].split(role_tag)[0].strip()
    return data


def pick_first_k_blocks(data: dict, fieldname: str = "watermarked_text", k: int = 2):
    if "\n\n" in data[fieldname]:
        data[fieldname] = "\n".join(data[fieldname].split("\n\n")[:k])
    return data
