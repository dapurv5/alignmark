def remove_prompt_from_response(
    data: dict,
    prompt_fieldname: str = "prompt",
    response_fieldname: str = "watermarked_text",
):
    if prompt_fieldname in data:
        if isinstance(data[response_fieldname], list):
            data[response_fieldname] = [
                text.replace(data[prompt_fieldname], "").strip()
                for text in data[response_fieldname]
            ]
        else:
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
    if isinstance(data[fieldname], list):
        data[fieldname] = [text.split(role_tag)[0].strip() for text in data[fieldname]]
    else:
        data[fieldname] = data[fieldname].split(role_tag)[0].strip()
    return data


def pick_first_k_blocks(data: dict, fieldname: str = "watermarked_text", k: int = 2):
    if isinstance(data[fieldname], list):
        data[fieldname] = [
            "\n".join(text.split("\n\n")[:k]) for text in data[fieldname]
        ]
    else:
        data[fieldname] = "\n".join(data[fieldname].split("\n\n")[:k])
    return data


def leave_last_block(data: dict, fieldname: str = "watermarked_text"):
    if isinstance(data[fieldname], list):
        data[fieldname] = [
            "\n".join(text.split("\n\n")[:-1]) if text.strip()[-1] != "." else text
            for text in data[fieldname]
        ]
    else:
        data[fieldname] = (
            "\n".join(data[fieldname].split("\n\n")[:-1])
            if data[fieldname].strip()[-1] != "."
            else data[fieldname]
        )
    return data


def remove_token(
    data: dict, fieldname: str = "watermarked_text", token: str = "&quot;"
):
    if isinstance(data[fieldname], list):
        data[fieldname] = [text.replace(token, "").strip() for text in data[fieldname]]
    else:
        data[fieldname] = data[fieldname].replace(token, "").strip()
    return data


def remove_last_incomplete_sentence(data: dict, fieldname: str = "watermarked_text"):
    if isinstance(data[fieldname], list):
        # first find position of last dot
        last_dot_positions = [text.rfind(".") for text in data[fieldname]]
        # then remove last incomplete sentence
        # make sure to keep the last dot
        data[fieldname] = [
            text[: last_dot_positions[i]].strip() + "."
            for i, text in enumerate(data[fieldname])
        ]
    else:
        # first find position of last dot
        last_dot_position = data[fieldname].rfind(".")
        # then remove last incomplete sentence
        # make sure to keep the last dot
        data[fieldname] = data[fieldname][:last_dot_position].strip() + "."
    return data


def cleanup(
    data: dict,
    role_tags: list[str],
    remove_tokens: list[str],
    prompt_field_name: str = "prompt",
) -> dict:
    """Clean up text data by applying a series of transformations.

    Applies the following cleanup steps to watermarked and unwatermarked text fields:
    1. Removes prompt text
    2. Removes role tags
    3. Prunes multiple conversation turns
    4. Keeps only the last text block
    5. Removes noise tokens

    Args:
        data: Dictionary containing text fields to clean up

    Returns:
        Dictionary with cleaned text fields
    """
    text_fields = [
        "watermarked_text",
        "unwatermarked_text",
        "watermarked_texts",
        "unwatermarked_texts",
    ]

    # Apply each cleanup step to all relevant text fields
    for field in text_fields:
        if field in data:
            # Remove prompt from response
            data = remove_prompt_from_response(data, prompt_field_name, field)

    # Remove role tags from prompt
    data = remove_role_tags(data, role_tags, prompt_field_name)

    # Clean up conversation structure
    for field in text_fields:
        if field in data:
            # Prune multiple turns for each role tag
            for role_tag in role_tags:
                data = prune_multiple_turns(data, field, role_tag)

            # Keep only last text block
            data = remove_last_incomplete_sentence(data, field)
            data = leave_last_block(data, field)

            # Remove noise tokens
            for token in remove_tokens:
                data = remove_token(data, field, token)

    return data
