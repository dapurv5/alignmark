import fire
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_model(model_name):
    AutoModelForCausalLM.from_pretrained(model_name)
    AutoTokenizer.from_pretrained(model_name)
    return None  # No return value


if __name__ == "__main__":
    fire.Fire(download_model)
