def get_device_to_use(
    num_processes: int,
    num_gpus_per_process: int,
) -> str:
    if num_gpus_per_process == 0 and num_processes == 1:
        return "cpu"
    elif num_gpus_per_process == 0 and num_processes > 1:
        raise ValueError(
            "num_gpus_per_process must be greater than 0 when num_processes is greater than 1"
        )
    elif num_gpus_per_process > 0 and num_processes == 1:
        return "cuda"
    elif num_gpus_per_process > 0 and num_processes > 1:
        return "cuda"
    else:
        raise ValueError("Invalid input parameters")
