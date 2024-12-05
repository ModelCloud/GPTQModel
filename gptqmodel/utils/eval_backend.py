from enum import Enum


class EVAL(Enum):
    LM_EVAL = 0
    EVALPLUS = 1

    @classmethod
    def get_task_enums(cls):
        return [member for member in cls]

    @classmethod
    def get_full_name(cls, member):
        return f"{cls.__name__}.{member.name}"

    @classmethod
    def get_all_eval_backend_string(cls):
        full_names = [cls.get_full_name(member) for member in cls]
        return ', '.join(full_names)


class LM_EVAL_TASK(Enum):
    ARC_CHALLENGE = "arc_challenge"
    MMLU = "mmlu"
    HELLASWAG = "hellaswag"
    GSM8K_COT = "gsm8k_cot"

    @classmethod
    def get_task_enums(cls):
        return [member for member in cls]

    @classmethod
    def get_full_name(cls, member):
        return f"{cls.__name__}.{member.name}"

    @classmethod
    def get_all_tasks_string(cls):
        full_names = [cls.get_full_name(member) for member in cls]
        return ', '.join(full_names)


class EVALPLUS_TASK(Enum):
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"

    @classmethod
    def get_task_enums(cls):
        return [member for member in cls]

    @classmethod
    def get_full_name(cls, member):
        return f"{cls.__name__}.{member.name}"

    @classmethod
    def get_all_tasks_string(cls):
        full_names = [cls.get_full_name(member) for member in cls]
        return ', '.join(full_names)



