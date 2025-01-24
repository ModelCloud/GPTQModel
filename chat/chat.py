import argparse
import json
from datetime import datetime

from colorama import Fore, init
from gptqmodel import GPTQModel

init(autoreset=True)


USER_PROMPT = "User >>> \n"
ASSISTANT_PROMPT = "Assistant >>> \n"

KEY_USER = 'user'
KEY_ASSISTANT = 'assistant'

ASSISTANT_HELLO = 'How can I help you?'
EXIT_MESSAGE = 'Exiting the program.'

MESSAGES = [
    {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."}
]

DEBUG = False


def load_model(model_path):
    print(Fore.BLUE + f"Loading model from `{model_path}` ...\n")
    model = GPTQModel.load(model_path)
    return model


def chat_prompt_progress(user_input, tokenizer):
    user_message = {"role": KEY_USER, "content": user_input}
    MESSAGES.append(user_message)
    input_tensor = tokenizer.apply_chat_template(MESSAGES, add_generation_prompt=True, return_tensors="pt")
    if DEBUG:
        debug(tokenizer)
    return input_tensor


def debug(tokenizer):
    print("********* DEBUG START *********")
    print("********* Chat Template info *********")
    print(tokenizer.apply_chat_template(MESSAGES, return_dict=False, tokenize=False, add_generation_prompt=True))
    print("********* DEBUG END *********")


def get_user_input():
    user_input = input(Fore.GREEN + USER_PROMPT)
    return user_input


def print_model_message(message):
    print(Fore.CYAN + f"{ASSISTANT_PROMPT}{message}\n")


def save_chat_history(chat_history, save_path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"chat_history_{timestamp}.json"
    if save_path is not None:
        filename = f"{save_path}/chat_history_{timestamp}.json"
    with open(filename, 'w') as file:
        json.dump(chat_history, file, indent=4, ensure_ascii=False)
    print(Fore.YELLOW + f"Chat history saved to '{filename}'.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a GPT model.")
    parser.add_argument('--model_path', type=str, help="Path to the model.")
    parser.add_argument('--save_chat_path', type=str, help="Path to save the chat history.")
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Print Debug Info')
    args = parser.parse_args()
    if args.model_path is None:
        raise ValueError("Model path is None, Please Set `--model_path`")
    DEBUG = args.debug

    model = load_model(args.model_path)

    print(Fore.CYAN + "Welcome to GPTQModel Chat Assistant!\n")
    print(Fore.YELLOW + "You can enter questions or commands as follows:\n")
    print(Fore.YELLOW + "1. Type your question for the model.\n")
    print(Fore.YELLOW + "2. Type 'exit' to quit the program.\n")
    print(Fore.YELLOW + "3. Type 'save' to save the chat history.\n")

    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    chat_history = []  # chat history

    print_model_message(ASSISTANT_HELLO)

    while True:
        user_input = get_user_input()

        if user_input.lower() == 'exit':
            print(Fore.RED + f"{EXIT_MESSAGE}\n")
            break
        elif user_input.lower() == 'save':
            save_chat_history(chat_history, args.save_chat_path)
        else:
            input_tensor = chat_prompt_progress(user_input, tokenizer)
            outputs = model.generate(
                input_ids=input_tensor.to(model.device),
                max_new_tokens=4096,
                pad_token_id=tokenizer.pad_token_id
            )
            assistant_response = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

            MESSAGES.append({"role": KEY_ASSISTANT, "content": assistant_response})
            chat_history.append({KEY_USER: user_input, KEY_ASSISTANT: assistant_response})

            print_model_message(assistant_response)
