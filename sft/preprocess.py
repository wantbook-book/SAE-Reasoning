# borrowed from deepseek r1
SAE_FINE_TUNE_SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>.
"""

def make_conv(example, dataset=None, tokenizer=None):
    if isinstance(example, str):
        example = [example]

    if "problem" not in example:
        if "question" in example:
            example["problem"] = example["question"]
        elif "prompt" in example:
            example["problem"] = example["prompt"]
        else:
            raise ValueError("No problem/question/prompt found in the example.")

    # Only use the final answer but in the thinking format
    if tokenizer is None:
        # combined_text = [
        #     f"{SAE_FINE_TUNE_SYSTEM_PROMPT}User:\n {problem}\n\nAssistant:\n <think> {answer} </think>\n<answer> Answer: {answer}</answer>"
        #     for problem, answer in zip(example["problem"], example["answer"])
        # ]
        combined_text = [
            f"<｜begin▁of▁sentence｜><｜User｜>\n{SAE_FINE_TUNE_SYSTEM_PROMPT+problem}<｜Assistant｜>\n<think> {answer}</think>\n<answer> Answer: {answer}</answer><｜end▁of▁sentence｜>"
            for problem, answer in zip(example["problem"], example["answer"])
        ]
    else:
        # Use tokenizer to format the conversation
        combined_text = []
        for problem, answer in zip(example["problem"], example["answer"]):
            messages = [
                {"role": "user", "content": SAE_FINE_TUNE_SYSTEM_PROMPT+problem},
                {"role": "assistant", "content": f"<think> {answer} </think>\n<answer> Answer: {answer}</answer>"}
            ]
            formatted_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            combined_text.append(formatted_text)

    return {
        "text": combined_text
    }