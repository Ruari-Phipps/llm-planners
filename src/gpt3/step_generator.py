import openai
from typing import List
from os.path import join
from .config import openai_api_key

openai.api_key = openai_api_key

class StepGenerator:
    def __init__(self, example_file, fail_prompt=None, engine="ada"):
        self.engine = engine
        with open(example_file, "r") as file:
            self.examples = file.read()
        if fail_prompt is not None:
            with open(fail_prompt, "r") as file:
                self.failprompt = file.read()
        else:
            self.failprompt = None


    def create_prompt(self, input:dict, steps:List[str]=[], fail=False) -> str:
        prompt = f"{self.examples}\n\n"
        if fail and self.failprompt is not None:
            prompt += f"{self.failprompt}\n\n"
        if "facts" in input:
            prompt += f"{input['facts']}\n\n"
        prompt += f"C: There is a {input['context']}\nR: {input['request']}\nS:"
        for step in steps:
            prompt += f" {step},"
        return prompt


    def generate_steps(self, input:dict, previous_steps:List[str] = [], fail=False) -> List[str]:
        prompt = self.create_prompt(input, previous_steps, fail)
        # print(prompt)
        result = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens = 250,
            stop=["\n", ", Done", ",Done"],
            temperature = 0,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0,
        )
        return result["choices"][0]["text"].split(", ")


    def generate_next_possible_next_step(self, input, previous_steps:List[str] = [], n=3):
        prompt = self.create_prompt(input, previous_steps)
        result = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            stop=[",", "\n"],
            temperature=0.2,
            max_tokens = 100,
            frequency_penalty = 0,
            presence_penalty = 0,
            n = n)
        top_n = [choice["text"] for choice in result["choices"]]
        return top_n


if __name__ == "__main__":
    facts = "white bin = mug, \nblack cube = coffee, \nwhite cube = sugar"
    context = "black bin, black cube, white cube, white bin"
    query = "Make a coffee with sugar"
    input = {"request": query, "context":context, "facts":facts}
    generator = StepGenerator(join("gpt3", "test_prompt_1.txt"), engine="text-davinci-002")
    steps = generator.generate_steps(input)
    print(steps)