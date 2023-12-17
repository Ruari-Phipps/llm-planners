from model.create_sim_objects import create_bins, create_objects, create_plates, create_given_objects
from model.run_model import perform_action, perform_action_encode_action
from model.models import ResNet18FiLMBert
from model.arm import SimPanda
from model.failure_check import Failure_checker
from gpt3.step_generator import StepGenerator

from os.path import dirname, realpath, join, exists
import torch
import numpy as np
import json

location = dirname(realpath(__file__))
SCENE_FILE = join(location, "model", 'scenes', 'can_scene_cube.ttt')
MODEL_PATH = join(location, "model", "models", "weights")

# Load simulator 
arm = SimPanda(SCENE_FILE, bottom=0.76, headless=False, vision_cam="wrist_cam")
arm.move_by_smooth(np.array([0.1, 0, 0.2]))

# Load model
model = ResNet18FiLMBert(c_in=4, c_out=4)
model.load_state_dict(torch.load(join(MODEL_PATH, "full_policy", "full_policy_pre_bert.model"), map_location=torch.device('cpu')))
model.eval()

# Load GPT3 API
sg = StepGenerator(join(location, "gpt3", "test_prompt_1.txt"), fail_prompt=join(location, "gpt3", "failprompt.txt"), engine="text-davinci-002")

# Load failure checker
fc = Failure_checker(join(MODEL_PATH, "classifier", "input_classifier.model"))

new = True
random_objects = True
facts = None
scenario_scene = {}
objects = []
plates = []
bins = []
while True:
    if new:
        new = False
        for obj in objects + plates + bins:
            obj.object.remove()

        if random_objects:       
            objects = create_objects(1, 5, low=[0.9, -0.15, 0.775], high=[1.2, 0.15, 0.775], gap=0.02, current_obj=[])
            plates = create_plates(0, 3,  low=[0.9, -0.15, 0.775], high=[1.2, 0.15, 0.775], gap=0.02, size=0.1, current_obj=objects)
            bins = create_bins(0, 3,  low=[0.9, -0.15, 0.775], high=[1.2, 0.15, 0.775], gap=0.05, size=0.1, current_obj=objects+plates)
            
        else:
            objects = create_given_objects(scenario_scene["objects"], low=[0.9, -0.15, 0.775], high=[1.2, 0.15, 0.775], gap=0.03)
            plates, bins = [], []
        grabable_objects = [obj for obj in objects if obj.type not in ["bin", "plate"]]

        arm.pr.step()
        context = ", ".join([obj.description for obj in objects+plates+bins])
    print("Context:", context)


    arm.reset()
    arm.straight_ee()
    arm.go_home()
    arm.move_by_smooth(np.array([0.1, 0, 0.2]))

    request = input("Enter request: ")
    if request == "new":
        new = True
        continue
    elif request == "random":
        new = True
        random_objects = True
        facts = None
        continue
    elif request == "scenario":
        new = True
        random_objects = False
        scenario = input("Enter scenario: ")
        if exists(join(location, "scenarios", f"{scenario}.json")):
            with open(join(location, "scenarios", f"{scenario}.json")) as f:
                scenario_scene = json.load(f)
            facts = ",\n".join(scenario_scene["facts"])
            if "additional example" in scenario_scene:
                facts += f"\n{scenario_scene['additional example']}"
        else:
            print("Scenario does not exist")

        continue

    elif request == "exit":
        break


    model_input = {"request": request, "context":context}
    if facts is not None:
        model_input["facts"] = facts
    actions = sg.generate_steps(model_input)
    print(actions)
    fail_count = 0
    previous_actions = []

    try:
        while True:
            done = False
            for action in actions:
                action = action.strip()
                previous_actions.append(action)
                if action.lower() in ["can't", "done", ""]:
                    done = True
                    break

                held_object = arm.get_gripped()
                held_object = held_object[0] if len(held_object) > 0 else None
                
                print("ACTION:", action)
                perform_action(arm, model, [action], grabable_objects)
                arm.move_by_smooth(np.array([0.1, 0, 0.2]))

                success, reason = fc.check_failure(action, arm, objects+plates+bins, held_object)
                if not success:
                    print(reason)
                    fail_count += 1
                    actions = sg.generate_steps(model_input, previous_actions + [f"Failure-{reason}"], fail=True)
                    print(actions)
                    break
            
            if done or success or fail_count >= 5:
                break
    except KeyboardInterrupt:
        pass
arm.shutdown()
    