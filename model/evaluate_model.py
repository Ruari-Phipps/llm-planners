import time
import numpy as np
from os.path import join, dirname, realpath, exists

import numpy as np
from numpy.random import choice, randint

from arm import SimPanda
from bert_encoder import bert_encode
from create_sim_objects import create_objects, create_plates, create_bins, OBJECT_TYPES, COLOURS

import torch

from run_model import generate_new_objects, perform_action, perform_action_encode_action
from models import ResNet18FiLM, ResNet18FiLMBert, ResNet18

from action_sentences import place_by_sentences_inc, place_by_sentences_non_inc, place_on_sentences_inc, place_on_sentences_non_inc, grab_sentences, articles, place_in_sentences_inc, place_in_sentences_non_inc

def grab_object(arm, object, STEPS=6):
    target_pos = object.object.get_position()
    target_ori = object.object.get_orientation()

    target = target_pos + np.array([0.005, 0, object.size[2]/2 + 0.03])  #Select a target position above the object of interest.


    for i in range(60):
        # Calculate movement
        current_pos = arm.get_position()
        step = (target - current_pos)/STEPS
        distance_step = np.linalg.norm(step)
        delta_target = step

        if object.type in ["sphere", "cylinder"]:
            diff = 0
        else:
            current_ori = arm.get_gripper_orientation()
            rot = (target_ori + current_ori) / STEPS
            diff = -rot[2]

        # Stop?
        if distance_step > 0.0015 and i < 35:
            arm.move_by_smooth(delta_target)
        else:
            if abs(diff) < 0.001:
                break 

            arm.rotate_gripper_by([0, 0, diff])

    # Grab object
    arm.down_grasp(by=0.07, objects=[object.object])

    # Return to start
    arm.reset_gripper()
    arm.go_home()

if __name__ == "__main__":
    # Load simulator 
    location = dirname(realpath(__file__))
    SCENE_FILE = join(location, 'scenes', 'can_scene_cube.ttt')
    arm = SimPanda(SCENE_FILE, bottom=0.76, headless=True, vision_cam="wrist_cam")

    # Load Model
    MODEL_PATH = join(location, "models", "weights")
    model = ResNet18FiLMBert(c_in=4, c_out=4)
    # model = ResNet18FiLM(c_in = 4, c_out=4, film_size=768)
    # model = ResNet18(c_in = 4, c_out=4)
    model.load_state_dict(torch.load(join(MODEL_PATH, "full_policy", "full_policy_bert.model"), map_location=torch.device('cpu')))
    model.eval()

    need_encode=False

    # Set up initial scene
    objects = []
    plates = []
    bins=[]
    # objects, plates = generate_new_objects(objects=[], plates=[])
    arm.reset_gripper()
    arm.go_home()

    object_keys = {}
    for i, object in enumerate(OBJECT_TYPES):
        object_keys[object['type']] = i
    object_keys['plate'] = i + 1
    object_keys["bin"] = i + 2

    colour_keys = {}
    for i, colour in enumerate(COLOURS):
        colour_keys[colour] = i

    ACTIONS = 250
    name = "full_bert"
    start = 0

    eval_type = "place on"

    # Grabbing eval
    if eval_type == "grab":
        if exists(join(location, "models", "eval_info", "grab", f"{name}_ob.npy")):
            ob_results = np.load(join(location, "models", "eval_info", "grab", f"{name}_ob.npy"))
            col_results = np.load(join(location, "models", "eval_info", "grab", f"{name}_col.npy"))
            tot_n = np.load(join(location, "models", "eval_info", "grab", f"{name}_tot.npy"))
            start = int(np.sum(ob_results))
        else:
            ob_results = np.zeros((len(OBJECT_TYPES), 3))
            col_results = np.zeros((len(COLOURS), 3))
            tot_n = np.zeros(1)

        print("\n===========\nGRABING EVAL\n===========\n")
        for a in range(start, ACTIONS):
            print(f"\n---------\nACTION {a+1}/{ACTIONS}")
            # Find a random place to go
            start = np.random.uniform(low=[-0.05, -0.05, 0.2], high=[0.05, 0.05, 0.3], size=3)
            # Go to random pose
            arm.reset()
            arm.move_by_smooth(start)
            arm.straight_ee()

            #objects, plates = generate_new_objects(objects, plates)
            for object in objects+plates+bins:
                object.object.remove()
            objects = create_objects(1, 6, low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.02, current_obj=[])
            
            if randint(0,2):
                plates = create_plates(1, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.02, size=0.1, current_obj=objects)
                bins = create_bins(0, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.05, size=0.1, current_obj=objects+plates)
            else:      
                bins = create_bins(1, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.05, size=0.1, current_obj=objects)
                plates = create_plates(0, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.02, size=0.1, current_obj=objects+bins)

            target_object = choice(objects)
            action = choice(grab_sentences).format(choice(articles), target_object.description)
            print("Action: Grab", target_object.description)
            if need_encode:
                perform_action_encode_action(arm, model, action, objects)
            else:
                perform_action(arm, model, [action], objects)
            
            gripped = arm.get_gripped()

            if target_object.object in gripped:
                ob_results[object_keys[target_object.type], 0] += 1
                col_results[colour_keys[target_object.colour_name], 0] += 1
                tot_n += 1
                print("Success")
            elif len(gripped) == 0:
                # Failure: Nothing grabbed
                ob_results[object_keys[target_object.type], 2] += 1
                col_results[colour_keys[target_object.colour_name], 2] += 1
                print("Failure: Nothing grabbed")
            else:
                # Failure: Wrong object
                grabbed = [obj for obj in objects if obj.object in gripped][0]

                ob_results[object_keys[target_object.type], 1] += 1
                col_results[colour_keys[target_object.colour_name], 1] += 1
                print("Failure: Wrong Object")

            if len(gripped) > 0:
                arm.open_gripper()

            with open(join(location, "models", "eval_info", "grab", f"{name}_ob.npy"), 'wb') as f:
                np.save(f, ob_results)
            with open(join(location, "models", "eval_info", "grab", f"{name}_col.npy"), 'wb') as f:
                np.save(f, col_results)
            with open(join(location, "models", "eval_info", "grab", f"{name}_tot.npy"), 'wb') as f:
                np.save(f, tot_n)
            
            print(f"TOTAL ACCURACY: {100*tot_n/(a+1)}%")

    # Place on eval
    elif eval_type == "place on":
        ACTIONS = 150
        start = 0

        if exists(join(location, "models", "eval_info", "place_on", f"{name}_ob.npy")):
            ob_results = np.load(join(location, "models", "eval_info", "place_on", f"{name}_ob.npy"))
            col_results = np.load(join(location, "models", "eval_info", "place_on", f"{name}_col.npy"))
            tot_n = np.load(join(location, "models", "eval_info", "place_on", f"{name}_tot.npy"))
            start = int(np.sum(ob_results))
        else:
            ob_results = np.zeros((3, 3))
            col_results = np.zeros((len(COLOURS), 3))
            tot_n = np.zeros(1)

        object_keys_place_on = {"cube": 0, "slice": 1, "plate": 2}

        print("\n===========\nPLACE ON EVAL\n===========\n")
        for a in range(start, ACTIONS):
            print(f"\n---------\nACTION {a+1}/{ACTIONS}")
            arm.reset()
            arm.straight_ee()


            for object in bins:
                object.object.remove()
            objects, plates = generate_new_objects(objects, plates)
            bins = create_bins(0, 2,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.05, size=0.1, current_obj=objects+plates+bins)

            # Grab random object
            while len(arm.get_gripped()) == 0:
                arm.open_gripper()
                grabbed_object = choice(objects)
                grab_object(arm, grabbed_object)

            # Go to random pose
            start = np.random.uniform(low=[-0.05, -0.05, 0.1], high=[0.05, 0.05, 0.3], size=3)
            arm.reset()
            arm.move_by_smooth(start)
            arm.straight_ee()

            # Execute test

            target_object = choice(plates + [object for object in objects if object.type in ["cube", "slice"] and object != grabbed_object])
            if randint(0, 2):
                action = choice(place_on_sentences_non_inc).format(choice(articles), target_object.description)
            else:
                action = choice(place_on_sentences_inc).format(choice(articles), grabbed_object.description, choice(articles), target_object.description)

            print("Action: Place on", target_object.description)
            if need_encode:
                perform_action_encode_action(arm, model, action, objects)
            else:
                perform_action(arm, model, [action], objects)
            
            held_placed_location = grabbed_object.object.get_position()
            target_object_location = target_object.object.get_position()

            if ((held_placed_location[2] - grabbed_object.size[2]/2)  > (target_object_location[2]) and
                np.linalg.norm(held_placed_location[:2] - target_object_location[:2]) < np.sqrt(sum([i**2 for i in target_object.size[:2]])) / 2 + 0.02):
                # Success
                ob_results[object_keys_place_on[target_object.type], 0] += 1
                col_results[colour_keys[target_object.colour_name], 0] += 1
                tot_n += 1
                print("Success")

            elif np.linalg.norm(held_placed_location[:2] - target_object_location[:2]) < (np.sqrt(sum([i**2 for i in target_object.size[:2]]) / 2 + 0.07)):
                # Failure: To side
                ob_results[object_keys_place_on[target_object.type], 1] += 1
                col_results[colour_keys[target_object.colour_name], 1] += 1
                print("Failure: To side")

            else:
                # Failure: Wrong object
                ob_results[object_keys_place_on[target_object.type], 2] += 1
                col_results[colour_keys[target_object.colour_name], 2] += 1
                print("Failure: Wrong Object")

            with open(join(location, "models", "eval_info", "place_on", f"{name}_ob.npy"), 'wb') as f:
                np.save(f, ob_results)
            with open(join(location, "models", "eval_info", "place_on", f"{name}_col.npy"), 'wb') as f:
                np.save(f, col_results)
            with open(join(location, "models", "eval_info", "place_on", f"{name}_tot.npy"), 'wb') as f:
                np.save(f, tot_n)
            
            print(f"TOTAL ACCURACY: {100*tot_n/(a+1)}%")

    # Place by eval
    elif eval_type == "place by":
        ACTIONS = 350
        start = 0

        if exists(join(location, "models", "eval_info", "place_by", f"{name}_ob.npy")):
            ob_results = np.load(join(location, "models", "eval_info", "place_by", f"{name}_ob.npy"))
            col_results = np.load(join(location, "models", "eval_info", "place_by", f"{name}_col.npy"))
            tot_n = np.load(join(location, "models", "eval_info", "place_by", f"{name}_tot.npy"))
            start = int(np.sum(ob_results))
        else:
            ob_results = np.zeros((len(OBJECT_TYPES) + 2, 3))
            col_results = np.zeros((len(COLOURS), 3))
            tot_n = np.zeros(1)

        # object_keys_place_on =

        print("\n===========\nPLACE BY EVAL\n===========\n")
        for a in range(start, ACTIONS):
            print(f"\n---------\nACTION {a+1}/{ACTIONS}")
            arm.reset()
            arm.straight_ee()

            for object in objects+plates+bins:
                object.object.remove()
            objects = create_objects(1, 6, low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.02, current_obj=[])
            
            if randint(0,2):
                plates = create_plates(1, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.02, size=0.1, current_obj=objects)
                bins = create_bins(0, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.05, size=0.1, current_obj=objects+plates)
            else:      
                bins = create_bins(1, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.05, size=0.1, current_obj=objects)
                plates = create_plates(0, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.02, size=0.1, current_obj=objects+bins)

            # Grab random object
            while len(arm.get_gripped()) == 0:
                arm.open_gripper()
                grabbed_object = choice(objects)
                grab_object(arm, grabbed_object)

            # Go to random pose
            start = np.random.uniform(low=[-0.05, -0.05, 0.1], high=[0.05, 0.05, 0.3], size=3)
            arm.reset()
            arm.move_by_smooth(start)
            arm.straight_ee()

            # Execute test
            target_object = choice(plates + [object for object in objects if object.object != grabbed_object.object] + bins)
            if randint(0, 2):
                action = choice(place_by_sentences_non_inc).format(choice(articles), target_object.description)
            else:
                action = choice(place_by_sentences_inc).format(choice(articles), grabbed_object.description, choice(articles), target_object.description)

            print("Action: Place by", target_object.description)
            if need_encode:
                perform_action_encode_action(arm, model, action, objects)
            else:
                perform_action(arm, model, [action], objects)

            held_placed_location = grabbed_object.object.get_position()
            target_object_location = target_object.object.get_position()

            if ((held_placed_location[2] - grabbed_object.size[2]/2)  > (target_object_location[2] + target_object.size[2]/2) and
                np.linalg.norm(held_placed_location - target_object_location) < np.sqrt(sum([i**2 for i in target_object.size[:2]])) / 2 + 0.01):
                # Failure: On top
                ob_results[object_keys[target_object.type], 1] += 1
                col_results[colour_keys[target_object.colour_name], 1] += 1
                print("Failure: On top")

            elif np.linalg.norm(held_placed_location[:2] - target_object_location[:2]) < (np.sqrt(sum([i**2 for i in target_object.size[:2]]) + 0.07)):
                # Success
                tot_n += 1
                ob_results[object_keys[target_object.type], 0] += 1
                col_results[colour_keys[target_object.colour_name], 0] += 1
                print("Success")

            else:
                # Failure: Wrong object
                ob_results[object_keys[target_object.type], 2] += 1
                col_results[colour_keys[target_object.colour_name], 2] += 1
                print("Failure: Wrong Object")

            with open(join(location, "models", "eval_info", "place_by", f"{name}_ob.npy"), 'wb') as f:
                np.save(f, ob_results)
            with open(join(location, "models", "eval_info", "place_by", f"{name}_col.npy"), 'wb') as f:
                np.save(f, col_results)
            with open(join(location, "models", "eval_info", "place_by", f"{name}_tot.npy"), 'wb') as f:
                np.save(f, tot_n)
            
            print(f"TOTAL ACCURACY: {100*tot_n/(a+1)}%")


    # Place in
    elif eval_type == "place in":
        ACTIONS = 50
        start = 0

        if exists(join(location, "models", "eval_info", "place_in", f"{name}_ob.npy")):
            ob_results = np.load(join(location, "models", "eval_info", "place_in", f"{name}_ob.npy"))
            col_results = np.load(join(location, "models", "eval_info", "place_in", f"{name}_col.npy"))
            tot_n = np.load(join(location, "models", "eval_info", "place_in", f"{name}_tot.npy"))
            start = int(np.sum(ob_results))
        else:
            ob_results = np.zeros((1, 3))
            col_results = np.zeros((len(COLOURS), 3))
            tot_n = np.zeros(1)

        object_keys_place_in = {"bin": 0}

        print("\n===========\nPLACE IN EVAL\n===========\n")
        for a in range(start, ACTIONS):
            print(f"\n---------\nACTION {a+1}/{ACTIONS}")
            arm.reset()
            arm.straight_ee()


            for object in objects+plates+bins:
                object.object.remove()
            
            bins = create_bins(1, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.05, size=0.1, current_obj=[])
            objects = create_objects(1, 5, low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.02, current_obj=bins)
            plates = create_plates(0, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.02, size=0.1, current_obj=objects+bins)

            # Grab random object
            while len(arm.get_gripped()) == 0:
                arm.open_gripper()
                grabbed_object = choice(objects)
                grab_object(arm, grabbed_object)

            # Go to random pose
            start = np.random.uniform(low=[-0.05, -0.05, 0.1], high=[0.05, 0.05, 0.3], size=3)
            arm.reset()
            arm.move_by_smooth(start)
            arm.straight_ee()

            # Execute test

            target_object = choice(bins)
            if randint(0, 2):
                action = choice(place_in_sentences_non_inc).format(choice(articles), target_object.description)
            else:
                action = choice(place_in_sentences_inc).format(choice(articles), grabbed_object.description, choice(articles), target_object.description)

            print("Action: Place in", target_object.description)
            if need_encode:
                perform_action_encode_action(arm, model, action, objects)
            else:
                perform_action(arm, model, [action], objects)
            
            held_placed_location = grabbed_object.object.get_position()
            target_object_location = target_object.object.get_position()

            if (np.linalg.norm(held_placed_location[:2] - target_object_location[:2]) < np.sqrt(sum([i**2 for i in target_object.size[:2]])) / 2 + 0.01):
                # Success
                ob_results[object_keys_place_in[target_object.type], 0] += 1
                col_results[colour_keys[target_object.colour_name], 0] += 1
                tot_n += 1
                print("Success")

            elif np.linalg.norm(held_placed_location[:2] - target_object_location[:2]) < (np.sqrt(sum([i**2 for i in target_object.size[:2]]) / 2 + 0.07)):
                # Failure: To side
                ob_results[object_keys_place_in[target_object.type], 1] += 1
                col_results[colour_keys[target_object.colour_name], 1] += 1
                print("Failure: To side")

            else:
                # Failure: Wrong object
                ob_results[object_keys_place_in[target_object.type], 2] += 1
                col_results[colour_keys[target_object.colour_name], 2] += 1
                print("Failure: Wrong Object")

            with open(join(location, "models", "eval_info", "place_in", f"{name}_ob.npy"), 'wb') as f:
                np.save(f, ob_results)
            with open(join(location, "models", "eval_info", "place_in", f"{name}_col.npy"), 'wb') as f:
                np.save(f, col_results)
            with open(join(location, "models", "eval_info", "place_in", f"{name}_tot.npy"), 'wb') as f:
                np.save(f, tot_n)
            
            print(f"TOTAL ACCURACY: {100*tot_n/(a+1)}%")