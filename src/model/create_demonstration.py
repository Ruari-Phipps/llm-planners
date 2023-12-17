from termios import INPCK
import time
import numpy as np
from numpy.random import uniform, choice, randint
from os import remove
from os.path import join, dirname, realpath
import glob
import pickle

import numpy as np

from arm import SimPanda
from bert_encoder import bert_encode
from create_sim_objects import create_bins, create_objects, create_plates

from action_sentences import place_by_sentences_inc, place_by_sentences_non_inc, place_on_sentences_inc, place_on_sentences_non_inc, place_in_sentences_inc, place_in_sentences_non_inc,grab_sentences, articles 

####################
## LOAD SIMULATOR ##
####################

location = dirname(realpath(__file__))
SCENE_FILE = join(location, 'scenes', 'can_scene_cube.ttt')

REPLAYS = 180
NEW_REPLAYS = False

arm = SimPanda(SCENE_FILE, bottom=0.76, headless=True, vision_cam="wrist_cam")


##################
## CREATE DEMOS ##
##################

def delete_replays(loc):
    files = glob.glob(join(loc, "*"))
    for f in files:
        remove(f)

# Delete old replays
if NEW_REPLAYS:
    delete_replays(join(location, "data", "full_policy", "inputs"))
    delete_replays(join(location, "data", "full_policy", "actions"))
    delete_replays(join(location, "data", "full_policy", "text_encoding"))
    delete_replays(join(location, "data", "full_policy", "raw_text"))

global_step_policy = 170 #len(glob.glob(join(location, "data", "full_policy", "actions", "*.npy")))
print("GLOBAL STEP", global_step_policy)
start_time = time.time()

# Main loop
STEPS = 6
objects=[]
plates = []
bins = []
for r in range(global_step_policy, REPLAYS):
    print(f"\n---------\nREPLAY {r}/{REPLAYS}")
    # First go home
    arm.reset_gripper()
    arm.go_home()
    arm.reset()

    # If any objects or plates, delete
    for object in objects+plates+bins:
            object.object.remove()

    objects= []
    plates = []
    bins = []

    # Each scene must have at least one object and plate
    while len(objects) == 0:
        for object in objects+plates+bins:
            object.object.remove()

        objects=[]
        plates = []
        bins = []

        objects = create_objects(2, 5, low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.02, current_obj=[])
        
        if randint(0,2):
            plates = create_plates(1, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.02, size=0.1, current_obj=objects)
            bins = create_bins(0, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.05, size=0.1, current_obj=objects+plates)
        else:      
            bins = create_bins(1, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.05, size=0.1, current_obj=objects)
            plates = create_plates(0, 3,  low=[0.9, -0.2, 0.775], high=[1.2, 0.2, 0.775], gap=0.02, size=0.1, current_obj=objects+bins)
    # objects = create_objects(1, 6, low=[0.9, -0.15, 0.775], high=[1.2, 0.15, 0.775], gap=0.02, current_obj=[])

    # First target object to pick up
    target_obj = choice(objects)
    target_description = target_obj.description
    target_enc = bert_encode(target_description, output="sentence").numpy()


    target_pos = target_obj.object.get_position()
    target_ori = -target_obj.object.get_orientation()

    print("Action: Pick up", target_description)

    target = target_pos + np.array([0.005, 0, target_obj.size[2]/2 + 0.05])  #Select a target position above the object of interest.

    # Find a random place to go
    start = np.random.uniform(low=[-0.1, -0.1, 0.1], high=[0.2, 0.1, 0.2], size=3)
    # Go to random pose
    arm.move_by_smooth(start)
    arm.straight_ee()

    # Episode data
    episode_imgs = []
    episode_actions = []
    episode_encodings = []
    episode_raw_text = []

    mov = True
    for i in range(70):
        # Calculate movement
        current_pos = arm.get_position()
        step = (target - current_pos)/STEPS
        distance_step = np.linalg.norm(step)
        delta_target = step

        if target_obj.type in ["sphere", "cylinder"]:
            diff = 0
        else:
            current_ori = arm.get_gripper_orientation()
            rot = (target_ori - current_ori) / STEPS
            diff = rot[2]

        action = np.append(delta_target, diff)

        # Save state and movement
        im, d = arm.get_im_d()
        im = np.moveaxis(im, -1, 0)
        d = np.expand_dims(d, axis=0)
        inp = np.concatenate((d, im), axis=0)

        if i % 2:
            print(f".", end="\r", flush=True)
        else:
            print(f" ", end="\r", flush=True)


        # Randomly create encoded sentence
        if i % 5 == 0:
            action_name = choice(grab_sentences).format(choice(articles), target_description)
            target_enc = bert_encode(action_name, output="sentence").numpy()

        episode_imgs.append(inp)
        episode_actions.append(action)
        episode_encodings.append(target_enc)
        episode_raw_text.append(action_name)

        # Stop?
        if distance_step > 0.002 and mov:
            arm.move_by_smooth(delta_target)
        else:
            mov = False
            if abs(diff) < 0.001:
                break 

            arm.rotate_gripper_by([0, 0, diff])

    # Grab object
    arm.down_grasp(by=0.1, objects=[target_obj.object])

    # Return to start
    arm.reset_gripper()
    arm.go_home()

    # Find a random place to go
    start = np.random.uniform(low=[-0.1, -0.1, 0.1], high=[0.1, 0.1, 0.2], size=3)
    # Go to random pose
    arm.move_by_smooth(start)
    arm.straight_ee()

    # If object gripped?
    if len(arm.get_gripped()) > 0:

        # Choose put down skill
        options = ["next"]
        if len(plates + [object for object in objects if object.type in ["cube", "slice"] and object != target_obj]) > 0:
            options.append("on")
        if len(bins) > 0:
            options.append("in")
        option = choice(options)

        if option == "on":
            # Select an object
            tar_inc = place_on_sentences_inc
            tar_non_inc = place_on_sentences_non_inc

            on_objects = plates + [object for object in objects if object.type in ["cube", "slice"] and object != target_obj]
            target_place = choice(on_objects)

            print("Action: Place on", target_place.description)
            # Target move place is above target object (include height of held object in calcuation)
            target_pos = target_place.object.get_position()
            target = target_pos + np.array([0.01, 0, target_place.size[2] / 2 + (target_obj.size[2] - 0.04) + 0.03])

        elif option == "next":
            tar_inc = place_by_sentences_inc
            tar_non_inc = place_by_sentences_non_inc

            # Get a possible targets
            next_objects = plates + [object for object in objects if object != target_obj] + bins

            # Get maximum width of object
            held_obj_width = np.sqrt(sum([i**2 for i in target_obj.size[:2]]))

            # Loop until a space is found next to a target
            got_target = False
            while not got_target:
                # Choose target
                target_place = choice(next_objects)
                target_pos = target_place.object.get_position()
                # Target Z location is on floor
                target_pos[2] = 0.775

                # Select a random direction below the object (So in cam the object appears in frame)
                angle = np.pi #uniform(np.pi/2, 3 * np.pi/2)

                # Get maximum width of object placing next to
                place_width = np.sqrt(sum([i**2 for i in target_place.size[:2]]))

                # Find wanted distance from held object to target object
                gap = place_width/2 + held_obj_width/2 + 0.01

                # Try 5 different locations
                for i in range(5):
                    # Get target location
                    target = target_pos + np.array([gap * np.cos(angle), gap * np.sin(angle), (target_obj.size[2] - 0.04) + 0.03])
                    # Is in range of robot?
                    if 0.9 < target[0] < 1.2 and -0.3 < target[1] < 0.3:
                        got_target = True
                        # Check whether other objects are too close
                        for obj in [object for object in next_objects if object != target_place]:
                            obj_width = np.sqrt(sum([i**2 for i in obj.size[:2]]))
                            if np.linalg.norm(obj.location - target) < (0.02 + held_obj_width/2 + obj_width/2):
                                got_target = False
                                break
            
                        if got_target:
                            break
                    # If object in way move angle round
                    angle += np.pi/5 % (2*np.pi)
                    if angle < np.pi / 2 or angle > 3 * np.pi / 2:
                        angle += np.pi
            
            print("Action: Place by", target_place.description)
        
        elif option == "in":
            # Select an object
            tar_inc = place_in_sentences_inc
            tar_non_inc = place_in_sentences_non_inc

            target_place = choice(bins)

            print("Action: Place in", target_place.description)
            # Target move place is above target object (include height of held object in calcuation)
            target_pos = target_place.object.get_position()
            target = target_pos + np.array([0.01, 0, target_place.size[2] / 2 + (target_obj.size[2] + 0.03)])

        # Find a random place to go
        start = np.random.uniform(low=[-0.05, -0.05, 0.1], high=[0.05, 0.05, 0.2], size=3)
        # Go to random pose
        arm.move_by_smooth(start)
        arm.straight_ee()

        for i in range(30):
            # Calculate movement
            current_pos = arm.get_position()
            step = (target - current_pos)/STEPS
            distance_step = np.linalg.norm(step)
            delta_target = step

            diff = 0

            action = np.append(delta_target, diff)

            # # Save state and movement
            im, d = arm.get_im_d()
            im = np.moveaxis(im, -1, 0)
            d = np.expand_dims(d, axis=0)
            inp = np.concatenate((d, im), axis=0)

            if i % 5 == 0:
                if randint(0, 2):
                    action_name = choice(tar_non_inc).format(choice(articles), target_place.description)
                else:
                    action_name = choice(tar_inc).format(choice(articles), target_description, choice(articles), target_place.description)
                target_enc = bert_encode(action_name, output="sentence").numpy()

            if i % 2:
                print(f".", end="\r", flush=True)
            else:
                print(f" ", end="\r", flush=True)


            episode_imgs.append(inp)
            episode_actions.append(action)
            episode_encodings.append(target_enc)
            episode_raw_text.append(action_name)

            # Stop?
            if distance_step > 0.002:
                arm.move_by_smooth(delta_target)
            else:
                break

        # Save episode info
        with open(join(location, "data", "full_policy", "inputs", "{}.npy".format(global_step_policy)), 'wb') as f:
            np.save(f, np.array(episode_imgs))
        with open(join(location, "data", "full_policy", "actions", "{}.npy".format(global_step_policy)), 'wb') as f:
            np.save(f, np.array(episode_actions))
        with open(join(location, "data", "full_policy", "text_encoding", "{}.npy".format(global_step_policy)), 'wb') as f:
            np.save(f, np.array(episode_encodings))
        with open(join(location, "data", "full_policy", "raw_text", "{}.ob".format(global_step_policy)), 'wb') as f:
            pickle.dump(episode_raw_text, f)
        global_step_policy += 1

    arm.open_gripper()

print("TOT TIME", time.time() - start_time)

arm.shutdown()