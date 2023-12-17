import time
import numpy as np
from random import randint
from os.path import join, dirname, realpath

import numpy as np

from .arm import SimPanda
from .bert_encoder import bert_encode
from .create_sim_objects import create_objects, create_plates

import torch


from .models import ResNet18FiLM, ResNet18FiLMBert

def generate_new_objects(objects, plates):
    for object in objects+plates:
        object.object.remove()

    objects=[]
    plates = []

    while len(objects) == 0 or len(plates) == 0:
        for object in objects+plates:
            object.object.remove()

        objects=[]
        plates = []

        objects = create_objects(1, 6, low=[0.9, -0.25, 0.775], high=[1.2, 0.25, 0.775], gap=0.02, current_obj=[])
        plates = create_plates(1, 3,  low=[0.9, -0.25, 0.775], high=[1.2, 0.25, 0.775], gap=0.02, size=0.1, current_obj=objects)

    return objects, plates


def perform_action(arm, model, text_inp, objects=[], maxsteps=50, verbose=False):
    # Order move then rotate
    move=True
    for i in range(maxsteps):
        # Get image
        im, d = arm.get_im_d()
        im = np.moveaxis(im, -1, 0)
        d = np.expand_dims(d, axis=0)
        inp = np.concatenate((d, im), axis=0)
        inp = torch.tensor(inp).view(1, 4, 128, 128).float()

        # Get movement and rotation from model
        # delta = model.forward(inp, text_inp).detach().numpy()[0]
        delta = model.forward(inp, text_inp).detach().numpy()[0]
        delta_mov, delta_rot = delta[:3], delta[3]
        if i % 2:
            print(f".", end="\r", flush=True)
        else:
            print(f" ", end="\r", flush=True)
        # Move/rotate
        if np.linalg.norm(delta_mov) > 0.004 and move:
            if not arm.move_by_smooth(delta_mov):
                break
            if verbose:
                print("Move", delta_mov)
        else:
            move=False
            if abs(delta_rot) < 0.005:
                break
            arm.rotate_gripper_by([0, 0, delta_rot])
            if verbose:
                print("Rot", delta_rot)

    # Grab the object
    if len(arm.get_gripped()) == 0:
        arm.down_grasp(by=0.12, objects=[object.object for object in objects])
        if len(arm.get_gripped()) == 0:
            arm.open_gripper()
    else:
        arm.open_gripper()

    arm.reset_gripper()
    arm.go_home()

def perform_action_encode_action(arm, model, action, objects=[], maxsteps=50, verbose=False):
    target_enc = bert_encode(action, output="sentence").numpy()
    enc = torch.tensor(target_enc).unsqueeze(0).float()
    perform_action(arm, model, enc, objects=objects, maxsteps=maxsteps, verbose=verbose)


if __name__ == "__main__":
    # Load simulator
    location = dirname(realpath(__file__))
    SCENE_FILE = join(location, 'scenes', 'can_scene_cube.ttt')
    arm = SimPanda(SCENE_FILE, bottom=0.76)

    # Load Model
    MODEL_PATH = join(location, "models", "weights")
    model = ResNet18FiLMBert(c_in=4, c_out=4, bert_init=join(location, "models", "weights", "bert", "base.model"))
    model.load_state_dict(torch.load(join(MODEL_PATH, "full_policy", "full_policy_pre_bert.model"), map_location=torch.device('cpu')))
    model.eval()

    # Set up initial scene
    objects, plates = generate_new_objects(objects=[], plates=[])
    arm.reset_gripper()
    arm.go_home()
    
    # Main loop
    ACTIONS = 100
    for a in range(ACTIONS):
        print("\n---------\nACTION", a)

        # Find a random place to go
        start = np.random.uniform(low=[-0.1, -0.1, 0.1], high=[0.1, 0.1, 0.3], size=3)
        # Go to random pose
        arm.move_by_smooth(start)
        arm.straight_ee()
        # Get action input ("new" gives new objects, "done" ends simulation)
        action = input("Action:")
        while action == "new":
            objects, plates = generate_new_objects(objects, plates)
            arm.pr.step()
            action = input("Action:")
        
        if action == "done":
            break
        
        #perform_action_encode_action(arm, model, action, objects=objects, verbose=True)
        perform_action(arm, model, [action], objects=objects, verbose=True)

    arm.shutdown()