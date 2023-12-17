from os.path import dirname, realpath, join
import torch
import numpy as np

location = dirname(realpath(__file__))
MODEL_PATH = join(location, "models", "weights")

from .train_bert_model_classifier import BertTrain, COLOURS, OBJECT_TYPES

class Failure_checker:
    def __init__(self, classifier_path) -> None:
        self.classifier = BertTrain()
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.eval()

        self.colour_map = {}
        for i in range(len(COLOURS)):
            self.colour_map[i] = list(COLOURS.keys())[i]

        self.object_map = {}
        for i in range(len(OBJECT_TYPES)):
            self.object_map[i] = OBJECT_TYPES[i]["type"]
        self.object_map[i+1] = "plate"
        self.object_map[i+2] = "bin"

    def check_failure(self, input, arm, objects, held_object):
        a, c, o = self.classifier([input])
        a = torch.argmax(a).item()
        c = torch.argmax(c).item()
        o = torch.argmax(o).item()

        #Grab
        if a == 0:
            return self._check_gripper(arm, objects, c, o)
        # Place by
        elif a == 1:
            return self._check_place_by(objects, c, o, held_object)
        # Place on
        elif a == 2:
            return self._check_place_on(objects, c, o, held_object)
        # Place in
        elif a == 3:
            return self._check_place_in(objects, c, o, held_object)
            

    def _check_gripper(self, arm, objects, c, o):
        if len(arm.get_gripped()) == 0:
            return False, "Nothing Grabbed"

        grabbed = arm.get_gripped()[0]
        object = [object for object in objects if object.object == grabbed]

        if len(object) == 0:
            return True, "Could not find"

        object = object[0]

        if object.type != self.object_map[o] or object.colour_name != self.colour_map[c]:
            return False, "Wrong item"
        
        return True, "Correct"


    def _check_place_by(self, objects, c, o, held_object):

        if held_object is None:
            return False, "Nothing placed"

        possible_items = [obj for obj in objects if obj.type == self.object_map[o] and obj.colour_name == self.colour_map[c]]

        if len(possible_items) == 0:
            return True, "Could not find"

        for item in possible_items:
            item_locaiton = item.object.get_position()
            held_object_location = held_object.get_position()

            print(np.linalg.norm(held_object_location[:2] - item_locaiton[:2]) - (np.sqrt(sum([i**2 for i in item.size[:2]]))/2))

            if 0 < (np.linalg.norm(held_object_location[:2] - item_locaiton[:2]) - (np.sqrt(sum([i**2 for i in item.size[:2]]))/2)  < 0.08):
                return True, "Success"

        return False, "Wrong location"

    def _check_place_on(self, objects, c, o, held_object):

        if held_object is None:
            return False, "Nothing placed"

        possible_items = [obj for obj in objects if obj.type == self.object_map[o] and obj.colour_name == self.colour_map[c]]

        if len(possible_items) == 0:
            return True, "Could not find"

        
        for item in possible_items:
            item_locaiton = item.object.get_position()
            held_object_location = held_object.get_position()

            if ((held_object_location[2])  > (item_locaiton[2] + item.size[2]/2) and
                    np.linalg.norm(item_locaiton[:2] - item_locaiton[:2]) < np.sqrt(sum([i**2 for i in item.size[:2]])) / 2 + 0.02):
                return True, "Success"

        return False, "Wrong location"


    def _check_place_in(self, objects, c, o, held_object):

        if held_object is None:
            return False, "Nothing placed"

        possbile_bins = [obj for obj in objects if obj.type == self.object_map[o] and obj.colour_name == self.colour_map[c]]

        if len(possbile_bins) == 0:
            return True, "Could not find"

        for bin in possbile_bins:
            bin_locaiton = bin.object.get_position()
            held_object_location = held_object.get_position()

            if (np.linalg.norm(held_object_location[:2] - bin_locaiton[:2]) < np.sqrt(sum([i**2 for i in bin.size[:2]])) / 2 + 0.01):
                return True, "Success"

        return False, "Wrong location"
