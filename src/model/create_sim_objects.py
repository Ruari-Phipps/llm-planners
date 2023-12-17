

from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

import numpy as np
from numpy.random import choice, randint, uniform

import math

from regex import R

COLOURS = {
    "white": [1.0, 1.0, 1.0],
    "grey": [0.5, 0.5, 0.5],
    "black": [0.0, 0.0, 0.0],
    "red": [1.0, 0.1, 0.1],
    "green": [0.1, 0.7, 0.1],
    "blue": [0.1, 0.1, 1.0],
    "yellow": [1.0, 1.0, 0.1],
    # "pink": [1.0, 0.1, 1.0],
    # "aqua": [0.1, 1.0, 1.0],
    # "orange": [1.0, 0.5, 0.1],
    # "purple": [0.5, 0.0, 1.0],
#     "light blue": [0.0, 0.8, 1.0],
#     "light green": [0.5, 1.0, 0.0],
#     "dark green": [0.0, 0.2, 0.0],
#     "dark blue": [0.0, 0.0, 0.5],
#     "dark red": [0.4, 0.0, 0.0],
#     "dark purple": [0.2, 0, 0.5],
#     "dark grey": [0.8, 0.8, 0.8],
#     "light purple": [0.8, 0.5, 0.8],
#     "light grey": [0.4, 0.4, 0.4],
}


OBJECT_TYPES = [
    {
        "type": "cylinder",
        "shape": PrimitiveShape.CYLINDER,
        "size": [0.04, 0.04, 0.05],
        "names": ["cylinder"] #, "tube", "can", "tin"]
    },
    {
        "type": "cube",
        "shape": PrimitiveShape.CUBOID,
        "size": [0.05, 0.05, 0.05],
        "names": ["cube"]#, "block"]
    },
    {
        "type": "box",
        "shape": PrimitiveShape.CUBOID,
        "size": [0.03, 0.05, 0.06],
        "names": ["box"]#, "cuboid"]
    },
    {
        "type": "sphere",
        "shape": PrimitiveShape.SPHERE,
        "size": [0.05, 0.05, 0.05],
        "names": ["sphere"]#, "ball", "orb", "globe"]
    },
    {
        "type": "slice",
        "shape": PrimitiveShape.CUBOID,
        "size": [0.05, 0.05, 0.02],
        "names": ["slice"]
    }
]

class SimObject:
    def __init__(self, type, shape, size, location, colour, colour_name, description, object=None) -> None:
        if object is not None:
            self.object = object
            self.object.set_position(location)
            self.object.set_color(colour)
        else:
            self.object = Shape.create(type=shape,
                        size=size,
                        position=location,
                        color=colour)
        self.type = type
        self.colour_name = colour_name
        self.description = description
        self.size = size
        self.location = location


def create_objects(min_objects, max_objects, low, high, gap, current_obj=[]):
    colours_free = list(COLOURS.keys()) # List of available colours, objects won't have the same colour
    objects = [] # Objects produced
    num = randint(min_objects, max_objects) # Produce a random nuymber of objects

    for i in range(num):
        # Try (for max of 50 times) to place an object in area with (gap)
        # distance to all other objects
        placed = False
        for j in range(50):
            # Choose random object and scalar for it
            object_type = choice(OBJECT_TYPES)
            size_multiplier = uniform(0.8, 1.2)
            size = [size_multiplier * x for x in object_type['size']]
            obj_loc = np.random.uniform(low=low, high=high, size=3)

            max_width = math.sqrt(sum([i**2 for i in size[:2]]))
            # Try to place this object
            placed = True
            for obj in objects+current_obj:
                obj_width = math.sqrt(sum([i**2 for i in obj.size[:2]]))
                if np.linalg.norm(obj.location - obj_loc) < (gap + max_width/2 + obj_width/2):
                    placed = False
            if placed:
                break
        if not placed:
            break

        # Choose random available colour and description
        colour = choice(colours_free)
        colours_free.remove(colour)
        desc = f"{colour} {choice(object_type['names'])}"

        # Create this object
        object = SimObject(object_type["type"], object_type["shape"], size, obj_loc, COLOURS[colour], colour, desc)

        # Give it a random rotation
        random_rot = uniform(-np.pi/4, np.pi / 4)
        object.object.set_orientation(np.array([0, 0, random_rot]))

        objects.append(object)

    return objects

def create_plates(min_objects, max_objects, low, high, gap, size, current_obj=[]):
    # Create this object
    colours_free = list(COLOURS.keys()) # List of available colours, objects won't have the same colour
    objects = [] # Objects produced
    num = randint(min_objects, max_objects) # Produce a random nuymber of objects
    
    for i in range(num):
        # Try (for max of 50 times) to place an object in area with (gap)
        # distance to all other objects
        placed = False
        for j in range(50):
            obj_loc = np.random.uniform(low=low, high=high, size=3)
            placed = True
            max_width = math.sqrt(size**2 * 2)
            # Try to place this object
            placed = True
            for obj in objects+current_obj:
                obj_width = math.sqrt(sum([i**2 for i in obj.size[:2]]))
                if np.linalg.norm(obj.location - obj_loc) < (gap + max_width/2 + obj_width/2):
                    placed = False
            if placed:
                break
        if not placed:
            break


        # Choose random available colour
        colour = choice(colours_free)
        colours_free.remove(colour)
        # Create this object
        desc = f"{colour} plate"
        colour_code = [min(max(x + uniform(-0.15, 0.15), 0.0), 1.0) for x in COLOURS[colour]]
        
        object = SimObject("plate", PrimitiveShape.CUBOID, [size, size, 0.005], obj_loc, colour_code, colour, desc)

        # Give it a random location
        random_rot = uniform(0, np.pi / 2)
        object.object.set_orientation(np.array([0, 0, random_rot]))

        objects.append(object)

    return objects

def create_bins(min_objects, max_objects, low, high, gap, size, current_obj=[]):
    # Create this object
    colours_free = list(COLOURS.keys()) # List of available colours, objects won't have the same colour
    objects = [] # Objects produced
    num = randint(min_objects, max_objects) # Produce a random nuymber of objects
    
    for i in range(num):
        # Try (for max of 50 times) to place an object in area with (gap)
        # distance to all other objects
        placed = False
        for j in range(50):
            obj_loc = np.random.uniform(low=low, high=high, size=3)
            placed = True
            max_width = math.sqrt(size**2 * 2)
            # Try to place this object
            placed = True
            for obj in objects+current_obj:
                obj_width = math.sqrt(sum([i**2 for i in obj.size[:2]]))
                if np.linalg.norm(obj.location - obj_loc) < (gap + max_width/2 + obj_width/2):
                    placed = False
            if placed:
                break
        if not placed:
            break


        # Choose random available colour
        colour = choice(colours_free)
        colours_free.remove(colour)
        # Create this object
        desc = f"{colour} bin"
        colour_code = [min(max(x + uniform(-0.05, 0.05), 0.0), 1.0) for x in COLOURS[colour]]

        obj = Shape("Bin").copy() 
        object = SimObject("bin", PrimitiveShape.CUBOID, [size, size, size], obj_loc, colour_code, colour, desc, object=obj)

        # Give it a random location
        # random_rot = uniform(0, np.pi / 2)
        # object.object.set_orientation(np.array([0, 0, random_rot]))

        objects.append(object)

    return objects


def create_given_objects(object_texts, low, high, gap):
    objects = []
    for i in range(10):
        for object in objects:
            object.object.remove()

        objects = [] # Objects produced

        for object_text in object_texts:
            object_text = object_text.split(" ")
            i = 0
            if len(object_text) > 2:
                ob_size = object_text[i].lower()
                i+=1
            else:
                ob_size = None
            colour = object_text[i].lower()
            if colour == "random":
                colour = choice(list(COLOURS.keys()))
            object_type = object_text[i+1].lower()

            if object_type == "bin":
                size = [0.1, 0.1, 0.1]
            elif object_type == "plate":
                size = [0.1, 0.1, 0.005]
            else:
                object = [obj for obj in OBJECT_TYPES if obj["type"] == object_type]
                if len(object) == 0:
                    print("NOT FOUND")
                    return None
                object = object[0]
                if ob_size == "small":
                    size_multiplier = 0.85
                elif ob_size == "large":
                    size_multiplier = 1.15
                else:
                    size_multiplier = uniform(0.8, 1.2)
                size = [size_multiplier * x for x in object['size']]
    
            placed = False
            for j in range(50):
                # Choose random object and scalar for it
                obj_loc = np.random.uniform(low=low, high=high, size=3)

                max_width = math.sqrt(sum([i**2 for i in size[:2]]))
                # Try to place this object
                placed = True
                for obj in objects:
                    obj_width = math.sqrt(sum([i**2 for i in obj.size[:2]]))
                    if np.linalg.norm(obj.location - obj_loc) < (gap + max_width/2 + obj_width/2):
                        placed = False
                if placed:
                    break
            if not placed:
                break

            # Choose random available colour and description

            if object_type == "bin":
                desc = f"{colour} bin"
                colour_code = [min(max(x + uniform(-0.05, 0.05), 0.0), 1.0) for x in COLOURS[colour]]

                obj = Shape("Bin").copy() 
                object = SimObject("bin", PrimitiveShape.CUBOID, size, obj_loc, colour_code, colour, desc, object=obj)
            elif object_type == "plate":
                desc = f"{colour} plate"
                colour_code = [min(max(x + uniform(-0.15, 0.15), 0.0), 1.0) for x in COLOURS[colour]]
                
                object = SimObject("plate", PrimitiveShape.CUBOID, size, obj_loc, colour_code, colour, desc)
            else:
                desc = f"{colour} {choice(object['names'])}"

                # Create this object
                object = SimObject(object["type"], object["shape"], size, obj_loc, COLOURS[colour], colour, desc)

            # Give it a random rotation
            random_rot = uniform(-np.pi/4, np.pi / 4)

            if object_type != "bin":
                object.object.set_orientation(np.array([0, 0, random_rot]))

            objects.append(object)

        if len(object_texts) == len(objects):
            return objects
    print("NOOOO")
    return None