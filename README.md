# Large Language Models as Planners for Robots

This repository holds the code and report for my final year project at ICL.
This project investigates how Large Language Models can be used to decompose high level actions into a set of instructions that can be run on a robot.

This project runs a robot simulated in [CoppeliaSim](https://www.coppeliarobotics.com/) and communicates with it using [PyRep](https://github.com/stepjam/PyRep).
The robot is controlled by a model trained to take in camera input + instruction text and output movement and rotation information.
See report for more details and outcome.

Demo:

[![Youtube](https://github.com/Ruari-Phipps/llm-planners/assets/113997551/d2eb2f67-ba09-429e-aeee-39759114c33e)](https://www.youtube.com/watch?v=qTuC5kixCEY)

The repository contains the code to gather training date, train and run the model.
The model weights have been omitted due to their large size.
