# Quadcopter Modeling, Identification, and Control with Neural Networks

This project collects progressive experiments on quadcopter modeling and PID imitation using neural networks. Each `Cxx` folder is a milestone that shows how the approach evolved.

## Requirements
- Python 3.10+ (tested with 3.13)
- numpy
- matplotlib
- torch
- scikit-learn

## Structure
- [C45](C45/) baseline linear Z-axis PID simulation, GRU training, and GRU controller test.
- [C46](C46/) narrowed to a single multi-step reference, adds a test-only script that loads a trained GRU and compares it against the fixed PID without retraining.
- `Cxx/` other milestones (to be added).

## Notes
- Each `Cxx` folder is standalone and documents a specific stage.
