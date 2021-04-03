import pickle

import cv2
import numpy as np
import imageio

from learning_agents.lfd.demo_preprocessor.base_preprocessor import Base_Demo_Preprocessor
from learning_agents.utils.trajectory_utils import extract_transitions_from_indexed_trajs


def get_training_testing_data(demo_path, state_processor=None, train_test_split=1.0, n_training_sample=None):
    """
    Split the demo data into training set and testing set.
        If n_training_sample is specified, then use the first n_training_sample samples from the dataset
    """
    state_processor = state_processor if state_processor else Base_Demo_Preprocessor()
    with open(demo_path, 'rb') as f:
        data = pickle.load(f)
    transitions = extract_transitions_from_indexed_trajs([data])

    n_sample = len(transitions)
    if n_training_sample is not None:
        testing_idx = set([i for i in range(n_training_sample, n_sample)])
    else:
        testing_idx = set(list(np.random.choice(n_sample, max(0, int(n_sample * (1 - train_test_split))), replace=False)))
    training_data = []
    testing_data = []

    for i, transition in enumerate(transitions):
        processed_transition = state_processor.get_processed_transition(transition)
        if i in testing_idx:
            testing_data.append(processed_transition)
        else:
            training_data.append(processed_transition)
    print('[INFO] demo_utils -',
          'Done splitting training and testing demo: {0} training samples, {1} testing samples'.format(
              len(training_data), len(testing_data)))
    return training_data, testing_data


def img_to_video(rgb_images, fname, fps=6):
    fname = fname + '.mp4'
    imageio.mimwrite(fname, rgb_images, fps=fps)


def imgs_add_annotation(rgb_images, bounding_boxes, box_rescale=1.0):
    assert len(rgb_images) == len(bounding_boxes)
    for i in range(len(rgb_images)):
        rgb_img = rgb_images[i]
        for j in range(len(bounding_boxes[i])):
            rMin, rMax, cMin, cMax = bounding_boxes[i][j]
            if box_rescale is not None:
                rMin, rMax, cMin, cMax = rMin * box_rescale, rMax * box_rescale, cMin * box_rescale, cMax * box_rescale
            rgb_img = cv2.rectangle(rgb_img, (int(cMin), int(rMin)), (int(cMax), int(rMax)),
                                    color=(255, 0, 0), thickness=1)
        rgb_images[i] = rgb_img
    return rgb_images


def main():
    pass


if __name__ == '__main__':
    main()
