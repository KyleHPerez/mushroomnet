# mushroomnet
Mushroomnet is a machine learning neural network for classifying mushrooms as poisonous or edible. The model was trained on data from uni.edu: over 5,000 instances of mushrooms, each with 22 qualitative features and a determination of poisonous or edible. 

ATTENTION! This API is for research purposes only and is NOT to be used for food-safety or medical advice.

GENERAL DESCRIPTION
The code here was written by myself (Kyle Hunter Perez) on 8 June 2025 as an experiment in learning neural network (NN) machine learning (ML) / artificial intelligence (AI) with Python. I used ChatGPT (version 4o) to first prepare a basic understanding of relevant topics in Python and the math underpinning those operations. I then directed and supervised ChatGPT 4o to incrementally build code in line with my vision for a simple NN model trained on high-fidelity, under-utilizied data (here, the data set of Agraricus lepiota measurements from UCI.edu) implemented with PyTorch, scikit-learn, pandas, and numpy. I directed code adjustments until a fully functional and deployable API running on Flask and reachable with curl was complete. In the process, I gained hands-on experience working with Python for NN ML/AI, using conventional libraries in that field, deploying the product service, and familiarity with how the code works and the mathematics that run under-the-hood.

This repository contains support files for running mushroomnet as a containerized API with Docker.
To run mushroomnet inside a Docker container:
1. (If you haven't already) download and install Docker https://docs.docker.com/desktop/setup/install/mac-install/
2. Change your working directory to the folder with the mushroomnet source code files (e.g. cd /home/users/me/Documents/mushroomnet )
3. From your command line, docker build -t mushroom-api .
4. Then, also from your command line, docker run -p 8000:5000 mushroom-api
5. Now you have a container running mushroomnet that you can reach at 127.0.0.1:8000
6. (Optional) test the API by sending a sample request:
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": {"cap-shape": "x", "cap-surface": "s", "cap-color": "n", "bruises": "t", "odor": "p", "gill-attachment": "f", "gill-spacing": "c", "gill-size": "n", "gill-color": "k", "stalk-shape": "e", "stalk-root": "e", "stalk-surface-above-ring": "s", "stalk-surface-below-ring": "s", "stalk-color-above-ring": "w", "stalk-color-below-ring": "w", "veil-type": "p", "veil-color": "w", "ring-number": "o", "ring-type": "p", "spore-print-color": "k", "population": "s", "habitat": "u"}}'

Enjoy!
And remember, mushroomnet is only for research/entertainment purposes. DO NOT use mushroomnet for food-safety or medical advice.


NEXT STEPS
Hosting the dockerized api on a cloud provider (Heroku or AWS ECS).
Build React front-end for cloud-hosted mushroomnet API container.

LINKS
Source for mushroom dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data
