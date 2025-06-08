# mushroomnet
Mushroomnet is a machine learning neural network for classifying mushrooms as poisonous or edible. The model was trained on data from uni.edu: over 5,000 instances of mushrooms, each with 22 qualitative features and a determination of poisonous or edible. 

ATTENTION! This API is for research purposes only and is NOT to be used for food-safety or medical advice.

GENERAL DESCRIPTION
The code here was written by myself (Kyle Hunter Perez) on 8 June 2025 as an experiment in learning neural network (NN) machine learning (ML) / artificial intelligence (AI) with Python. I used ChatGPT (version 4o) to first prepare a basic understanding of relevant topics in Python and the math underpinning those operations. I then directed and supervised ChatGPT 4o to incrementally build code in line with my vision for a simple NN model trained on high-fidelity, under-utilizied data (here, the data set of Agraricus lepiota measurements from UCI.edu) implemented with PyTorch, scikit-learn, pandas, and numpy. I directed code adjustments until a fully functional and deployable API running on Flask and reachable with curl was complete. In the process, I gained hands-on experience working with Python for NN ML/AI, using conventional libraries in that field, deploying the product service, and familiarity with how the code works and the mathematics that run under-the-hood.

NEXT STEPS
The next phase of this project will be containerizing the flask_app.py API to run platform-independently on Docker. From there, I plan on hosting the dockerized api on a cloud provider (Heroku or AWS ECS) and connecting it with a React-based front-end.
