# Reinforcement Learning for Real-Time Autonomous Decision-Making in Simulated Environment

## Overview

This project implements a Reinforcement Learning (RL) agent capable of making autonomous decisions in a simulated environment based on the traditional Tigers and Goats (Huligutta) strategy game. The agent learns optimal actions through interaction with the environment using the Q-Learning algorithm.

## Objective

The primary objective of this project is to demonstrate how Reinforcement Learning can be used for real-time decision-making by enabling an intelligent agent to learn winning strategies without explicit programming.

## Features

* Implementation of Tabular Q-Learning.
* Autonomous decision-making agent.
* Real-time learning through environment interaction.
* Reward-based optimization.
* Performance evaluation using multiple metrics.
* Strategy visualization and analysis.

## Technologies Used

* Python
* Reinforcement Learning
* Q-Learning Algorithm
* NumPy
* Matplotlib
* Jupyter Notebook

## Methodology

1. Designed the Tigers and Goats game environment.
2. Defined states, actions, and reward structure.
3. Implemented the Q-Learning algorithm.
4. Trained the agent for 10,000 episodes.
5. Evaluated performance using classification metrics and win-rate analysis.

## Results

* Agent convergence achieved after approximately 4,500 training episodes.
* Win rate improved from approximately 10% to 88%.
* Accuracy: 92%
* Precision: 90%
* Recall: 93%
* F1-Score: 91%
* Specificity: 97%

## Project Structure

```text
├── src/
├── dataset/
├── models/
├── results/
├── README.md
└── requirements.txt
```

## Future Enhancements

* Deep Q-Network (DQN) implementation.
* Multi-agent reinforcement learning.
* Graphical User Interface (GUI).
* Deployment as a web application.
* Integration with larger strategic game environments.

## Research Publication

This project was developed as part of the research work titled:

"Reinforcement Learning for Real-Time Autonomous Decision-Making in Simulated Environment"

Authors:

* Bandari Nikshitha
* Chevvakula Srisir
* Chitla Sathwika

## Conclusion

The project successfully demonstrates the effectiveness of Reinforcement Learning in autonomous decision-making scenarios. The trained agent learned optimal strategies and achieved significant performance improvements through continuous interaction with the environment.
