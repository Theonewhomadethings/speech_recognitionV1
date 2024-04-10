# Probabilistic Speech Recognition System

This project aims to develop a simple speech recognizer in MATLAB that uses Hidden Markov Models (HMMs) with a small vocabulary. 

We initialise, train and test a set of Hidden Markov Models (HMMs) with one for each of the key words in the vocabulary. Each model is an 8-state HMM with 13-dimensional continious probability density functions, N=8 and K = 13.

For feature extraction we use 13 Mel-Frequency cepstral coefficients including the zeroth coefficient. 

For model initialisation we compute the global mean and variance across the whole dev set. 

For training we apply the Baum-Welch equations to re-estimate the models. 

For monitoring the training process on dev set and for evaluation we use the Viterbi algorithm as the basis of the decoder. 


This project performs isolated word recognition by training HMMs according to the Expecation-Maximisation method with the Baum-Welch equations and runs the recognizer by decoding with the Viterbi Algorithm.
