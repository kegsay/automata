[![GoDoc](https://godoc.org/github.com/Kegsay/automata?status.svg)](https://godoc.org/github.com/Kegsay/automata) [![Build Status](https://travis-ci.org/Kegsay/automata.svg?branch=master)](https://travis-ci.org/Kegsay/automata) [![Coverage Status](https://coveralls.io/repos/github/Kegsay/automata/badge.svg)](https://coveralls.io/github/Kegsay/automata)

*Disclaimer: This project is currently a work in progress. As such, the API has not been finalised and breaking changes are frequent.*

This project can be used to create neural networks with varying architectures. It has built-in support for LSTMs, Perceptrons and Hopfield networks. You can check the tests or GoDoc to find out more.

### Aims

The aim of this project is to implement a generalised neural network library in pure Go.

The goal is to allow for people to experiment and discover new ways to design neural networks. As such, the focus is on:
 - Clarity: People with very basic knowledge on ANNs should be able to use this library.
 - Extensibility: It should be easy to add new layers / activation functions / etc. With this in mind, this project is
   based off this excellent [paper](http://www.overcomplete.net/papers/nn2012.pdf).

Whilst the following are considered, they are not the core goals of this project. If they contend with the core goals of this project,
they will lose out:
 - Speed: Parallelisation is not built-in or considered upfront.
 - Memory consumption: Neurons are modelled as actual `structs` rather than matrix math which results in much higher memory overheads.
