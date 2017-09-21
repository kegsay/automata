package automata

// Network represents an arbitrary artificial neural network.
type Network struct {
	Input  []Layer
	Hidden []Layer
	Output []Layer
}

func (n *Network) Activate(input Neuron) {

}

// Propagate the error through the entire network.
//
// This is subject to the Vanishing Gradient problem. As errors propagate, they will be multiplied by
// the weights which are typically a fraction between 0-1. This will rapidly diminish the error value
// as the error continues to propagate, and can be exacerbated depending on the squashing function used,
// making earlier layers difficult to train.
func (n *Network) Propagate() {}

// TODO:
// - Gate
// - Project
