package automata

// Network represents an arbitrary artificial neural network.
type Network struct {
	Input  *Layer
	Hidden []Layer
	Output *Layer
}

// Activate the network with the given neuron, feeding forward to produce an output.
func (n *Network) Activate(input []float64) ([]float64, error) {
	n.Input.Activate(input)
	for _, layer := range n.Hidden {
		layer.Activate(nil)
	}
	return n.Output.Activate(nil)
}

// Propagate the error through the entire network.
//
// This is subject to the Vanishing Gradient problem. As errors propagate, they will be multiplied by
// the weights which are typically a fraction between 0-1. This will rapidly diminish the error value
// as the error continues to propagate, and can be exacerbated depending on the squashing function used,
// making earlier layers difficult to train.
func (n *Network) Propagate(rate float64, target []float64) error {
	err := n.Output.Propagate(rate, target)
	if err != nil {
		return err
	}
	for i := len(n.Hidden) - 1; i >= 0; i-- {
		err = n.Hidden[i].Propagate(rate, nil)
		if err != nil {
			return err
		}
	}
	return nil
}

func (n *Network) ProjectLayer(layer *Layer, ltype LayerType) {
	n.Output.Project(layer, ltype)
}

func (n *Network) ProjectNetwork(network *Network, ltype LayerType) {
	n.Output.Project(network.Input, ltype)
}

// TODO:
// - Gate
