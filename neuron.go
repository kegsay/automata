package automata

var neuronCount = 0

// Neuron represents a base unit of work in a neural network.
type Neuron struct {
	ID         int64
	State      int64
	Old        int64
	Activation int64
	Bias       float64
	Inputs     ConnMap
	Projected  ConnMap
	Gated      ConnMap
}

func NewNeuron() Neuron {
	return Neuron{}
}

func (n *Neuron) Activate(input *Neuron) float64 {
	return 0
}
