package automata

var neuronCount = 0

// Neuron represents a base unit of work in a neural network.
type Neuron struct {
	ID                  int64
	State               int64
	Old                 int64
	Activation          float64
	Bias                float64
	Inputs              ConnMap
	Projected           ConnMap
	Gated               ConnMap
	ErrorResponsibility float64
	ErrorProjected      float64
	ErrorGated          float64
}

func NewNeuron() Neuron {
	return Neuron{}
}

func (n *Neuron) Activate(input *Neuron) float64 {
	return 0 // TODO
}

// Propagate an error through this neuron. 'rate' is the learning rate for this neuron, target is set if this neuron
// forms part of an output layer, otherwise is nil.
func (n *Neuron) Propagate(rate float64, target *float64) {
	isOutput := target != nil

	if isOutput {
		// output neurons get their error from the environment
		n.ErrorResponsibility = *target - n.Activation
		n.ErrorProjected = *target - n.Activation
	} else {
		// compute errors from back-propagation

	}

	n.learn(rate)
}

func (n *Neuron) Project(neu *Neuron, weights []float64) Connection {
	return NewConnection(*n, *neu, weights[0]) // TODO
}

// learn by adjusting this neuron's weight.
func (n *Neuron) learn(rate float64) {

}
