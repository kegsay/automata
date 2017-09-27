package automata

import "math/rand"

type NeuronID int64

var neuronCount = 0

func neuronUID() NeuronID {
	neuronCount += 1
	return NeuronID(neuronCount)
}

// Neuron represents a base unit of work in a neural network.
type Neuron struct {
	ID         NeuronID
	Old        float64
	State      float64
	Derivative float64
	Activation float64
	Self       *Connection
	Squash     Squasher
	Bias       float64
	Neighbours map[NeuronID]*Neuron

	Inputs    ConnMap
	Projected ConnMap
	Gated     ConnMap

	ErrorResponsibility float64
	ErrorProjected      float64
	ErrorGated          float64

	TraceEligibility map[ConnID]float64
	TraceExtended    map[NeuronID]map[ConnID]float64
	TraceInfluences  map[NeuronID]map[ConnID]Connection
}

func NewNeuron() Neuron {
	n := Neuron{
		ID:               neuronUID(),
		Squash:           &SquashLogistic{},
		Bias:             (rand.Float64() / 2) - 0.25, // Bias range from -0.25 ~ 0.25 initially
		Neighbours:       make(map[NeuronID]*Neuron),
		Inputs:           make(ConnMap),
		Projected:        make(ConnMap),
		Gated:            make(ConnMap),
		TraceEligibility: make(map[ConnID]float64),
		TraceExtended:    make(map[NeuronID]map[ConnID]float64),
		TraceInfluences:  make(map[NeuronID]map[ConnID]Connection),
	}
	w := float64(0)
	n.Self = NewConnection(&n, &n, &w) // 0 weight means unconnected
	return n
}

// Activate this neuron with an optional input.
//
// The logic in this function is based on "A generalized LSTM-like training algorithm for second-order recurrent neural networks"
// See: http://www.overcomplete.net/papers/nn2012.pdf
func (n *Neuron) Activate(input *float64) float64 {
	// check for activation from the environment
	if input != nil {
		n.Activation = *input
		n.Derivative = 0
		n.Bias = 0
		return n.Activation
	}

	n.Old = n.State
	// Eq. 15
	n.State = n.Self.Gain*n.Self.Weight*n.State + n.Bias
	for _, input := range n.Inputs {
		n.State += input.From.Activation * input.Weight * input.Gain
	}

	// Eq. 16
	n.Activation = n.Squash.Squash(n.State, false)

	// f'(s)
	n.Derivative = n.Squash.Squash(n.State, true)

	// update traces
	influences := make(map[NeuronID]float64)
	for neuronID := range n.TraceExtended {
		neuron := n.Neighbours[neuronID]
		var influence float64
		if neuron.Self.Gater == n {
			influence = neuron.Old
		}

		for _, incoming := range n.TraceInfluences[neuron.ID] {
			influence += incoming.Weight * incoming.From.Activation
		}
		influences[neuron.ID] = influence
	}

	for _, input := range n.Inputs {
		// Eq. 17: eligibility trace
		n.TraceEligibility[input.ID] = n.Self.Gain*n.Self.Weight*n.TraceEligibility[input.ID] + input.Gain*input.From.Activation

		for neuronID := range n.TraceExtended {
			xtrace := n.TraceExtended[neuronID]
			neuron := n.Neighbours[neuronID]
			influence := influences[neuronID]

			// Eq. 18
			xtrace[input.ID] = neuron.Self.Gain*neuron.Self.Weight*xtrace[input.ID] + n.Derivative*n.TraceEligibility[input.ID]*influence
		}
	}

	// Update gated connection gains
	for connID, conn := range n.Gated {
		conn.Gain = n.Activation
		n.Gated[connID] = conn
	}

	return n.Activation
}

// Propagate an error through this neuron. 'rate' is the learning rate for this neuron, target is set if this neuron
// forms part of an output layer, otherwise is nil.
//
// The logic in this function is based on "A generalized LSTM-like training algorithm for second-order recurrent neural networks"
// See: http://www.overcomplete.net/papers/nn2012.pdf
func (n *Neuron) Propagate(rate float64, target *float64) {
	isOutput := target != nil

	if isOutput {
		// Eq. 10: output neurons get their error from the environment
		n.ErrorResponsibility = *target - n.Activation
		n.ErrorProjected = *target - n.Activation
	} else {
		// Eq. 21: error responsibilities from all the connections projected from this neuron
		var accumulatedError float64
		for _, conn := range n.Projected {
			accumulatedError += conn.To.ErrorResponsibility * conn.Gain * conn.Weight
		}

		n.ErrorProjected = n.Derivative * accumulatedError

		// TODO: Eq. 22: gated error responsibility
		n.ErrorGated = 0

		n.ErrorResponsibility = n.ErrorProjected + n.ErrorGated
	}

	n.learn(rate)
}

func (n *Neuron) Project(targetNeuron *Neuron, weight *float64) *Connection {
	if targetNeuron == n {
		n.Self.Weight = 1 // make connection live (1 = connected)
		return n.Self
	}

	// check if this connection already exists
	conn := n.Projected.getConnectionForNeuron(targetNeuron)
	if conn != nil {
		if weight != nil {
			conn.Weight = *weight
		}
		return conn
	} else {
		conn = NewConnection(n, targetNeuron, weight)

		// reference this connection and traces
		n.Projected[conn.ID] = conn
		n.Neighbours[targetNeuron.ID] = targetNeuron
		targetNeuron.Inputs[conn.ID] = conn
		targetNeuron.TraceEligibility[conn.ID] = 0
		for nID := range n.TraceExtended {
			trace := n.TraceExtended[nID]
			trace[conn.ID] = 0
		}
		return conn
	}
}

// ConnectionForNeuron returns the connection between the two neurons or nil if there is no connection.
func (n *Neuron) ConnectionForNeuron(target *Neuron) *Connection {
	c := n.Projected.getConnectionForNeuron(target)
	if c != nil {
		return c
	}
	c = n.Inputs.getConnectionForNeuron(target)
	if c != nil {
		return c
	}
	return n.Gated.getConnectionForNeuron(target)
}

// learn by adjusting weights.
func (n *Neuron) learn(rate float64) {
	for connID, conn := range n.Inputs {
		// Eq. 24
		gradient := n.ErrorProjected * n.TraceEligibility[conn.ID]
		for neuronID := range n.TraceExtended {
			neuron := n.Neighbours[neuronID]
			gradient += neuron.ErrorResponsibility * n.TraceExtended[neuronID][conn.ID]
		}

		conn.Weight += rate * gradient
		n.Inputs[connID] = conn
	}

	n.Bias += rate * n.ErrorResponsibility
}