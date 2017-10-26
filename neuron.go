package automata

import (
	"math/rand"
)

type NeuronID int64

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
	Neighbours []NeuronID

	Inputs    []ConnID
	Projected []ConnID
	Gated     []ConnID

	ErrorResponsibility float64
	ErrorProjected      float64
	ErrorGated          float64

	TraceEligibility []float64 // Efficient form of map[ConnID]float64
	TraceExtended    map[NeuronID]map[ConnID]float64
	TraceInfluences  map[NeuronID][]ConnID

	LookupTable *LookupTable
}

func NewNeuron(table *LookupTable) *Neuron {
	n := Neuron{
		Squash:          &SquashLogistic{},
		Bias:            (rand.Float64() / 2) - 0.25, // Bias range from -0.25 ~ 0.25 initially
		TraceExtended:   make(map[NeuronID]map[ConnID]float64),
		TraceInfluences: make(map[NeuronID][]ConnID),
		LookupTable:     table,
	}
	id := table.SetNeuron(&n)
	n.ID = id
	w := float64(0)
	n.Self = NewConnection(&n, &n, &w) // 0 weight means unconnected
	return &n
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
		// fmt.Println(n.ID, " Activate INPUT NEURON => ", *input)
		return n.Activation
	}

	n.Old = n.State
	// Eq. 15
	n.State = n.Self.Gain*n.Self.Weight*n.State + n.Bias
	for _, inputCID := range n.Inputs {
		input := n.LookupTable.GetConnection(inputCID)
		n.State += input.From.Activation * input.Weight * input.Gain
	}

	// Eq. 16
	n.Activation = n.Squash.Squash(n.State, false)

	// f'(s)
	n.Derivative = n.Squash.Squash(n.State, true)

	// update traces
	influences := make(map[NeuronID]float64)
	for neuronID := range n.TraceExtended {
		neuron := n.LookupTable.GetNeuron(neuronID)
		var influence float64
		if neuron.Self.Gater == n {
			influence = neuron.Old
		}

		for _, cid := range n.TraceInfluences[neuron.ID] {
			incoming := n.LookupTable.GetConnection(cid)
			influence += incoming.Weight * incoming.From.Activation
		}
		influences[neuron.ID] = influence
	}

	for _, inputCID := range n.Inputs {
		input := n.LookupTable.GetConnection(inputCID)
		// Eq. 17: eligibility trace
		val := n.Self.Gain*n.Self.Weight*n.getTraceEligibility(input.ID) + input.Gain*input.From.Activation
		n.setTraceEligibility(input.ID, val)

		for neuronID := range n.TraceExtended {
			xtrace := n.TraceExtended[neuronID]
			neuron := n.LookupTable.GetNeuron(neuronID)
			influence := influences[neuronID]

			// Eq. 18
			xtrace[input.ID] = neuron.Self.Gain*neuron.Self.Weight*xtrace[input.ID] + n.Derivative*n.getTraceEligibility(input.ID)*influence
			n.TraceExtended[neuronID] = xtrace
		}
	}

	// Update gated connection gains
	for _, connID := range n.Gated {
		conn := n.LookupTable.GetConnection(connID)
		conn.Gain = n.Activation
		n.LookupTable.SetConnectionWithID(connID, conn)
	}
	//fmt.Println(n.ID, " Activate => ", n.Activation, "old=", n.Old, " state=", n.State)
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
		for _, connID := range n.Projected {
			conn := n.LookupTable.GetConnection(connID)
			accumulatedError += conn.To.ErrorResponsibility * conn.Gain * conn.Weight
		}

		n.ErrorProjected = n.Derivative * accumulatedError

		accumulatedError = 0
		for nid := range n.TraceExtended {
			var influence float64
			neuron := n.LookupTable.GetNeuron(nid) // gated neuron
			if neuron.Self.Gater == n {
				influence = neuron.Old
			}
			for _, cid := range n.TraceInfluences[nid] {
				conn := n.LookupTable.GetConnection(cid)
				influence += conn.Weight * conn.From.Activation
			}
			// Eq. 22 gated error responsibility
			accumulatedError += neuron.ErrorResponsibility * influence
		}
		n.ErrorGated = n.Derivative * accumulatedError

		// Eq. 23
		n.ErrorResponsibility = n.ErrorProjected + n.ErrorGated
	}

	n.learn(rate)
}

func (n *Neuron) Project(targetNeuron *Neuron, weight *float64) *Connection {
	if targetNeuron == n {
		// fmt.Println("PROJECT: self", n.ID)
		n.Self.Weight = 1 // make connection live (1 = connected)
		return n.Self
	}

	// check if this connection already exists
	conn := n.getConnectionForNeuron(n.Projected, targetNeuron)
	if conn != nil {
		// fmt.Println("PROJECT: Already found (", n.ID, "to", targetNeuron.ID, ")")
		if weight != nil {
			conn.Weight = *weight
		}
		return conn
	}
	conn = NewConnection(n, targetNeuron, weight)

	// reference this connection and traces
	n.Projected = append(n.Projected, conn.ID)
	n.Neighbours = append(n.Neighbours, targetNeuron.ID)
	targetNeuron.Inputs = append(targetNeuron.Inputs, conn.ID)
	targetNeuron.setTraceEligibility(conn.ID, 0)
	for nID := range n.TraceExtended {
		trace := n.TraceExtended[nID]
		trace[conn.ID] = 0
	}
	// fmt.Println("PROJECT: hooked ", n.ID, "to", targetNeuron.ID)
	return conn
}

func (n *Neuron) Gate(conn *Connection) {
	n.Gated = append(n.Gated, conn.ID)
	if _, ok := n.TraceExtended[conn.To.ID]; !ok {
		n.Neighbours = append(n.Neighbours, conn.To.ID)
		n.TraceExtended[conn.To.ID] = make(map[ConnID]float64)
		for _, inputCID := range n.Inputs {
			input := n.LookupTable.GetConnection(inputCID)
			n.TraceExtended[conn.To.ID][input.ID] = 0
		}
	}

	if n.TraceInfluences[conn.To.ID] == nil {
		n.TraceInfluences[conn.To.ID] = make([]ConnID, 1)
	}
	arr := n.TraceInfluences[conn.To.ID]
	arr = append(arr, conn.ID)
	n.TraceInfluences[conn.To.ID] = arr
	conn.Gater = n
}

// ConnectionForNeuron returns the connection between the two neurons or nil if there is no connection.
func (n *Neuron) ConnectionForNeuron(target *Neuron) *Connection {
	if target == n && n.Self.Weight != 0 {
		return target.Self
	}

	if c := n.getConnectionForNeuron(n.Projected, target); c != nil {
		return c
	}
	if c := n.getConnectionForNeuron(n.Inputs, target); c != nil {
		return c
	}
	return n.getConnectionForNeuron(n.Gated, target)
}

// learn by adjusting weights.
func (n *Neuron) learn(rate float64) {
	for _, connID := range n.Inputs {
		conn := n.LookupTable.GetConnection(connID)
		// Eq. 24
		gradient := n.ErrorProjected * n.getTraceEligibility(conn.ID)
		for neuronID := range n.TraceExtended {
			neuron := n.LookupTable.GetNeuron(neuronID)
			gradient += neuron.ErrorResponsibility * n.TraceExtended[neuronID][conn.ID]
		}

		conn.Weight += rate * gradient
		n.LookupTable.SetConnectionWithID(connID, conn)
	}

	n.Bias += rate * n.ErrorResponsibility
}

func (n *Neuron) getConnectionForNeuron(cidList []ConnID, target *Neuron) *Connection {
	for _, cid := range cidList {
		conn := n.LookupTable.GetConnection(cid)
		if conn.From == target || conn.To == target {
			return conn
		}
	}
	return nil
}

func (n *Neuron) setTraceEligibility(id ConnID, val float64) {
	if int(id) > (len(n.TraceEligibility) - 1) {
		diff := int(id) - (len(n.TraceEligibility) - 1)
		n.TraceEligibility = append(n.TraceEligibility, make([]float64, diff)...)
	}
	n.TraceEligibility[id] = val
}

func (n *Neuron) getTraceEligibility(id ConnID) float64 {
	return n.TraceEligibility[id]
}
